use std::{
  ffi::c_void,
  marker::PhantomData,
  ops::Deref,
  ptr::{self, NonNull},
};

use ash::vk;
use mem_type_assignment::{
  MemoryAssignmentError, UnassignedToMemoryObjectsData,
  assign_memory_type_indexes_to_objects_for_allocation,
};

mod create_objs;
#[cfg(feature = "log_alloc")]
mod logging;
mod mapped_host_obj;
mod mem_type_assignment;
mod memory_bound;
mod staging_buffers;

pub use mapped_host_obj::{HostMemorySyncError, MappedHostBuffer, MappedHostImage};
pub use memory_bound::MemoryBound;
pub use staging_buffers::{
  DeviceMemoryInitializationError, SingleUseStagingBuffers, create_single_use_staging_buffers,
};

#[allow(unused_imports)]
#[cfg(feature = "log_alloc")]
pub use logging::{debug_print_device_memory_info, debug_print_possible_memory_type_assignment};
use vkinitialization::device::{Device, PhysicalDevice};
use vkobjects::{
  DeviceManuallyDestroyed,
  errors::OutOfMemoryError,
  utility::{self, OnErr},
};

use crate::mapped_host_obj::MappedHostObject;

#[derive(Debug, Default, Clone, Copy)]
pub struct DetailedMemory {
  pub memory: vk::DeviceMemory,
  pub type_index: usize,
  pub size: u64,
}

impl Deref for DetailedMemory {
  type Target = vk::DeviceMemory;

  fn deref(&self) -> &Self::Target {
    &self.memory
  }
}

impl DeviceManuallyDestroyed for DetailedMemory {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    unsafe {
      self.memory.destroy_self(device);
    }
  }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryPlacement {
  pub memory_index: usize,
  pub memory_offset: vk::DeviceSize,
  pub object_size: vk::DeviceSize,
}

#[derive(Debug, Clone, Copy)]
pub struct AllocationSuccess<const S: usize> {
  pub memories: [DetailedMemory; vk::MAX_MEMORY_TYPES],
  pub memory_count: usize,
  // memory index, offset
  pub obj_to_memory_assignment: [MemoryPlacement; S],
}

impl<const S: usize> AllocationSuccess<S> {
  pub fn get_memories(&self) -> &[DetailedMemory] {
    &self.memories[0..self.memory_count]
  }
}

impl<const S: usize> DeviceManuallyDestroyed for AllocationSuccess<S> {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    unsafe {
      self.get_memories().destroy_self(device);
    }
  }
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum AllocationError {
  #[error("Object to memory type assignment did not succeed")]
  AssignmentError(#[from] MemoryAssignmentError),
  #[error(transparent)]
  OutOfMemoryError(#[from] OutOfMemoryError),
  #[error("Vulkan returned VK_ERROR_UNKNOWN")]
  Unknown,
  #[error("Vulkan returned VK_ERROR_VALIDATION_FAILED")]
  ValidationFailed,
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum HostAllocationError {
  #[error("Not all memory properties have the HOST_VISIBLE bit set")]
  NoHostVisibleBit,
  #[error(transparent)]
  MemoryMapError(#[from] MemoryMapError),
  #[error(transparent)]
  AllocationError(#[from] AllocationError),
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum MemoryMapError {
  #[error("Vulkan returned VK_ERROR_MEMORY_MAP_FAILED")]
  MemoryMapFailed,
  #[error(transparent)]
  OutOfMemoryError(#[from] OutOfMemoryError),
  #[error("Vulkan returned VK_ERROR_UNKNOWN")]
  Unknown,
  #[error("Vulkan returned VK_ERROR_VALIDATION_FAILED")]
  ValidationFailed,
}

impl From<vk::Result> for MemoryMapError {
  fn from(value: vk::Result) -> Self {
    match value {
      vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
        Self::OutOfMemoryError(OutOfMemoryError::from(value))
      }
      vk::Result::ERROR_MEMORY_MAP_FAILED => Self::MemoryMapFailed,
      vk::Result::ERROR_UNKNOWN => Self::Unknown,
      vk::Result::ERROR_VALIDATION_FAILED_EXT => Self::ValidationFailed,
      _ => panic!("Unhandled vk::Result when converting to MemoryMapError"),
    }
  }
}

fn is_power_of_2(x: u64) -> bool {
  (x & (x - 1)) == 0
}

// todo: not checking heap capacity, maxMemoryAllocationSize
// (https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_memory_overallocation_behavior.html)
pub fn allocate_memory<const P: usize, const S: usize>(
  device: &Device,
  physical_device: &PhysicalDevice,
  mem_props: [vk::MemoryPropertyFlags; P],
  objs: [&dyn MemoryBound; S],
  priority: f32, // only set if VK_EXT_memory_priority is enabled
  align_objects_to_non_coherent_atom_size: bool,
  #[cfg(feature = "log_alloc")] obj_labels: Option<[&'static str; S]>,
  #[cfg(feature = "log_alloc")] allocation_name: &str,
) -> Result<AllocationSuccess<S>, AllocationError> {
  let mem_types = physical_device.memory_types();
  let mut obj_reqs = unsafe { objs.map(|obj| obj.get_memory_requirements(device)) };

  for req in obj_reqs {
    debug_assert!(is_power_of_2(req.alignment));
  }
  if align_objects_to_non_coherent_atom_size {
    let atom_size = physical_device.properties.p10.limits.non_coherent_atom_size;
    debug_assert!(is_power_of_2(atom_size));
    for req in obj_reqs.iter_mut() {
      req.alignment = req.alignment.max(atom_size);
      req.size = utility::round_up_to_power_of_2_u64(req.size, atom_size);
    }
  }

  let assign_result =
    assign_memory_type_indexes_to_objects_for_allocation(UnassignedToMemoryObjectsData {
      mem_types,
      mem_props,
      obj_reqs,
      #[cfg(feature = "log_alloc")]
      obj_labels,
    });

  #[cfg(feature = "log_alloc")]
  {
    let labels: Option<&[&str]> = match &obj_labels {
      Some(labels_arr) => Some(labels_arr.as_slice()),
      None => None,
    };
    let mut output = String::new();
    logging::display_mem_assignment_result::<P, S>(
      &mut output,
      assign_result,
      allocation_name,
      mem_types,
      &mem_props,
      &obj_reqs,
      labels,
    )
    .unwrap();
    let _ = output.pop(); // remove last \n
    if assign_result.is_ok() {
      log::debug!("{}", output);
    } else {
      log::error!("{}", output);
    }
  }

  let (assigned, unique_type_ixs_count) = assign_result?;

  let mut working_memory_types = [usize::MAX; vk::MAX_MEMORY_TYPES];
  let mut working_memory_types_size = 0;
  for type_i in assigned {
    if !working_memory_types[0..working_memory_types_size].contains(&type_i) {
      working_memory_types[working_memory_types_size] = type_i;
      working_memory_types_size += 1;
    }
  }
  debug_assert_eq!(working_memory_types_size, unique_type_ixs_count);

  // mem index, offset
  let mut allocation_result = [MemoryPlacement {
    memory_index: usize::MAX,
    memory_offset: u64::MAX,
    object_size: 0,
  }; S];
  let mut memories = [DetailedMemory {
    memory: vk::DeviceMemory::null(),
    type_index: usize::MAX,
    size: 0,
  }; vk::MAX_MEMORY_TYPES];
  for (mem_i, &type_i) in working_memory_types[0..working_memory_types_size]
    .iter()
    .enumerate()
  {
    let mut total_size = 0;
    for i in 0..S {
      if assigned[i] == type_i {
        let offset: u64 = utility::round_up_to_power_of_2_u64(total_size, obj_reqs[i].alignment);
        total_size = offset + obj_reqs[i].size;
        allocation_result[i].memory_offset = offset;
        allocation_result[i].object_size = obj_reqs[i].size;
      }
    }

    // todo: this probably would require dividing the allocations
    assert!(total_size <= physical_device.properties.p11.max_memory_allocation_size);

    let mut allocate_info = vk::MemoryAllocateInfo {
      s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
      p_next: ptr::null(),
      allocation_size: total_size,
      memory_type_index: type_i as u32,
      _marker: PhantomData,
    };
    let priority_info = vk::MemoryPriorityAllocateInfoEXT::default().priority(priority);
    if device.enabled_extensions.memory_priority {
      allocate_info.p_next =
        &priority_info as *const vk::MemoryPriorityAllocateInfoEXT as *const c_void;
    }
    let memory = unsafe { device.allocate_memory(&allocate_info, None) }
      .on_err(|_| unsafe {
        for &mem in memories[0..mem_i].iter() {
          mem.destroy_self(device);
        }
      })
      .map_err(|vkerr| AllocationError::OutOfMemoryError(OutOfMemoryError::from(vkerr)))?;
    if let Some(loader) = device.pageable_device_local_memory_loader.as_ref() {
      unsafe {
        // set manually priority as well for the pageable_device_local_memory extension
        // only raw function pointers for now
        (loader.fp().set_device_memory_priority_ext)(device.handle(), memory, priority);
      }
    }
    memories[mem_i] = DetailedMemory {
      memory,
      type_index: type_i,
      size: total_size,
    };

    for i in 0..S {
      if assigned[i] == type_i {
        allocation_result[i].memory_index = mem_i;
      }
    }
  }

  Ok(AllocationSuccess {
    memories,
    memory_count: working_memory_types_size,
    obj_to_memory_assignment: allocation_result,
  })
}

pub fn allocate_and_bind_memory<const P: usize, const S: usize>(
  device: &Device,
  physical_device: &PhysicalDevice,
  mem_props: [vk::MemoryPropertyFlags; P],
  objs: [&dyn MemoryBound; S],
  priority: f32, // only set if VK_EXT_memory_priority is enabled
  align_objects_to_non_coherent_atom_size: bool,
  #[cfg(feature = "log_alloc")] obj_labels: Option<[&'static str; S]>,
  #[cfg(feature = "log_alloc")] allocation_name: &str,
) -> Result<AllocationSuccess<S>, AllocationError> {
  let alloc = allocate_memory(
    device,
    physical_device,
    mem_props,
    objs,
    priority,
    align_objects_to_non_coherent_atom_size,
    #[cfg(feature = "log_alloc")]
    obj_labels,
    #[cfg(feature = "log_alloc")]
    allocation_name,
  )?;
  let memories = alloc.get_memories();

  for (i, &placement) in alloc.obj_to_memory_assignment.iter().enumerate() {
    unsafe {
      objs[i]
        .bind(
          device,
          *memories[placement.memory_index],
          placement.memory_offset,
        )
        .on_err(|_| alloc.destroy_self(device))?;
    }
  }

  Ok(alloc)
}

pub fn allocate_and_map_host_memory<const P: usize, const S: usize>(
  device: &Device,
  physical_device: &PhysicalDevice,
  mem_props: [vk::MemoryPropertyFlags; P],
  objs: [&dyn MemoryBound; S],
  priority: f32, // only set if VK_EXT_memory_priority is enabled
  #[cfg(feature = "log_alloc")] obj_labels: Option<[&'static str; S]>,
  #[cfg(feature = "log_alloc")] allocation_name: &str,
) -> Result<(AllocationSuccess<S>, [MappedHostObject; S]), HostAllocationError> {
  for prop in mem_props {
    if !prop.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
      return Err(HostAllocationError::NoHostVisibleBit);
    }
  }

  let alloc = allocate_and_bind_memory(
    device,
    physical_device,
    mem_props,
    objs,
    priority,
    true,
    #[cfg(feature = "log_alloc")]
    obj_labels,
    #[cfg(feature = "log_alloc")]
    allocation_name,
  )?;

  let mapped_ptrs = map_host_visible_allocation(device, &alloc)?;
  let objects = MappedHostObject::from_allocation(physical_device, &alloc, &objs, mapped_ptrs);

  Ok((alloc, objects))
}

/// allocation must have had the HOST_VISIBLE property flag
pub fn map_host_visible_allocation<const S: usize>(
  device: &Device,
  allocation: &AllocationSuccess<S>,
) -> Result<[NonNull<u8>; S], MemoryMapError> {
  let mut pointers = [NonNull::dangling(); S];
  let mut assigned_count = 0;
  for (cur_memory_i, memory) in allocation.get_memories().iter().enumerate() {
    let mem_ptr =
      unsafe { device.map_memory(**memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }?
        as *mut u8;

    for (
      obj_i,
      &MemoryPlacement {
        memory_index: obj_memory_i,
        memory_offset,
        object_size: _object_size,
      },
    ) in allocation.obj_to_memory_assignment.iter().enumerate()
    {
      if cur_memory_i == obj_memory_i {
        pointers[obj_i] = NonNull::new(unsafe { mem_ptr.byte_add(memory_offset as usize) })
          .expect("Vulkan device.map_memory() returned a null pointer");
        assigned_count += 1;
      }
    }
    assert_eq!(assigned_count, S);
  }

  Ok(pointers)
}

#[cfg(test)]
mod tests {
  // use super::*;
}
