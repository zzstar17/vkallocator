use std::{ops::BitOr, ptr::copy_nonoverlapping};

use ash::vk;
use vkinitialization::device::{Device, PhysicalDevice};
use vkobjects::{
  DeviceManuallyDestroyed,
  errors::{DeviceIsLost, OutOfMemoryError, QueueSubmitError},
};

use crate::{MemoryMapError, MemoryPlacement, create_objs::create_buffer, utility::OnErr};

use super::{AllocationError, MemoryBound};

#[derive(Debug, thiserror::Error)]
pub enum DeviceMemoryInitializationError {
  #[error("Failed to allocate memory for staging buffers:\n{}", {0})]
  AllocationError(#[from] AllocationError),
  #[error("Failed to map staging buffers: {0}")]
  MemoryMapFailed(#[from] MemoryMapError),
  #[error("Generic out of memory error not caused by a failed allocation ({})", {0})]
  GenericOutOfMemory(#[from] OutOfMemoryError),
  #[error(transparent)]
  DeviceIsLost(#[from] DeviceIsLost),
}

impl From<vk::Result> for DeviceMemoryInitializationError {
  fn from(value: vk::Result) -> Self {
    match value {
      vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
        Self::GenericOutOfMemory(OutOfMemoryError::from(value))
      }
      vk::Result::ERROR_DEVICE_LOST => Self::DeviceIsLost(DeviceIsLost {}),
      _ => panic!("Unhandled vk::Result when converting to RecordMemoryInitializationFailedError"),
    }
  }
}

impl From<QueueSubmitError> for DeviceMemoryInitializationError {
  fn from(value: QueueSubmitError) -> Self {
    match value {
      QueueSubmitError::DeviceIsLost(_) => {
        DeviceMemoryInitializationError::DeviceIsLost(DeviceIsLost {})
      }
      QueueSubmitError::OutOfMemory(v) => v.into(),
    }
  }
}

// to be destroyed after command buffer finishes
#[must_use]
#[derive(Debug)]
pub struct SingleUseStagingBuffers<const S: usize> {
  pub buffers: [vk::Buffer; S],
  memories: [vk::DeviceMemory; vk::MAX_MEMORY_TYPES],
  memory_count: usize,
}

impl<const S: usize> DeviceManuallyDestroyed for SingleUseStagingBuffers<S> {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    unsafe {
      self.buffers.destroy_self(device);
      self.memories[0..self.memory_count].destroy_self(device);
    }
  }
}

// creates staging buffers with data to be copied and records appropriate commands to the
// initialization command buffer
pub unsafe fn create_single_use_staging_buffers<const S: usize>(
  device: &Device,
  physical_device: &PhysicalDevice,
  data: [(*const u8, vk::DeviceSize); S],
  #[cfg(feature = "log_alloc")] allocation_name: &str,
  #[cfg(feature = "vl")] marker: &vkinitialization::DebugUtilsMarker,
) -> Result<SingleUseStagingBuffers<S>, DeviceMemoryInitializationError> {
  assert!(!data.is_empty());
  let staging_buffers: [vk::Buffer; S] = vkobjects::fill_destroyable_array_from_iter_using_default!(
    device,
    data.iter().map(|&(_, size)| create_buffer(
      device,
      size,
      vk::BufferUsageFlags::TRANSFER_SRC,
      #[cfg(feature = "vl")]
      marker,
      #[cfg(feature = "vl")]
      c"init_buffer",
    )),
    S
  )?;
  let trait_objs = {
    let mut tmp: [&dyn MemoryBound; S] = [&staging_buffers[0]; S];
    for i in 0..S {
      tmp[i] = &staging_buffers[i];
    }
    tmp
  };

  let staging_alloc = super::allocate_and_bind_memory(
    device,
    physical_device,
    [
      vk::MemoryPropertyFlags::HOST_VISIBLE.bitor(vk::MemoryPropertyFlags::HOST_COHERENT),
      vk::MemoryPropertyFlags::HOST_VISIBLE,
    ],
    trait_objs,
    0.5,
    #[cfg(feature = "log_alloc")]
    None,
    #[cfg(feature = "log_alloc")]
    &format!("INITIALIZATION STAGING BUFFERS <{}>", allocation_name),
  )
  .on_err(|_| unsafe {
    staging_buffers.destroy_self(device);
  })?;

  let destroy_created_objs = || unsafe {
    staging_buffers.destroy_self(device);
    staging_alloc.destroy_self(device);
  };

  let mem_ptrs =
    super::map_host_visible_allocation(device, staging_alloc).on_err(|_| destroy_created_objs())?;

  for (
    MemoryPlacement {
      memory_index,
      memory_offset,
    },
    (ptr, size),
  ) in staging_alloc
    .obj_to_memory_assignment
    .into_iter()
    .zip(data.into_iter())
  {
    unsafe {
      copy_nonoverlapping(
        ptr,
        mem_ptrs[memory_index]
          .as_ptr()
          .byte_add(memory_offset as usize),
        size as usize,
      )
    };
  }

  // no explicit flushing: memory flushed implicitly on queue submit

  let memories = staging_alloc.memories.map(|m| m.memory);
  for &memory in &memories[0..staging_alloc.memory_count] {
    unsafe { device.unmap_memory(memory) };
  }

  Ok(SingleUseStagingBuffers {
    buffers: staging_buffers,
    memories,
    memory_count: staging_alloc.memory_count,
  })
}
