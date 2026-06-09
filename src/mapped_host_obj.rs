use std::{
  assert_matches,
  ops::Deref,
  ptr::{self, NonNull},
};

use ash::vk::{self, Handle};
use vkobjects::{DeviceManuallyDestroyed, errors::OutOfMemoryError};

use crate::{AllocationSuccess, DetailedMemory, MemoryBound, memory_bound::MemoryBoundType};

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum HostMemorySyncError {
  #[error(transparent)]
  OutOfMemoryError(#[from] OutOfMemoryError),
  #[error("Vulkan returned VK_ERROR_UNKNOWN")]
  Unknown,
  #[error("Vulkan returned VK_ERROR_VALIDATION_FAILED")]
  ValidationFailed,
}

impl From<vk::Result> for HostMemorySyncError {
  fn from(value: vk::Result) -> Self {
    match value {
      vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
        Self::OutOfMemoryError(OutOfMemoryError::from(value))
      }
      vk::Result::ERROR_UNKNOWN => Self::Unknown,
      vk::Result::ERROR_VALIDATION_FAILED_EXT => Self::ValidationFailed,
      _ => panic!("Unhandled vk::Result when converting to HostMemorySyncError"),
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct MappedHostObject {
  pub handle: u64,
  pub object_type: MemoryBoundType,

  pub data_ptr: ptr::NonNull<u8>,

  pub memory: vk::DeviceMemory,
  pub mem_host_coherent: bool,

  // in case mem_host_coherent is true, these should be multiples of nonCoherentAtomSize
  pub obj_offset: u64,
  pub obj_size: u64,
}

impl MappedHostObject {
  pub fn from_allocation<const P: usize, const S: usize>(
    allocation: &AllocationSuccess<S>,
    mem_props: [vk::MemoryPropertyFlags; P],
    objs: &[&dyn MemoryBound; S],
    mapped_ptrs: [NonNull<u8>; S],
  ) -> [Self; S] {
    // can't transmute between S sized arrays so no MaybeUninit this time
    let default = Self {
      handle: 0,
      object_type: MemoryBoundType::Buffer,
      data_ptr: NonNull::dangling(),
      memory: vk::DeviceMemory::null(),
      mem_host_coherent: false,
      obj_offset: 0,
      obj_size: 0,
    };
    let mut result = [default; S];

    let memories = allocation.get_memories();
    for (i, memory_placement) in allocation.obj_to_memory_assignment.iter().enumerate() {
      let DetailedMemory {
        memory,
        type_index,
        size: _mem_size,
      } = memories[memory_placement.memory_index];
      debug_assert!(mem_props[type_index].contains(vk::MemoryPropertyFlags::HOST_VISIBLE));

      let mem_host_coherent =
        mem_props[type_index].contains(vk::MemoryPropertyFlags::HOST_COHERENT);

      let obj = Self {
        handle: objs[i].object_handle(),
        object_type: objs[i].object_type(),

        data_ptr: mapped_ptrs[i],
        memory,
        mem_host_coherent,

        obj_offset: memory_placement.memory_offset,
        obj_size: memory_placement.object_size,
      };

      result[i] = obj;
    }

    result
  }

  pub fn into_buffer<T>(self) -> MappedHostBuffer<T> {
    assert_matches!(self.object_type, MemoryBoundType::Buffer);
    MappedHostBuffer {
      buffer: vk::Buffer::from_raw(self.handle),
      data_ptr: NonNull::new(self.data_ptr.as_ptr() as *mut T).unwrap(),
      memory: self.memory,
      mem_host_coherent: self.mem_host_coherent,
      buffer_offset: self.obj_offset,
      buffer_size: self.obj_size,
    }
  }

  pub fn into_image(self) -> MappedHostImage {
    assert_matches!(self.object_type, MemoryBoundType::Image);
    MappedHostImage {
      image: vk::Image::from_raw(self.handle),
      data_ptr: self.data_ptr,
      memory: self.memory,
      mem_host_coherent: self.mem_host_coherent,
      image_offset: self.obj_offset,
      image_size: self.obj_size,
    }
  }
}

impl DeviceManuallyDestroyed for MappedHostObject {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    match self.object_type {
      MemoryBoundType::Buffer => {
        let buffer = vk::Buffer::from_raw(self.handle);
        unsafe {
          buffer.destroy_self(device);
        }
      }
      MemoryBoundType::Image => {
        let image = vk::Image::from_raw(self.handle);
        unsafe {
          image.destroy_self(device);
        }
      }
    }
  }
}

/// Buffer and its mapped pointer
#[derive(Debug, Clone, Copy)]
pub struct MappedHostBuffer<T> {
  pub buffer: vk::Buffer,
  pub data_ptr: ptr::NonNull<T>,

  pub memory: vk::DeviceMemory,
  pub mem_host_coherent: bool,

  // multiples of nonCoherentAtomSize
  pub buffer_offset: u64,
  pub buffer_size: u64,
}

impl<T> MappedHostBuffer<T> {
  pub unsafe fn copy_to_buffer_memory(&self, src: &[T]) {
    debug_assert!(self.buffer_size as usize >= size_of::<T>() * src.len());
    unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.data_ptr.as_ptr(), src.len()) };
  }

  pub unsafe fn copy_to_buffer_memory_ptr(&self, src: *const u8, size_bytes: usize) {
    debug_assert!(self.buffer_size as usize >= size_bytes);
    unsafe { ptr::copy_nonoverlapping(src, self.data_ptr.as_ptr() as *mut u8, size_bytes) };
  }

  pub unsafe fn read_to_box(&self, count: usize) -> Box<[T]> {
    let mut values = Box::<[T]>::new_uninit_slice(count);
    unsafe {
      ptr::copy_nonoverlapping(self.data_ptr.as_ptr(), values.as_mut_ptr() as *mut T, count);
      values.assume_init()
    }
  }

  pub fn memory_range(&self) -> vk::MappedMemoryRange<'_> {
    vk::MappedMemoryRange {
      memory: self.memory,
      offset: self.buffer_offset,
      size: self.buffer_size,
      ..Default::default()
    }
  }

  pub unsafe fn flush_memory_range(&self, device: &ash::Device) -> Result<(), HostMemorySyncError> {
    if !self.mem_host_coherent {
      let ranges = [self.memory_range()];
      unsafe {
        device.flush_mapped_memory_ranges(&ranges)?;
      }
    }
    Ok(())
  }

  pub unsafe fn invalidate_memory_range(
    &self,
    device: &ash::Device,
  ) -> Result<(), HostMemorySyncError> {
    if !self.mem_host_coherent {
      let ranges = [self.memory_range()];
      unsafe {
        device.invalidate_mapped_memory_ranges(&ranges)?;
      }
    }
    Ok(())
  }
}

impl<T> Deref for MappedHostBuffer<T> {
  type Target = vk::Buffer;
  fn deref(&self) -> &Self::Target {
    &self.buffer
  }
}

impl<T> DeviceManuallyDestroyed for MappedHostBuffer<T> {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    unsafe {
      self.buffer.destroy_self(device);
    }
  }
}

/// Image and its mapped pointer
#[derive(Debug, Clone, Copy)]
pub struct MappedHostImage {
  pub image: vk::Image,
  pub data_ptr: ptr::NonNull<u8>,

  pub memory: vk::DeviceMemory,
  pub mem_host_coherent: bool,

  // multiples of nonCoherentAtomSize
  pub image_offset: u64,
  pub image_size: u64,
}

impl MappedHostImage {
  pub unsafe fn copy_to_image_memory(&self, src: &[u8]) {
    unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.data_ptr.as_ptr(), src.len()) };
  }

  pub unsafe fn copy_to_image_memory_ptr(&self, src: *const u8, size_bytes: usize) {
    unsafe { ptr::copy_nonoverlapping(src, self.data_ptr.as_ptr(), size_bytes) };
  }

  pub fn memory_range(&self) -> vk::MappedMemoryRange<'_> {
    vk::MappedMemoryRange {
      memory: self.memory,
      offset: self.image_offset,
      size: self.image_size,
      ..Default::default()
    }
  }

  pub unsafe fn flush_memory_range(&self, device: &ash::Device) -> Result<(), HostMemorySyncError> {
    if !self.mem_host_coherent {
      let ranges = [self.memory_range()];
      unsafe {
        device.flush_mapped_memory_ranges(&ranges)?;
      }
    }
    Ok(())
  }

  pub unsafe fn invalidate_memory_range(
    &self,
    device: &ash::Device,
  ) -> Result<(), HostMemorySyncError> {
    if !self.mem_host_coherent {
      let ranges = [self.memory_range()];
      unsafe {
        device.invalidate_mapped_memory_ranges(&ranges)?;
      }
    }
    Ok(())
  }
}

impl Deref for MappedHostImage {
  type Target = vk::Image;
  fn deref(&self) -> &Self::Target {
    &self.image
  }
}

impl DeviceManuallyDestroyed for MappedHostImage {
  unsafe fn destroy_self(&self, device: &ash::Device) {
    unsafe {
      self.image.destroy_self(device);
    }
  }
}
