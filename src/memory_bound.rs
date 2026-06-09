use ash::vk::{self, Handle};
use vkobjects::errors::OutOfMemoryError;

#[derive(Debug, Clone, Copy)]
pub enum MemoryBoundType {
  Buffer,
  Image,
}

pub trait MemoryBound {
  unsafe fn bind(
    &self,
    device: &ash::Device,
    memory: vk::DeviceMemory,
    offset: u64,
  ) -> Result<(), OutOfMemoryError>;

  unsafe fn get_memory_requirements(&self, device: &ash::Device) -> vk::MemoryRequirements;

  fn object_type(&self) -> MemoryBoundType;
  fn object_handle(&self) -> u64;
}

impl MemoryBound for vk::Buffer {
  unsafe fn bind(
    &self,
    device: &ash::Device,
    memory: vk::DeviceMemory,
    offset: u64,
  ) -> Result<(), OutOfMemoryError> {
    unsafe { device.bind_buffer_memory(*self, memory, offset) }.map_err(|err| err.into())
  }

  unsafe fn get_memory_requirements(&self, device: &ash::Device) -> vk::MemoryRequirements {
    unsafe { device.get_buffer_memory_requirements(*self) }
  }

  fn object_type(&self) -> MemoryBoundType {
    MemoryBoundType::Buffer
  }

  fn object_handle(&self) -> u64 {
    self.as_raw()
  }
}

impl MemoryBound for vk::Image {
  unsafe fn bind(
    &self,
    device: &ash::Device,
    memory: vk::DeviceMemory,
    offset: u64,
  ) -> Result<(), OutOfMemoryError> {
    unsafe { device.bind_image_memory(*self, memory, offset) }.map_err(|err| err.into())
  }

  unsafe fn get_memory_requirements(&self, device: &ash::Device) -> vk::MemoryRequirements {
    unsafe { device.get_image_memory_requirements(*self) }
  }

  fn object_type(&self) -> MemoryBoundType {
    MemoryBoundType::Image
  }

  fn object_handle(&self) -> u64 {
    self.as_raw()
  }
}
