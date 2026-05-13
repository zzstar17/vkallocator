use ash::vk;
#[cfg(feature = "vl")]
use ash::vk::Handle;
use std::{marker::PhantomData, ptr};
use vkobjects::errors::OutOfMemoryError;

pub fn create_buffer(
  device: &ash::Device,
  size: u64,
  usage: vk::BufferUsageFlags,
  #[cfg(feature = "vl")] marker: &vkinitialization::DebugUtilsMarker,
  #[cfg(feature = "vl")] name: &std::ffi::CStr,
) -> Result<vk::Buffer, OutOfMemoryError> {
  let create_info = vk::BufferCreateInfo {
    s_type: vk::StructureType::BUFFER_CREATE_INFO,
    p_next: ptr::null(),
    flags: vk::BufferCreateFlags::empty(),
    size,
    usage,
    sharing_mode: vk::SharingMode::EXCLUSIVE,
    queue_family_index_count: 0,
    p_queue_family_indices: ptr::null(),
    _marker: PhantomData,
  };
  unsafe {
    let buffer = device.create_buffer(&create_info, None)?;
    #[cfg(feature = "vl")]
    marker.set_obj_name(vk::ObjectType::BUFFER, buffer.as_raw(), name)?;
    Ok(buffer)
  }
}
