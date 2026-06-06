use ash::vk;

#[derive(Debug)]
pub struct UnassignedToMemoryObjectsData<'a, const P: usize, const S: usize> {
  /// Physical's device memory types
  /// Contains heap index and supported memory properties
  pub mem_types: &'a [vk::MemoryType],
  // Assign objects to the first set of memory properties and otherwise try to put them together
  pub mem_props: [vk::MemoryPropertyFlags; P],
  /// Memory requirements from each object
  /// Each object is only supported in a set of memory types
  pub obj_reqs: [vk::MemoryRequirements; S],
  #[cfg(feature = "log_alloc")]
  pub obj_labels: Option<[&'static str; S]>,
}

#[cfg(feature = "log_alloc")]
impl<'a, const P: usize, const S: usize> std::fmt::Display
  for UnassignedToMemoryObjectsData<'a, P, S>
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let labels = self
      .obj_labels
      .as_ref()
      .map(|labels_arr| labels_arr.as_slice());
    super::logging::write_properties_table(
      f,
      self.mem_types,
      &self.mem_props,
      &self.obj_reqs,
      None,
      labels,
    )
  }
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum MemoryAssignmentError {
  #[error("All given memory property sets are unsupported")]
  AllPropertiesUnsupported,
  #[error(
    "One of the given objects (o{0}) memory requirements is incompatible with all memory property sets"
  )]
  ObjectIncompatibleWithAllProperties(usize),
}

/// Assigns an memory type index to each object in a way that each object gets the earliest
/// supported memory properties and that preferably objects stay together
///
/// Returns assigned memory types for each object and the total number of unique memory types
///
/// todo: doesn't test if requirements exceed memory heap capacity
pub fn assign_memory_type_indexes_to_objects_for_allocation<const P: usize, const S: usize>(
  data: UnassignedToMemoryObjectsData<P, S>,
) -> Result<([usize; S], usize), MemoryAssignmentError> {
  // note: memory types bitmask in vulkan are ordered from right to left, so the first type index
  // is the last bit (so bitmask & 1 > 0 tests if the first type is compatible)

  // bitmask of memory types that are supported by the given desired properties
  let bit_switch = 1 << (data.mem_types.len() - 1);
  let supported_properties = data.mem_props.map(|p: vk::MemoryPropertyFlags| {
    let mut support_bitmask: u32 = 0;
    for t in data.mem_types {
      support_bitmask >>= 1;
      if t.property_flags.contains(p) {
        support_bitmask |= bit_switch; // switch last bit to 1
      }
    }
    support_bitmask
  });

  if supported_properties.iter().all(|&bitmask| bitmask == 0) {
    // no desired properties are supported by the system
    return Err(MemoryAssignmentError::AllPropertiesUnsupported);
  }

  // write final choice masks for objects
  let mut object_masks = [0; S];
  for (i, requirements) in data.obj_reqs.iter().enumerate() {
    for supported in supported_properties {
      let this_supported = requirements.memory_type_bits & supported;
      if this_supported > 0 {
        object_masks[i] = this_supported;
        break;
      }
    }
    if object_masks[i] == 0 {
      // this object is unsupported by all given memory properties
      return Err(MemoryAssignmentError::ObjectIncompatibleWithAllProperties(
        i,
      ));
    }
  }

  // count how many of each type the objects support
  let mut memory_type_counters = [0usize; vk::MAX_MEMORY_TYPES];
  for mask in object_masks {
    for i in 0..(data.mem_types.len()) {
      if mask & (1 << i) > 0 {
        memory_type_counters[i] += 1
      }
    }
  }

  let mut assigned = [usize::MAX; S];
  let mut remaining = S;
  let mut unique_type_count = 0;
  // choose the first type with the highest amount of objects and add assign them
  // repeat until all objects are assigned
  while remaining > 0 {
    // find first max
    let mut cur_max = 0;
    let mut max_i = usize::MAX;
    for i in 0..(data.mem_types.len()) {
      if memory_type_counters[i] > cur_max {
        cur_max = memory_type_counters[i];
        max_i = i;
      }
    }
    // assert that the chosen counter is actually valid
    assert!(max_i != usize::MAX);
    assert!(cur_max <= remaining);

    unique_type_count += 1;

    if cur_max == remaining {
      // all remaining objects support this type
      for (obj_i, _obj_mask) in object_masks.iter().enumerate() {
        // object unassigned
        if assigned[obj_i] == usize::MAX {
          assigned[obj_i] = max_i;
          remaining -= 1;
        }
      }
    } else {
      // assign all objects to type with index max_i if they support it
      for (obj_i, obj_mask) in object_masks.iter().enumerate() {
        // object unassigned and type is supported
        if assigned[obj_i] == usize::MAX && obj_mask & (1 << max_i) > 0 {
          assigned[obj_i] = max_i;

          // remove from counters
          for i in 0..(data.mem_types.len()) {
            if obj_mask & (1 << i) > 0 {
              memory_type_counters[i] -= 1
            }
          }

          remaining -= 1;
        }
      }
    }
  }

  Ok((assigned, unique_type_count))
}
