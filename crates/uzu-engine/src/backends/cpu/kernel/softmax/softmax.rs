use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::{DataType, DataTypeRepr};

#[kernel(Softmax)]
#[variants(T, f32, f16, bf16)]
pub fn softmax<T: DataTypeRepr + Float>(
    values: *mut T,
    #[optional(has_sinks)] sinks: Option<*const T>,
    row_length: u32,
    outer_dim: u32,
    batch_dim: u32,
    #[specialize] has_sinks: bool,
) {
    for outer_index in 0..outer_dim {
        for batch_index in 0..batch_dim {
            let row_offset = ((outer_index * batch_dim + batch_index) * row_length) as usize;
            let sink = if has_sinks {
                unsafe { *sinks.unwrap().add(outer_index as usize) }.to_f32().unwrap()
            } else {
                f32::NEG_INFINITY
            };

            let mut max_value = if has_sinks {
                sink
            } else {
                f32::NEG_INFINITY
            };
            for i in 0..row_length as usize {
                max_value = max_value.max(unsafe { *values.add(row_offset + i) }.to_f32().unwrap());
            }

            let mut sum_exp = if has_sinks {
                (sink - max_value).exp()
            } else {
                0.0
            };
            let mut partial_sums = [0.0f32; 4];
            for i in 0..row_length as usize {
                let p = (unsafe { *values.add(row_offset + i) }.to_f32().unwrap() - max_value).exp();
                if T::data_type() == DataType::BF16 && !has_sinks {
                    partial_sums[i % 4] += p;
                } else {
                    sum_exp += p;
                }
            }
            if T::data_type() == DataType::BF16 && !has_sinks {
                sum_exp = (partial_sums[0] + partial_sums[1]) + (partial_sums[2] + partial_sums[3]);
            }

            let inv = 1.0 / sum_exp;
            for i in 0..row_length as usize {
                unsafe {
                    // Round the normalized probability once, as the Metal
                    // implementation does; never round the numerator first.
                    let x = (*values.add(row_offset + i)).to_f32().unwrap();
                    *values.add(row_offset + i) = T::from((x - max_value).exp() * inv).unwrap();
                }
            }
        }
    }
}
