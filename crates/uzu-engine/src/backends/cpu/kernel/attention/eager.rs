use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

// Explicit eager graph: BF16 QK, BF16 scale, F32 softmax -> BF16, BF16 PV.
#[kernel(AttentionEagerScores)]
#[variants(T, f32, bf16)]
pub fn attention_eager_scores<T: DataTypeRepr + Float>(
    queries: *const T,
    keys: *const T,
    scores: *mut T,
    length: u32,
    heads: u32,
    head_dim: u32,
    window: u32,
    scale: f32,
) {
    for h in 0..heads as usize {
        for q in 0..length as usize {
            for k in 0..length as usize {
                let mut dot = 0.0f32;
                let allowed = window == 0 || q.abs_diff(k) <= (window / 2) as usize;
                if allowed {
                    dot = crate::backends::cpu::kernel::matmul::dot::bf16_vector_dot(head_dim as usize, |d| unsafe {
                        (
                            (*queries.add((h * length as usize + q) * head_dim as usize + d)).to_f32().unwrap(),
                            (*keys.add((k * heads as usize + h) * head_dim as usize + d)).to_f32().unwrap(),
                        )
                    });
                }
                let score = if allowed {
                    T::from(T::from(dot).unwrap().to_f32().unwrap() * scale).unwrap()
                } else {
                    T::neg_infinity()
                };
                unsafe {
                    *scores.add((h * length as usize + q) * length as usize + k) = score;
                }
            }
        }
    }
}

#[kernel(AttentionEagerValues)]
#[variants(T, f32, bf16)]
pub fn attention_eager_values<T: DataTypeRepr + Float>(
    probabilities: *const T,
    values: *const T,
    output: *mut T,
    length: u32,
    heads: u32,
    head_dim: u32,
) {
    for q in 0..length as usize {
        for h in 0..heads as usize {
            for d in 0..head_dim as usize {
                let sum = crate::backends::cpu::kernel::matmul::dot::bf16_strided_dot(length as usize, |k| unsafe {
                    (
                        (*probabilities.add((h * length as usize + q) * length as usize + k)).to_f32().unwrap(),
                        (*values.add((k * heads as usize + h) * head_dim as usize + d)).to_f32().unwrap(),
                    )
                });
                unsafe {
                    *output.add((q * heads as usize + h) * head_dim as usize + d) = T::from(sum).unwrap();
                }
            }
        }
    }
}
