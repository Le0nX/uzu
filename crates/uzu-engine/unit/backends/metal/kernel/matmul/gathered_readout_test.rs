use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            kernel::{
                Kernels,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
            },
        },
        metal::{Metal, context::MetalContext},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

#[uzu_test]
fn gathered_bf16_readout_covers_small_options_and_k_tails() {
    let context = MetalContext::new().unwrap();
    for m in [1, 2] {
        for n in [1, 2, 3, 27, 255] {
            for k in [4, 129, 4096] {
                let vocabulary = 263;
                let a: Vec<_> = (0..m * k).map(|i| bf16::from_f32((i % 13) as f32 / 8.0 - 0.75)).collect();
                let b: Vec<_> = (0..vocabulary * k).map(|i| bf16::from_f32((i % 17) as f32 / 16.0 - 0.5)).collect();
                // Noncontiguous, per-input candidate rows; output order must be preserved.
                let indices: Vec<_> = (0..m * n).map(|i| (i * 17 + 5) % vocabulary).collect();
                let a_buf = create_buffer_with_data::<Metal, _>(&context, &a);
                let b_buf = create_buffer_with_data::<Metal, _>(&context, &b);
                let gather = create_buffer_with_data::<Metal, _>(&context, &indices);
                let mut output = context.create_buffer((m * n) as usize * 2).unwrap();
                let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                    &context,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                )
                .unwrap();
                let mut cb = context.create_command_buffer(None, None).unwrap();
                kernel
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &a_buf,
                                offset: 0,
                            },
                            b: MatmulB::FullPrecision {
                                b: &b_buf,
                            },
                            b_leading_dimension: None,
                            b_transpose: true,
                            output: MatmulOutput::new(
                                &mut output,
                                MatmulDOps::<Metal> {
                                    ab_scale: 1.0,
                                    bias: None,
                                    rht_factors: None,
                                    soft_cap: None,
                                },
                            ),
                            gather_indices: Some(&gather),
                            m,
                            n,
                            k,
                        },
                        &mut cb,
                    )
                    .unwrap();
                cb.end_encoding().submit().wait_until_completed().unwrap();
                let actual = buffer_to_vec::<Metal, bf16>(&output);
                for row in 0..m {
                    for col in 0..n {
                        let index = (row * n + col) as usize;
                        let expected: f32 = (0..k)
                            .map(|depth| {
                                a[(row * k + depth) as usize].to_f32()
                                    * b[(indices[index] * k + depth) as usize].to_f32()
                            })
                            .sum();
                        assert_eq!(actual[index], bf16::from_f32(expected), "m={m} n={n} k={k} row={row} col={col}");
                    }
                }
            }
        }
    }
}
