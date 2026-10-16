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
        metal::{Metal, context::MetalContext, kernel::matmul::gemm::GemmEngine},
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

// References start with the actual rounded BF16 inputs and unrounded FP32 weights.
// Comparing against the original, unrounded input would test a different operation.
#[derive(Clone, Copy, Debug)]
struct Case {
    m: u32,
    n: u32,
    k: u32,
    transpose: bool,
    padded: bool,
    bias: bool,
    scale: f32,
}

fn check<A: DataTypeRepr + Copy + Into<f32>, B: DataTypeRepr + Copy + Into<f32>, D: DataTypeRepr + Into<f32>>(
    context: &MetalContext,
    case: Case,
    engine: Option<GemmEngine>,
    round_a: impl Fn(f32) -> A,
    round_b: impl Fn(f32) -> B,
) {
    let Case {
        m,
        n,
        k,
        transpose,
        padded,
        bias,
        scale,
    } = case;
    let ld = if transpose {
        k
    } else {
        n
    } + if padded {
        3
    } else {
        0
    };
    let a: Vec<A> = (0..m * k).map(|i| round_a(((i * 7 % 31) as f32 - 15.0) / 16.0)).collect();
    let mut b = vec![
        round_b(f32::NAN);
        (if transpose {
            n
        } else {
            k
        } * ld) as usize
    ];
    for col in 0..n {
        for depth in 0..k {
            let index = if transpose {
                col * ld + depth
            } else {
                depth * ld + col
            };
            b[index as usize] = round_b(((col * 11 + depth * 3) % 37) as f32 / 19.0 - 0.97731);
        }
    }
    let biases: Vec<B> = (0..n).map(|i| round_b(i as f32 / 101.0 - 0.12345)).collect();
    let a_buf = create_buffer_with_data::<Metal, A>(context, &a);
    let b_buf = create_buffer_with_data::<Metal, B>(context, &b);
    let bias_buf = bias.then(|| create_buffer_with_data::<Metal, B>(context, &biases));
    let mut output = context.create_buffer((m * n) as usize * std::mem::size_of::<D>()).unwrap();
    let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        B::data_type(),
        A::data_type(),
        D::data_type(),
    )
    .unwrap();
    let mut cmd = context.create_command_buffer(None, None).unwrap();
    let args = MatmulArguments {
        a: MatmulA::FullPrecision {
            values: &a_buf,
            offset: 0,
        },
        b: MatmulB::FullPrecision {
            b: &b_buf,
        },
        b_leading_dimension: padded.then_some(ld),
        b_transpose: transpose,
        output: MatmulOutput::new(
            &mut output,
            MatmulDOps::<Metal> {
                ab_scale: scale,
                bias: bias_buf.as_ref(),
                rht_factors: None,
                soft_cap: None,
            },
        ),
        gather_indices: None::<&<Metal as Backend>::GlobalBuffer>,
        m,
        n,
        k,
    };
    if let Some(engine) = engine {
        kernel.encode_with_gemm_engine(args, engine, &mut cmd).unwrap();
    } else {
        kernel.encode(args, &mut cmd).unwrap();
    }
    cmd.end_encoding().submit().wait_until_completed().unwrap();
    let actual = buffer_to_vec::<Metal, D>(&output);
    let mut max_error = 0.0_f32;
    for row in 0..m {
        for col in 0..n {
            let mut expected = 0.0_f32;
            let mut absolute_sum = 0.0_f32;
            for depth in 0..k {
                let bi = if transpose {
                    col * ld + depth
                } else {
                    depth * ld + col
                };
                let product = a[(row * k + depth) as usize].into() * b[bi as usize].into();
                expected += product;
                absolute_sum += product.abs();
            }
            expected *= scale;
            if bias {
                expected += biases[col as usize].into();
            }
            let value: f32 = actual[(row * n + col) as usize].into();
            let tolerance = if D::data_type() == DataType::BF16 {
                0.02 + 0.004 * expected.abs() + 0.00001 * absolute_sum * scale.abs()
            } else if A::data_type() == DataType::F32 && B::data_type() == DataType::F32 {
                // Existing homogeneous MXU uses relaxed_precision; keep its established tolerance.
                0.01 + 0.00001 * absolute_sum * scale.abs()
            } else {
                0.0001 + 0.00001 * absolute_sum * scale.abs()
            };
            let error = (value - expected).abs();
            max_error = max_error.max(error);
            assert!(
                value.is_finite() && error <= tolerance,
                "{case:?} {engine:?} [{row},{col}] expected={expected}, actual={value}, tolerance={tolerance}"
            );
        }
    }
    eprintln!(
        "mixed_matmul case={case:?} engine={engine:?} input={:?} weights={:?} output={:?} max_abs_error={max_error} peak_backend_bytes={:?}",
        A::data_type(),
        B::data_type(),
        D::data_type(),
        context.peak_memory_usage()
    );
}

#[uzu_test]
fn mixed_precision_shapes_and_dispatch() {
    let context = MetalContext::new().unwrap();
    let engines = [None, Some(GemmEngine::Simdgroup)];
    // Small rows, tile boundaries, K tails, split K, and the actual d3 readout.
    let shapes = [
        (1, 1, 1),
        (1, 255, 5120),
        (2, 65, 257),
        (4, 32, 1024),
        (8, 33, 63),
        (16, 128, 256),
        (17, 129, 259),
        (33, 65, 127),
        (65, 129, 513),
        (129, 65, 257),
    ];
    for engine in engines {
        for (m, n, k) in shapes {
            for transpose in [false, true] {
                for padded in [false, true] {
                    check::<bf16, f32, f32>(
                        &context,
                        Case {
                            m,
                            n,
                            k,
                            transpose,
                            padded,
                            bias: padded,
                            scale: if padded {
                                -0.5
                            } else {
                                1.0
                            },
                        },
                        engine,
                        bf16::from_f32,
                        |x| x,
                    );
                }
            }
        }
    }
}

#[uzu_test]
fn mixed_precision_homogeneous_regressions() {
    let context = MetalContext::new().unwrap();
    let mut engines = vec![None, Some(GemmEngine::Simdgroup)];
    if context.supports_mxu {
        engines.push(Some(GemmEngine::Mxu));
    }
    for engine in engines {
        for (m, n, k) in [(1, 65, 257), (33, 65, 127), (4, 32, 1024)] {
            for transpose in [false, true] {
                let case = Case {
                    m,
                    n,
                    k,
                    transpose,
                    padded: false,
                    bias: true,
                    scale: 0.5,
                };
                check::<f32, f32, f32>(&context, case, engine, |x| x, |x| x);
                check::<bf16, bf16, bf16>(&context, case, engine, bf16::from_f32, bf16::from_f32);
            }
        }
    }
}

#[uzu_test]
fn mixed_precision_preserves_weight_bits_and_memory() {
    let context = MetalContext::new().unwrap();
    let scalar = Case {
        m: 1,
        n: 1,
        k: 1,
        transpose: true,
        padded: false,
        bias: false,
        scale: 1.0,
    };
    // These FP32 weights change under BF16 / half rounding. A cast of the
    // weights or relaxed MXU arithmetic would fail the FP32 tolerance.
    for weight in [1.0019531_f32, 65537.0_f32] {
        check::<bf16, f32, f32>(&context, scalar, None, |_| bf16::from_f32(1.0), |_| weight);
    }
    let head = Case {
        m: 1,
        n: 255,
        k: 5120,
        ..scalar
    };
    for _ in 0..3 {
        check::<bf16, f32, f32>(&context, head, None, bf16::from_f32, |x| x);
    }
    let warm_peak = context.peak_memory_usage().unwrap();
    for _ in 0..32 {
        check::<bf16, f32, f32>(&context, head, None, bf16::from_f32, |x| x);
    }
    let final_peak = context.peak_memory_usage().unwrap();
    eprintln!("mixed_matmul memory_repeats=32 warm_peak_bytes={warm_peak} final_peak_bytes={final_peak}");
    assert_eq!(final_peak, warm_peak, "backend allocations grew after warmup");

    let kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        DataType::F32,
        DataType::BF16,
        DataType::F32,
    )
    .unwrap();
    use crate::backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
        kernel::matmul::MatmulShape,
    };
    let shape = MatmulShape {
        m: 1,
        n: 255,
        k: 5120,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::FullPrecision,
        b_is_trellis: false,
        b_bits: None,
        b_group_size: None,
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: None,
        d_transform: GemmDTransform::empty(),
    };
    assert!(
        kernel.gemm.select_plan_for_engine(&shape, GemmEngine::Mxu, &context).is_err(),
        "mixed FP32 weights must not select relaxed MXU"
    );
}
