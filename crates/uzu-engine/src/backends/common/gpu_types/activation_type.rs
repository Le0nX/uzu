use num_traits::Float;

const GELU_K0: f32 = 0.044715;
const GELU_K1: f32 = 0.7978846; // sqrt(2/pi)

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationType {
    SILU,
    GELUApprox,
    GELUExact,
    IDENTITY,
    SOFTPLUS,
    RELU,
    GELUExactPyTorchBF16,
}

impl ActivationType {
    pub fn activate<T: Float>(
        &self,
        x: T,
    ) -> T {
        match self {
            ActivationType::SILU => silu(x),
            ActivationType::GELUApprox => gelu_approx(x),
            ActivationType::GELUExact => gelu_exact(x),
            ActivationType::GELUExactPyTorchBF16 => {
                if T::epsilon().to_f32() == Some(0.0078125) {
                    let xf = x.to_f32().unwrap();
                    T::from(0.5 * xf * (1.0 + pytorch_neon_erf(xf * std::f32::consts::FRAC_1_SQRT_2))).unwrap()
                } else {
                    gelu_exact(x)
                }
            },
            ActivationType::IDENTITY => x,
            ActivationType::SOFTPLUS => softplus(x),
            ActivationType::RELU => {
                if x < T::zero() {
                    T::zero()
                } else {
                    x
                }
            },
        }
    }
}

pub fn activation_silu_alpha<T: Float>(
    x: T,
    alpha: f32,
) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = x_float / (1.0f32 + (-alpha * x_float).exp());
    T::from(y_float).unwrap()
}

fn silu<T: Float>(x: T) -> T {
    activation_silu_alpha(x, 1.0f32)
}

fn gelu_approx<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let tan_arg = GELU_K1 * (x_float + GELU_K0 * x_float * x_float * x_float);
    let y_float = 0.5f32 * x_float * (1.0f32 + tan_arg.tanh());
    T::from(y_float).unwrap()
}

fn gelu_exact<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = 0.5f32 * x_float * (1.0f32 + libm::erff(x_float * std::f32::consts::FRAC_1_SQRT_2));
    T::from(y_float).unwrap()
}

fn softplus<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    if x_float > 20f32 {
        return x;
    }

    let result = (1f32 + x_float.exp()).ln();
    T::from(result).unwrap()
}

// BF16 reference compatibility, derived from PyTorch's AArch64 erf and the
// Arm optimized-routines exp polynomial. See THIRD_PARTY_DECISION_NOTICES.txt.
fn pytorch_neon_exp(x: f32) -> f32 {
    if !x.is_finite() || x.abs() > 87.3 {
        return x.exp();
    }
    let inv_ln2 = f32::from_bits(0x3fb8aa3b);
    let ln2_hi = f32::from_bits(0x3f317200);
    let ln2_lo = f32::from_bits(0x35bfbe8e);
    let c0 = f32::from_bits(0x3c072010);
    let c2 = f32::from_bits(0x3e2aaf33);
    let c1 = f32::from_bits(0x3d2b9f17);
    let c3 = f32::from_bits(0x3efffedb);
    let c4 = f32::from_bits(0x3f7ffff6);
    let n = (x * inv_ln2).round();
    let r = (-n).mul_add(ln2_hi, x);
    let r = (-n).mul_add(ln2_lo, r);
    let scale = f32::from_bits(0x3f800000u32.wrapping_add((n as i32 as u32).wrapping_shl(23)));
    let r2 = r * r;
    let p = r.mul_add(c0, c1);
    let q = r.mul_add(c2, c3);
    let q = r2.mul_add(p, q);
    let poly = r2.mul_add(q, c4 * r);
    scale.mul_add(poly, scale)
}
fn pytorch_neon_erf(x: f32) -> f32 {
    let t = 1.0 / 0.3275911f32.mul_add(x.abs(), 1.0);
    let r = 1.061405429f32.mul_add(t, -1.453152027);
    let r = r.mul_add(t, 1.421413741);
    let r = r.mul_add(t, -0.284496736);
    let r = r.mul_add(t, 0.254829592);
    let value = (-t * pytorch_neon_exp(-(x * x))).mul_add(r, 1.0);
    f32::from_bits(value.to_bits() ^ (x.to_bits() & 0x80000000))
}
