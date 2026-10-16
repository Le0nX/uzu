mod error;
mod kernel;
mod policy;
mod selection;
mod specialization;
mod trellis;

pub use error::GemmSpecializationError;
pub use kernel::GemmKernel;
pub use selection::GemmPlanError;
pub(super) use selection::GemmProblem;

use crate::{backends::metal::gpu_types::gemm::GemmTiling, data_type::DataType};

// The existing MXU path enables relaxed_precision. Use SIMD-group MMA for
// BF16 x FP32 so the newly supported operation preserves the FP32 weights.
pub(super) fn requires_simdgroup(
    weights: DataType,
    input: DataType,
) -> bool {
    weights == DataType::F32 && input == DataType::BF16
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmEngine {
    Simdgroup,
    Mxu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlan {
    pub(super) engine: GemmEngine,
    pub(super) tiling: GemmTiling,
    pub(super) split_k: u32,
}
