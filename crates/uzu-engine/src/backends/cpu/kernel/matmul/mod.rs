pub(crate) mod dot;
mod error;
mod kernel;
mod reference;

pub use error::MatmulCpuError;
pub use kernel::MatmulCpuKernel;
