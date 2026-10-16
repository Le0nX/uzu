use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, Kernels, gpu_types::activation_type::ActivationType, kernel::ActivationKernel,
    },
    config::mlp::feed_forward_mlp::FeedForwardMLPConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::{Mlp, MlpEncodeError},
    },
    safetensors::ParameterTree,
};
#[derive(Debug, Error)]
pub enum FeedForwardNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}
pub struct FeedForward<B: Backend> {
    up: Box<dyn Linear<B>>,
    down: Box<dyn Linear<B>>,
    activation: ActivationType,
    kernel: <B::Kernels as Kernels>::ActivationKernel,
    hidden_dimension: u32,
}
impl<B: Backend> FeedForward<B> {
    pub fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &FeedForwardMLPConfig,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalBuffer>), FeedForwardNewError<B>> {
        let (up, factors) = <dyn Linear<B>>::new_with_input_rht(
            model_dimension,
            hidden_dimension,
            data_type,
            config.has_up_biases,
            &tree.subtree("up_projection"),
            context,
        )?;
        let down = <dyn Linear<B>>::new(
            hidden_dimension,
            model_dimension,
            data_type,
            config.has_down_biases,
            &tree.subtree("down_projection"),
            context,
        )?;
        let kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(FeedForwardNewError::Backend)?;
        Ok((
            Self {
                up,
                down,
                kernel,
                activation: config.activation.act_type(),
                hidden_dimension,
            },
            factors,
        ))
    }
}
impl<B: Backend> Mlp<B> for FeedForward<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let mut hidden = self.up.encode(input, batch_dim, command_buffer)?;
        self.kernel.encode(
            None::<&B::ScratchBuffer>,
            &mut hidden,
            batch_dim * self.hidden_dimension,
            self.activation,
            command_buffer,
        );
        Ok(self.down.encode(hidden, batch_dim, command_buffer)?)
    }
}
