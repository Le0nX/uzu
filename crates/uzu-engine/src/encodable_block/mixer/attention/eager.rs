//! Correctness-oriented eager BF16 path for stateless marker encoders.
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{AttentionEagerScoresKernel, AttentionEagerValuesKernel, SoftmaxKernel},
    },
    config::token_mixer::attention::AttentionConfig,
    data_type::DataType,
};
pub(super) struct EagerAttention<B: Backend> {
    scores: <B::Kernels as Kernels>::AttentionEagerScoresKernel,
    softmax: <B::Kernels as Kernels>::SoftmaxKernel,
    values: <B::Kernels as Kernels>::AttentionEagerValuesKernel,
    data_type: DataType,
    heads: u32,
    head_dim: u32,
    window: u32,
    scale: f32,
}
impl<B: Backend> EagerAttention<B> {
    pub fn new(
        config: &AttentionConfig,
        data_type: DataType,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            scores: <B::Kernels as Kernels>::AttentionEagerScoresKernel::new(context, data_type)?,
            softmax: <B::Kernels as Kernels>::SoftmaxKernel::new(context, data_type, false)?,
            values: <B::Kernels as Kernels>::AttentionEagerValuesKernel::new(context, data_type)?,
            data_type,
            heads: config.num_heads,
            head_dim: config.head_dim,
            window: config.sliding_window_size.unwrap_or(0),
            scale: config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
        })
    }
    pub fn encode(
        &self,
        queries: impl BufferRef<Backend = B>,
        keys: impl BufferRef<Backend = B>,
        values: impl BufferRef<Backend = B>,
        length: u32,
        cb: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut scores = cb.allocate_scratch_for_shape(&[self.heads, length, length], self.data_type)?;
        self.scores.encode(queries, keys, &mut scores, length, self.heads, self.head_dim, self.window, self.scale, cb);
        self.softmax.encode(&mut scores, None::<&B::GlobalBuffer>, length, self.heads, length, cb);
        let mut output = cb.allocate_scratch_for_shape(&[length, self.heads, self.head_dim], self.data_type)?;
        self.values.encode(&scores, values, &mut output, length, self.heads, self.head_dim, cb);
        Ok(output)
    }
}
