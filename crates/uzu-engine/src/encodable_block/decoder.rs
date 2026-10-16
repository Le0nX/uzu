use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::decoder::DecoderConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError, TransformerState},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
    pub final_hidden: Option<B::ScratchBuffer>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub fn new(
        data_type: DataType,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, DecoderError<B>> {
        let (embedding, readout_input_hadamard_factors) = Embedding::new(
            config.vocab_size,
            config.transformer_config.model_dim,
            data_type,
            &config.embedding_config,
            &parameter_tree.subtree("embedding"),
            context,
        )?;

        let embedding_norm = config
            .embedding_norm_config
            .as_ref()
            .map(|norm_config| {
                Normalization::new(
                    config.transformer_config.model_dim,
                    data_type,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    norm_config,
                    &parameter_tree.subtree("embedding_norm"),
                    context,
                )
            })
            .transpose()?;

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len() as u32,
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(PerLayerEmbedding::new(
                config.transformer_config.model_dim,
                data_type,
                ple_config,
                &parameter_tree.subtree("per_layer_embedding"),
                context,
            )?)
        } else {
            None
        };

        let transformer = Transformer::new(
            data_type,
            readout_input_hadamard_factors,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
            context,
        )?;

        Ok(Self {
            embedding,
            embedding_norm,
            per_layer_embedding,
            transformer,
        })
    }

    pub fn speculation_supported(&self) -> bool {
        self.transformer.speculation_supported()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.transformer.prefill_cache_skips_trailing_layers()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub(crate) fn encode_code_prompt(
        &self,
        input: &[u32],
        batch: &BatchTopology,
        context: &B::Context,
    ) -> Result<B::GlobalBuffer, Box<dyn std::error::Error>> {
        use crate::backends::common::{
            Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        };
        assert!(self.embedding_norm.is_none() && self.per_layer_embedding.is_none());
        context.release_cached_buffers();
        let mut cb = context.create_command_buffer(Some("code decision embedding"), None).map_err(|e| e.to_string())?;
        let ids = cb.allocate_constant_from_slice(input).map_err(|e| e.to_string())?;
        let hidden = self.embedding.encode_lookup(&ids, batch.size(), &mut cb).map_err(|e| e.to_string())?;
        let mut saved = context.create_buffer(hidden.size()).map_err(|e| e.to_string())?;
        cb.encode_copy(&hidden, &mut saved);
        drop(hidden);
        drop(ids);
        cb.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        self.transformer.encode_code_prompt(saved, batch, context)
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        self.encode_inner(
            token_ids,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            None,
            true,
            state,
            command_buffer,
        )
    }

    /// Score only the requested vocabulary rows at the final prompt position.
    /// This is a decoder forward pass, without sampling or generated tokens.
    pub fn encode_candidates(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        candidates: &B::ConstantBuffer,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        self.encode_inner(
            token_ids,
            batch_dim,
            Some((batch_dim.size() - 1..batch_dim.size()).into()),
            None,
            Some(candidates),
            true,
            state,
            command_buffer,
        )
    }

    fn encode_inner(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        candidates: Option<&B::ConstantBuffer>,
        readout: bool,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("decoder");

        let embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), command_buffer)?;
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(per_layer_embedding.encode(token_ids, &embedded, batch_dim.size(), command_buffer)?)
        } else {
            None
        };

        let transformer_output = self.transformer.encode(
            embedded,
            per_layer_inputs.as_ref(),
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            Some(state),
            command_buffer,
        )?;

        let logits = if let Some(output_range) = output_range.filter(|_| readout) {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            Some(self.embedding.encode_readout(
                output,
                output_range.end - output_range.start,
                candidates.map_or(self.embedding.vocab_size(), |ids| ids.as_slice::<u32>().len() as u32),
                candidates,
                true,
                command_buffer,
            )?)
        } else {
            None
        };
        let final_hidden = if hidden_feature_layer_indices.is_none() {
            None
        } else {
            transformer_output.output
        };

        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}
