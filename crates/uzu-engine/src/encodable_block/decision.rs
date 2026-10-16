use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{activation_type::ActivationType, trie::TrieNode},
        kernel::{ActivationKernel, TensorAddScaleKernel},
    },
    config::decision::DecisionConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer::{Transformer, TransformerEncodeError, TransformerNewError},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecisionError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
    #[error("Transformer encoding error: {0}")]
    TransformerEncode(#[from] TransformerEncodeError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct Decision<B: Backend> {
    config: DecisionConfig,
    data_type: DataType,
    embedding: Embedding<B>,
    embedding_norm: Normalization<B>,
    encoder: Transformer<B>,
    type_embedding: Embedding<B>,
    head: Transformer<B>,
    scorer_norm: Normalization<B>,
    scorer_dense: Box<dyn Linear<B>>,
    scorer_readout: Box<dyn Linear<B>>,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    activation: <B::Kernels as Kernels>::ActivationKernel,
}
impl<B: Backend> Decision<B> {
    pub fn new(
        data_type: DataType,
        config: &DecisionConfig,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, DecisionError<B>> {
        if !matches!(data_type, DataType::BF16 | DataType::F32) || config.head_config.layer_configs.is_empty() {
            return Err(DecisionError::UnsupportedConfiguration(
                "requires F32/BF16 and at least one head layer".into(),
            ));
        }
        if config.model_dim == 0
            || config.vocab_size == 0
            || config.num_question_types != 3
            || config.max_sequence_length == 0
            || config.max_head_tokens == 0
            || config.encoder_config.model_dim != config.model_dim
            || config.head_config.model_dim != config.model_dim
            || config.encoder_config.layer_configs.is_empty()
        {
            return Err(DecisionError::UnsupportedConfiguration(
                "invalid dimensions, context, or question type vocabulary".into(),
            ));
        }
        for stack in [&config.encoder_config, &config.head_config] {
            if stack.hidden_dim == 0 {
                return Err(DecisionError::UnsupportedConfiguration("feed-forward width must be nonzero".into()));
            }
            for layer in &stack.layer_configs {
                if layer.hidden_dim == Some(0) {
                    return Err(DecisionError::UnsupportedConfiguration("feed-forward width must be nonzero".into()));
                }
                if let crate::config::token_mixer::AnyTokenMixerConfig::AttentionConfig(attention) = &layer.mixer_config
                    && (attention.num_heads == 0
                        || attention.num_groups == 0
                        || attention.head_dim == 0
                        || !attention.num_heads.is_multiple_of(attention.num_groups)
                        || attention.num_heads.checked_mul(attention.head_dim) != Some(config.model_dim))
                {
                    return Err(DecisionError::UnsupportedConfiguration(
                        "invalid attention dimensions or query/key grouping".into(),
                    ));
                }
            }
        }
        for layer in &config.head_config.layer_configs {
            let crate::config::token_mixer::AnyTokenMixerConfig::AttentionConfig(attention) = &layer.mixer_config
            else {
                return Err(DecisionError::UnsupportedConfiguration("decision head requires full attention".into()));
            };
            if attention.is_causal
                || attention.sliding_window_size.is_some()
                || layer.rope_config.is_some()
                || layer.pre_mixer_norm_config.is_none()
                || layer.post_mixer_norm_config.is_some()
                || layer.post_mlp_norm_config.is_some()
                || layer.kv_source_layer_index.is_some()
            {
                return Err(DecisionError::UnsupportedConfiguration(
                    "decision head requires pre-normalized full bidirectional attention without RoPE or KV sharing"
                        .into(),
                ));
            }
        }
        let (embedding, _) = Embedding::new(
            config.vocab_size,
            config.model_dim,
            data_type,
            &config.embedding_config,
            &tree.subtree("embedding"),
            context,
        )?;
        let embedding_norm = Normalization::new(
            config.model_dim,
            data_type,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.embedding_norm_config,
            &tree.subtree("embedding_norm"),
            context,
        )?;
        let encoder = Transformer::new(data_type, None, &config.encoder_config, &tree.subtree("encoder"), context)?;
        let encoder = if data_type == DataType::BF16 && config.pytorch_bf16 {
            encoder.with_bf16_rope_tables()
        } else {
            encoder
        };
        let (type_embedding, _) = Embedding::new(
            config.num_question_types,
            config.model_dim,
            data_type,
            &config.type_embedding_config,
            &tree.subtree("type_embedding"),
            context,
        )?;
        let head = Transformer::new(data_type, None, &config.head_config, &tree.subtree("head"), context)?;
        let scorer_norm = Normalization::new(
            config.model_dim,
            data_type,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.scorer_norm_config,
            &tree.subtree("scorer_norm"),
            context,
        )?;
        let scorer_dense = <dyn Linear<B>>::new(
            config.model_dim,
            config.model_dim,
            data_type,
            true,
            &tree.subtree("scorer_dense"),
            context,
        )?;
        let scorer_readout =
            <dyn Linear<B>>::new(config.model_dim, 1, data_type, true, &tree.subtree("scorer_readout"), context)?;
        let add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(DecisionError::Backend)?;
        let activation =
            <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true).map_err(DecisionError::Backend)?;
        Ok(Self {
            config: config.clone(),
            data_type,
            embedding,
            embedding_norm,
            encoder,
            type_embedding,
            head,
            scorer_norm,
            scorer_dense,
            scorer_readout,
            add,
            activation,
        })
    }
    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        length: u32,
        question_type: u32,
        markers: &[u32],
        cb: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, DecisionError<B>> {
        let nodes = (0..length)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: length - 1,
                height: index,
            })
            .collect::<Vec<_>>();
        let topology = BatchTopology::new(&nodes, true);
        let embedded = self.embedding.encode_lookup(token_ids, length, cb)?;
        let hidden = self
            .embedding_norm
            .encode(&embedded, 0, length, None::<&mut B::ScratchBuffer>, cb)
            .map_err(DecisionError::Backend)?;
        let hidden = self
            .encoder
            .encode(hidden, None::<&B::ScratchBuffer>, &topology, Some((0..length).into()), None, None, cb)?
            .output
            .unwrap();
        let type_id = cb.allocate_constant_from_slice(&[question_type]).map_err(DecisionError::Backend)?;
        let type_hidden = self.type_embedding.encode_lookup(&type_id, 1, cb)?;
        let mut typed = cb
            .allocate_scratch_for_shape(&[length, self.config.model_dim], self.data_type)
            .map_err(DecisionError::Backend)?;
        self.add.encode(
            Some(&hidden),
            &type_hidden,
            &mut typed,
            self.config.model_dim,
            length * self.config.model_dim,
            1.0,
            cb,
        );
        // Capture the residual after the final head layer. Transformer::output
        // applies a normalizer; the reference head has no final normalization.
        let last = self.config.head_config.layer_configs.len() as u32 - 1;
        let features = self
            .head
            .encode(typed, None::<&B::ScratchBuffer>, &topology, None, Some(&[last]), None, cb)?
            .hidden_features
            .unwrap();
        let hidden = &features[0];
        let row_bytes = self.config.model_dim as usize * self.data_type.size_in_bytes();
        let mut selected = cb.allocate_scratch(markers.len() * row_bytes).map_err(DecisionError::Backend)?;
        for (row, &marker) in markers.iter().enumerate() {
            cb.encode_copy(
                hidden.subrange(marker as usize * row_bytes..(marker as usize + 1) * row_bytes),
                (&mut selected).subrange_mut(row * row_bytes..(row + 1) * row_bytes),
            );
        }
        let count = markers.len() as u32;
        let normalized = self
            .scorer_norm
            .encode(&selected, 0, count, None::<&mut B::ScratchBuffer>, cb)
            .map_err(DecisionError::Backend)?;
        let mut projected = self.scorer_dense.encode(normalized, count, cb)?;
        self.activation.encode(
            None::<&B::ScratchBuffer>,
            &mut projected,
            count * self.config.model_dim,
            if self.config.pytorch_bf16 {
                ActivationType::GELUExactPyTorchBF16
            } else {
                ActivationType::GELUExact
            },
            cb,
        );
        Ok(self.scorer_readout.encode(projected, count, cb)?)
    }
}

#[cfg(test)]
#[path = "../../unit/decision_head_test.rs"]
mod tests;

#[cfg(test)]
#[path = "../../unit/decision_trace.rs"]
mod trace;
