use uzu_engine_macros::uzu_config;

use crate::config::{
    embedding::AnyEmbeddingConfig, linear::LinearConfig, normalization::NormalizationConfig,
    transformer::TransformerConfig,
};

/// Marker-scoring encoder used by Julia and Laya. This is separate from a
/// pooled classifier: each question has a variable number of option markers.
#[uzu_config]
pub struct DecisionConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub embedding_norm_config: NormalizationConfig,
    pub encoder_config: TransformerConfig,
    pub type_embedding_config: AnyEmbeddingConfig,
    pub head_config: TransformerConfig,
    pub scorer_norm_config: NormalizationConfig,
    pub scorer_dense_config: LinearConfig,
    pub scorer_readout_config: LinearConfig,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub num_question_types: u32,
    pub max_sequence_length: u32,
    pub max_head_tokens: u32,
    /// All-BF16 module semantics against the explicit PyTorch CPU reference.
    /// This is distinct from BF16 weights with F32 activations (benchmark v2).
    #[serde(default)]
    pub pytorch_bf16: bool,
}
