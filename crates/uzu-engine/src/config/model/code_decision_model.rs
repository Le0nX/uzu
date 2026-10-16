use uzu_engine_macros::uzu_config;

use crate::config::decoder::DecoderConfig;

/// Input vocabulary and code readout have different sizes. This is never a
/// generative LanguageModelConfig: readout row ids are answer codes, not tokens.
#[uzu_config(super::ModelConfig)]
pub struct CodeDecisionModelConfig {
    pub architecture_id: String,
    pub decoder_config: DecoderConfig,
    pub readout_rows: u32,
    pub readout_dtype: String,
    pub max_sequence_length: u32,
}
