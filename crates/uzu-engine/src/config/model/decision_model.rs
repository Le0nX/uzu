use uzu_engine_macros::uzu_config;

use crate::config::decision::DecisionConfig;
#[uzu_config(super::ModelConfig)]
pub struct DecisionModelConfig {
    pub decision_config: DecisionConfig,
}
