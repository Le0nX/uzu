use uzu_engine_macros::uzu_config;

#[uzu_config(super::Activation)]
pub struct GELU {
    pub approximate: bool,
    #[serde(default)]
    pub pytorch_bf16: bool,
}
