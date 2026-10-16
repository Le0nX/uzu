use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionPrepareKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{
                Attention, AttentionEncodeError, KVCacheView, qkv_norm::QKVNorm, rope::PrecalculatedRoPE,
                state::AttentionState,
            },
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: QKVNorm<B>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let mut projected = self.lin.encode(hidden, batch_dim, command_buffer)?;
        #[cfg(test)]
        crate::engine::code_decision_model::trace_d3::capture::<B>("qkvg_raw", &projected, command_buffer);
        self.norm.encode(&mut projected, batch_dim, command_buffer);
        #[cfg(test)]
        crate::engine::code_decision_model::trace_d3::capture::<B>("qkvg_norm", &projected, command_buffer);
        Ok(projected)
    }
}

impl<B: Backend> Attention<B> {
    pub(super) fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        if self.eager.is_some() && (state.is_some() || !batch_dim.is_flat()) {
            return Err(AttentionEncodeError::EagerStateUnsupported);
        }
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;
        let gate = self.has_gate.then(|| {
            qkvg.subrange(
                (self.projection_dim - self.num_q_heads * self.head_dim) as usize * self.data_type.size_in_bytes()..,
            )
        });

        let attention_output = match state {
            Some(MaybeMut::Mut(state)) => {
                let prefix_len = state.view().prefix_len();
                let queries = self
                    .prepare_kv_and_queries(
                        &qkvg,
                        state.keys.as_mut(),
                        state.values.as_mut(),
                        prefix_len,
                        self.num_q_heads,
                        precalculated_rope,
                        batch_dim.size(),
                        command_buffer,
                    )
                    .map_err(AttentionEncodeError::Backend)?;
                #[cfg(test)]
                {
                    crate::engine::code_decision_model::trace_d3::capture::<B>("queries", &queries, command_buffer);
                    crate::engine::code_decision_model::trace_d3::capture::<B>(
                        "keys",
                        state.keys.as_ref(),
                        command_buffer,
                    );
                    crate::engine::code_decision_model::trace_d3::capture::<B>(
                        "values",
                        state.values.as_ref(),
                        command_buffer,
                    );
                }
                self.run_core(&queries, gate, batch_dim, state, command_buffer)?
            },
            Some(MaybeMut::Const(state)) => {
                // KV sharing: QKVG contains queries and an optional gate only.
                let queries = self
                    .prepare_queries(&qkvg, precalculated_rope, batch_dim.size(), command_buffer)
                    .map_err(AttentionEncodeError::Backend)?;
                self.run_core(&queries, gate, batch_dim, state, command_buffer)?
            },
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let mut keys = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;
                let mut values = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)
                    .map_err(AttentionEncodeError::Backend)?;

                let queries = self
                    .prepare_kv_and_queries(
                        &qkvg,
                        &mut keys,
                        &mut values,
                        0,
                        self.num_q_heads,
                        precalculated_rope,
                        batch_dim.size(),
                        command_buffer,
                    )
                    .map_err(AttentionEncodeError::Backend)?;

                let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));

                if let Some(eager) = &self.eager {
                    eager
                        .encode(&queries, &keys, &values, batch_dim.size(), command_buffer)
                        .map_err(AttentionEncodeError::Backend)?
                } else {
                    self.kernel
                        .encode(
                            AttentionArguments {
                                queries: &queries,
                                keys: &keys,
                                values: &values,
                                suffix_length: batch_dim.size(),
                                trie: None::<&B::ConstantBuffer>,
                                sinks: self.sinks.as_ref(),
                                cache,
                                gate,
                            },
                            command_buffer,
                        )
                        .map_err(AttentionEncodeError::Kernel)?
                }
            },
        };

        #[cfg(test)]
        crate::engine::code_decision_model::trace_d3::capture::<B>(
            "attention_gated",
            &attention_output,
            command_buffer,
        );
        self.out_projection
            .encode(attention_output, batch_dim.size(), command_buffer)
            .map_err(AttentionEncodeError::Linear)
    }

    /// Noncausal code-decision attention with all K/V visible to every query.
    /// Query tiling changes dispatch geometry only, never the attention mask.
    pub(crate) fn encode_code_prompt(
        &self,
        input: &B::GlobalBuffer,
        n: u32,
        rope_config: Option<&crate::config::rope::AnyRoPEConfig>,
        bf16_rope: bool,
        context: &B::Context,
    ) -> Result<B::GlobalBuffer, Box<dyn std::error::Error>> {
        use crate::backends::common::{Buffer, CommandBufferExecutable, CommandBufferPending, Context};
        assert!(!self.is_causal && self.ring_capacity.is_none() && self.eager.is_none());
        let row_bytes = input.size() / n as usize;
        let scalar = self.data_type.size_in_bytes();
        let qrow = self.head_dim as usize * scalar;
        let projrow = self.projection_dim as usize * scalar;
        let mut state = AttentionState::create_empty(self, Some(n), context).map_err(|e| e.to_string())?;
        state.prepare(0, n).map_err(|e| e.to_string())?;
        let mut queries =
            context.create_buffer(n as usize * self.num_q_heads as usize * qrow).map_err(|e| e.to_string())?;
        let mut projected = context.create_buffer(n as usize * projrow).map_err(|e| e.to_string())?;
        for start in (0..n).step_by(4096) {
            let count = (n - start).min(4096);
            context.release_cached_buffers();
            let mut cb = context
                .create_command_buffer(Some("code attention all-KV projection"), None)
                .map_err(|e| e.to_string())?;
            let mut hidden = cb.allocate_scratch(count as usize * row_bytes).map_err(|e| e.to_string())?;
            cb.encode_copy(
                input.subrange(start as usize * row_bytes..(start + count) as usize * row_bytes),
                &mut hidden,
            );
            let qkvg = self.projection.project(hidden, count, &mut cb).map_err(|e| e.to_string())?;
            let positions = (start..start + count).collect::<Vec<_>>();
            let rope = rope_config
                .map(|r| PrecalculatedRoPE::<B>::precalculate_with_precision(r, &positions, bf16_rope, &mut cb))
                .transpose()
                .map_err(|e| e.to_string())?;
            let q = self
                .prepare_kv_and_queries(
                    &qkvg,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    start,
                    self.num_q_heads,
                    rope.as_ref(),
                    count,
                    &mut cb,
                )
                .map_err(|e| e.to_string())?;
            for head in 0..self.num_q_heads as usize {
                let src = head * count as usize * qrow;
                let dst = (head * n as usize + start as usize) * qrow;
                cb.encode_copy(
                    q.subrange(src..src + count as usize * qrow),
                    queries.subrange_mut(dst..dst + count as usize * qrow),
                );
            }
            cb.encode_copy(&qkvg, projected.subrange_mut(start as usize * projrow..(start + count) as usize * projrow));
            drop(q);
            drop(qkvg);
            drop(rope);
            cb.end_encoding()
                .submit()
                .wait_until_completed()
                .map_err(|e| format!("attention KV projection {start}: {e}"))?;
        }
        let mut result = context.create_buffer(input.size()).map_err(|e| e.to_string())?;
        for start in (0..n).step_by(512) {
            let count = (n - start).min(512);
            context.release_cached_buffers();
            let mut cb = context
                .create_command_buffer(Some("code attention query tile, complete KV"), None)
                .map_err(|e| e.to_string())?;
            let mut q =
                cb.allocate_scratch(count as usize * self.num_q_heads as usize * qrow).map_err(|e| e.to_string())?;
            for head in 0..self.num_q_heads as usize {
                let src = (head * n as usize + start as usize) * qrow;
                let dst = head * count as usize * qrow;
                cb.encode_copy(
                    queries.subrange(src..src + count as usize * qrow),
                    q.subrange_mut(dst..dst + count as usize * qrow),
                );
            }
            let gate = self.has_gate.then(|| {
                projected.subrange(
                    start as usize * projrow
                        + (self.projection_dim - self.num_q_heads * self.head_dim) as usize * scalar..,
                )
            });
            // The kernel ABI represents K length as prefix + query length.
            // With no causal/window mask, only their sum n is meaningful:
            // every query tile attends to keys [0,n), including future tokens.
            let out = self
                .kernel
                .encode(
                    AttentionArguments {
                        queries: &q,
                        keys: state.keys.as_ref(),
                        values: state.values.as_ref(),
                        suffix_length: count,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: self.sinks.as_ref(),
                        cache: KVCacheView::full(n - count),
                        gate,
                    },
                    &mut cb,
                )
                .map_err(|e| e.to_string())?;
            let out = self.out_projection.encode(out, count, &mut cb).map_err(|e| e.to_string())?;
            cb.encode_copy(&out, result.subrange_mut(start as usize * row_bytes..(start + count) as usize * row_bytes));
            drop(q);
            drop(out);
            cb.end_encoding()
                .submit()
                .wait_until_completed()
                .map_err(|e| format!("attention queries {start}..{} / KV {n}: {e}", start + count))?;
        }
        drop(queries);
        drop(projected);
        drop(state);
        context.release_cached_buffers();
        Ok(result)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: impl BufferMut<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.projection.norm.encode_key_value(key_value.reborrow(), batch_dim, command_buffer);
        let prefix_len = state.view().prefix_len();
        self.prepare_kv_and_queries(
            key_value.as_ref(),
            state.keys.as_mut(),
            state.values.as_mut(),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn run_core(
        &self,
        queries: impl BufferRef<Backend = B>,
        gate: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, AttentionEncodeError<B>> {
        let trie = if batch_dim.is_flat() {
            None
        } else {
            Some(
                command_buffer
                    .allocate_constant_from_slice(batch_dim.nodes())
                    .map_err(AttentionEncodeError::Backend)?,
            )
        };

        self.kernel
            .encode(
                AttentionArguments {
                    queries,
                    keys: state.keys.as_ref(),
                    values: state.values.as_ref(),
                    suffix_length: batch_dim.size(),
                    trie: trie.as_ref(),
                    sinks: self.sinks.as_ref(),
                    cache: state.view(),
                    gate,
                },
                command_buffer,
            )
            .map_err(AttentionEncodeError::Kernel)
    }

    fn prepare_kv_and_queries(
        &self,
        input: impl BufferRef<Backend = B>,
        keys: impl BufferMut<Backend = B>,
        values: impl BufferMut<Backend = B>,
        kv_token_offset: u32,
        num_q_heads: u32,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let num_kv_heads = self.num_kv_heads.expect("KV prepare requires KV heads");
        // Appended KV is tightly packed; attention projections may have a trailing gate segment.
        let input_row_stride = if num_q_heads == 0 {
            2 * num_kv_heads * self.head_dim
        } else {
            self.projection_dim
        };
        let mut queries = if num_q_heads == 0 {
            command_buffer.allocate_scratch(self.data_type.size_in_bytes())?
        } else {
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?
        };
        self.prepare.encode(
            input,
            &mut queries,
            Some(keys),
            Some(values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            num_q_heads,
            Some(num_kv_heads),
            self.head_dim,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim),
            Some(kv_token_offset),
            input_row_stride,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }

    fn prepare_queries(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut queries =
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?;
        self.prepare.encode(
            qkvg,
            &mut queries,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            precalculated_rope.map(|rope| &rope.cosines),
            precalculated_rope.map(|rope| &rope.sines),
            self.num_q_heads,
            None,
            self.head_dim,
            precalculated_rope.map(|rope| rope.dim),
            None,
            self.projection_dim,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }
}
