use uzu_engine_macros::uzu_test;

use super::KVCacheState;

fn accept(
    state: &mut KVCacheState,
    indices: &[u32],
) -> Vec<(u32, u32)> {
    state.accept(indices).into_iter().map(|copy| (copy.source, copy.destination)).collect()
}

#[uzu_test]
fn ring_accepts_append_then_wrap() {
    let mut state = KVCacheState::ring(2);

    assert_eq!(accept(&mut state, &[1]), [(1, 0)]);
    assert_eq!(state.view().prefix_len(), 1);

    assert!(accept(&mut state, &[0]).is_empty());
    assert_eq!(state.view().prefix_len(), 2);

    assert_eq!(accept(&mut state, &[0]), [(2, 0)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);
    assert_eq!(state.required_prefix_len(99), 2);
}

#[uzu_test]
fn ring_accept_over_capacity_keeps_last() {
    let mut state = KVCacheState::ring(2);

    // Only the last `capacity` tokens survive; earlier copies are dead.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(2, 0)]);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);

    // Same with a wrapped ring: destinations keep rotating.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(3, 0), (4, 1)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 0);
}

fn whole_prompt_prepare<B: crate::backends::common::Backend>() {
    use super::AttentionState;
    use crate::{
        backends::common::{
            Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            DeviceCapabilities, Kernels, kernel::KVCacheUpdateKernel,
        },
        data_type::DataType,
        encodable_block::mixer::MixerState,
        tests::helpers::create_context,
    };
    let ctx = create_context::<B>();
    let n = 19_467u32;
    let width = 64u32;
    let bytes = (n * width * 2) as usize;
    let sparse = ctx.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS);
    let make = || -> Box<dyn Buffer<Backend = B>> {
        if sparse {
            Box::new(ctx.create_sparse_buffer(bytes).unwrap())
        } else {
            Box::new(ctx.create_buffer(bytes).unwrap())
        }
    };
    let mut state = AttentionState {
        element_dim: width,
        data_type: DataType::BF16,
        cache: KVCacheState::full(),
        is_sparse: sparse,
        keys: make(),
        values: make(),
        kv_cache_update: <B::Kernels as Kernels>::KVCacheUpdateKernel::new(ctx.as_ref(), DataType::BF16).unwrap(),
    };
    state.prepare(0, n).unwrap();
    let mut cb = ctx.create_command_buffer(None, None).unwrap();
    cb.encode_fill(state.keys.as_mut(), 0x5a);
    cb.encode_fill(state.values.as_mut(), 0xa5);
    let mut keys = ctx.create_buffer(bytes).unwrap();
    let mut values = ctx.create_buffer(bytes).unwrap();
    cb.encode_copy(state.keys.as_ref(), &mut keys);
    cb.encode_copy(state.values.as_ref(), &mut values);
    cb.end_encoding().submit().wait_until_completed().unwrap();
    assert!(keys.copyout::<u8>().iter().all(|&x| x == 0x5a));
    assert!(values.copyout::<u8>().iter().all(|&x| x == 0xa5));
    // Retain the real allocation boundary for both sparse and dense caches.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.prepare(1, n))).is_err());
}

#[uzu_test]
fn whole_prompt_prepare_maps_all_kv_pages() {
    crate::tests::helpers::for_each_backend!(|B| whole_prompt_prepare::<B>());
}
