//! F32 cascade Welford moments, with the reduced-float lane layout used by
//! PyTorch RowwiseMoments. Adapted from ATen moments_utils.h (BSD-3-Clause);
//! see THIRD_PARTY_DECISION_NOTICES.txt. No PyTorch runtime dependency.
#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: [f32; 4],
    m2: [f32; 4],
}
impl Moments {
    fn merge(
        &mut self,
        other: Self,
    ) {
        let count = self.count + other.count;
        let factor = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..4 {
            let delta = other.mean[lane] - self.mean[lane];
            let weighted = factor * delta;
            self.mean[lane] += weighted;
            self.m2[lane] = (delta * self.count as f32).mul_add(weighted, self.m2[lane] + other.m2[lane]);
        }
        self.count = count;
    }
}
pub(super) fn bf16_moments(
    length: usize,
    mut value: impl FnMut(usize) -> f32,
) -> (f32, f32) {
    let vectors = length / 8;
    let chunks = vectors.div_ceil(16);
    let depth = (usize::BITS - chunks.saturating_sub(1).leading_zeros()).max(1) as usize;
    let mut stack = vec![Moments::default(); depth];
    for chunk in 0..chunks {
        let count = (vectors - chunk * 16).min(16);
        let mut halves = [Moments::default(); 2];
        for vector in 0..count {
            let factor = 1.0 / (vector + 1) as f32;
            for (half, moment) in halves.iter_mut().enumerate() {
                for lane in 0..4 {
                    let x = value((chunk * 16 + vector) * 8 + half * 4 + lane);
                    let delta = x - moment.mean[lane];
                    moment.mean[lane] = delta.mul_add(factor, moment.mean[lane]);
                    moment.m2[lane] = delta.mul_add(x - moment.mean[lane], moment.m2[lane]);
                }
            }
        }
        for mut half in halves {
            half.count = count;
            stack[0].merge(half);
        }
        let mut mask = chunk + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let lower = std::mem::take(&mut stack[level - 1]);
            stack[level].merge(lower);
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let extra = stack[level];
        stack[0].merge(extra);
    }
    let (mut count, mut mean, mut m2) = (0usize, 0.0f32, 0.0f32);
    for index in vectors * 8..length {
        let x = value(index);
        let delta = x - mean;
        count += 1;
        mean += delta / count as f32;
        m2 += delta * (x - mean);
    }
    let add_count = vectors * 2;
    for lane in 0..4 {
        let total = count + add_count;
        let factor = if total == 0 {
            0.0
        } else {
            add_count as f32 / total as f32
        };
        let delta = stack[0].mean[lane] - mean;
        mean += factor * delta;
        m2 += stack[0].m2[lane] + delta * delta * factor * count as f32;
        count = total;
    }
    (mean, m2 / length as f32)
}
