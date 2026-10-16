//! Fixed-order F32 partial reductions for BF16 operands. Independent partial
//! sums avoid serial reduction drift at the final BF16 rounding boundary.
#[inline]
pub(crate) fn bf16_vector_dot(
    length: usize,
    mut operands: impl FnMut(usize) -> (f32, f32),
) -> f32 {
    let mut sums = [[0.0f32; 4]; 8];
    let aligned = length / 32 * 32;
    for base in (0..aligned).step_by(32) {
        for (group, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let (a, b) = operands(base + group * 4 + lane);
                *sum = a.mul_add(b, *sum);
            }
        }
    }
    for width in [4, 2, 1] {
        for group in 0..width {
            for lane in 0..4 {
                sums[group][lane] += sums[group + width][lane];
            }
        }
    }
    let mut result = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
    let mut tail = [0.0f32; 4];
    let vector_end = length / 8 * 8;
    for base in (aligned..vector_end).step_by(8) {
        for half in 0..2 {
            for (lane, sum) in tail.iter_mut().enumerate() {
                let (a, b) = operands(base + half * 4 + lane);
                *sum = a.mul_add(b, *sum);
            }
        }
    }
    result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for index in vector_end..length {
        let (a, b) = operands(index);
        result += a * b;
    }
    result
}

#[inline]
pub(crate) fn bf16_strided_dot(
    length: usize,
    mut operands: impl FnMut(usize) -> (f32, f32),
) -> f32 {
    let mut sums = [0.0f32; 4];
    let aligned = length / 4 * 4;
    for base in (0..aligned).step_by(4) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let (a, b) = operands(base + lane);
            *sum += a * b;
        }
    }
    for index in aligned..length {
        let (a, b) = operands(index);
        sums[0] += a * b;
    }
    ((sums[0] + sums[1]) + sums[2]) + sums[3]
}
