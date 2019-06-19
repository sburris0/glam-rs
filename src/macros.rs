#[macro_use]
#[allow(unused_macros)]
macro_rules! _ident_index {
    (x) => {
        0
    };
    (y) => {
        1
    };
    (z) => {
        2
    };
    (w) => {
        3
    };
}

#[allow(unused_macros)]
macro_rules! swizzle_mask {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        (($w & 0x3) << 6) | (($z & 0x3) << 4) | (($y & 0x3) << 2) | ($x & 0x3)
    };
    ([$x:ident, $y:ident, $z:ident, $w:ident]) => {
        swizzle_mask!(_ident_index!($x), _ident_index!($y), _ident_index!($z), _ident_index!($w))
    };
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
macro_rules! swizzle_sse2 {
    ($v:expr, $x:ident, $y:ident, $z:ident, $w:ident) => {
        swizzle_sse2!($v, swizzle_mask!($x, $y, $z, $w))
    };
    ($v:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        swizzle_sse2!($v, swizzle_mask!($x, $y, $z, $w))
    };
    ($v:expr, $mask:expr) => {
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128($v), $mask & 0xff))
        // _mm_shuffle_ps($v, $v, $mask & 0xff)
    };
}

// #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
// macro_rules! swizzle_vec4 {
//     ($v:expr, $x:ident, $y:ident, $z:ident, $w:ident) => {
//         swizzle_vec4!($v, swizzle_mask!($x, $y, $z, $w))
//     };
//     ($v:expr, $x:literal, $y:literal, $z:literal, $w:literal) => {
//         swizzle_vec4!($v, swizzle_mask!($x, $y, $z, $w))
//     };
//     ($v:expr, $mask:expr) => {
//         Vec4(unsafe { swizzle_sse2!(*(&$v as *const Vec4 as *const __m128), $mask) })
//     }
// }

// #[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
// macro_rules! swizzle {
// }

#[test]
fn test_ident_index() {
    assert_eq!(_ident_index!(x), 0);
    assert_eq!(_ident_index!(y), 1);
    assert_eq!(_ident_index!(z), 2);
    assert_eq!(_ident_index!(w), 3);
    // assert_eq!(swizzle_mask!([x, y, z, w]), 0b00_01_10_11);
}
