// #[macro_use]

// Use this to make it simpler to handle the branch between using `sse2` or
// not.
//
// The SIMD path is automatically wrapped in `unsafe`. Also, the cargo feature
// `debug_force_no_sse2` will cause this macro to pick the non-sse2 path even
// if the CPU supports it. Use this for easy testing of the fallback path.
// macro_rules! if_sse2 {
//     ($ib:block else $eb:block) => {
//         #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
//         $ib
//         #[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
//         $eb
//     };
// }

// #[test]
// fn test_if_sse2_macro() {
//   if_sse2! {{
//     assert!(cfg!(target_feature="sse2") && !cfg!(feature="scalar-math"))
//   } else {
//     assert!(!(cfg!(target_feature="sse2") && !cfg!(feature="scalar-math")))
//   }}
// }
