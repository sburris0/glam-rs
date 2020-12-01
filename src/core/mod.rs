pub mod matrix_traits;
pub mod quaternion_traits;
pub mod scalar_traits;
pub mod storage;
pub mod vector_traits;

mod impl_scalar;
#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
mod impl_sse2;
