use crate::vector_traits::{MaskVector, MaskVector4, MaskVectorConsts};
#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::ops::*;

#[cfg(all(
    target_arch = "x86",
    target_feature = "sse2",
    not(feature = "scalar-math")
))]
use core::arch::x86::*;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "scalar-math")
))]
use core::arch::x86_64::*;

use core::{cmp::Ordering, hash};

macro_rules! impl_vec4mask {
    ($vec4mask:ident, $inner:ident) => {
        impl Default for $vec4mask {
            #[inline]
            fn default() -> Self {
                Self($inner::FALSE)
            }
        }

        impl PartialEq for $vec4mask {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_ref().eq(other.as_ref())
            }
        }

        impl Eq for $vec4mask {}

        impl Ord for $vec4mask {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                self.as_ref().cmp(other.as_ref())
            }
        }

        impl PartialOrd for $vec4mask {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl hash::Hash for $vec4mask {
            #[inline]
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                self.as_ref().hash(state);
            }
        }

        impl $vec4mask {
            /// Creates a new `$vec4mask`.
            #[inline]
            pub fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
                Self(MaskVector4::new(x, y, z, w))
            }

            /// Returns a bitmask with the lowest four bits set from the elements of `self`.
            ///
            /// A true element results in a `1` bit and a false element in a `0` bit.  Element `x` goes
            /// into the first lowest bit, element `y` into the second, etc.
            #[inline]
            pub fn bitmask(self) -> u32 {
                self.0.bitmask()
            }

            /// Returns true if any of the elements are true, false otherwise.
            ///
            /// In other words: `x || y || z || w`.
            #[inline]
            pub fn any(self) -> bool {
                self.0.any()
            }

            /// Returns true if all the elements are true, false otherwise.
            ///
            /// In other words: `x && y && z && w`.
            #[inline]
            pub fn all(self) -> bool {
                self.0.all()
            }
        }

        impl BitAnd for $vec4mask {
            type Output = Self;
            #[inline]
            fn bitand(self, other: Self) -> Self {
                Self(self.0.bitand(other.0))
            }
        }

        impl BitAndAssign for $vec4mask {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                self.0 = self.0.bitand(other.0);
            }
        }

        impl BitOr for $vec4mask {
            type Output = Self;
            #[inline]
            fn bitor(self, other: Self) -> Self {
                Self(self.0.bitor(other.0))
            }
        }

        impl BitOrAssign for $vec4mask {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                self.0 = self.0.bitor(other.0);
            }
        }

        impl Not for $vec4mask {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                Self(self.0.not())
            }
        }

        impl fmt::Debug for $vec4mask {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let arr = self.as_ref();
                write!(
                    f,
                    "{}({:#x}, {:#x}, {:#x}, {:#x})",
                    stringify!($vec4mask),
                    arr[0],
                    arr[1],
                    arr[2],
                    arr[3]
                )
            }
        }

        impl fmt::Display for $vec4mask {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let arr = self.as_ref();
                write!(
                    f,
                    "[{}, {}, {}, {}]",
                    arr[0] != 0,
                    arr[1] != 0,
                    arr[2] != 0,
                    arr[3] != 0
                )
            }
        }

        impl From<$vec4mask> for [u32; 4] {
            #[inline]
            fn from(mask: $vec4mask) -> Self {
                *mask.as_ref()
            }
        }

        impl From<$vec4mask> for $inner {
            #[inline]
            fn from(t: $vec4mask) -> Self {
                t.0
            }
        }

        impl AsRef<[u32; 4]> for $vec4mask {
            #[inline]
            fn as_ref(&self) -> &[u32; 4] {
                unsafe { &*(self as *const Self as *const [u32; 4]) }
            }
        }
    };
}

/// A 4-dimensional vector mask.
///
/// This type is typically created by comparison methods on `Vec4`.  It is
/// essentially a vector of four boolean values.
#[cfg(doc)]
#[repr(C)]
pub struct Vec4Mask(u32, u32, u32, u32);

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
type XYZWU32 = crate::XYZW<u32>;

#[cfg(not(doc))]
#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec4Mask(pub(crate) __m128);

#[cfg(not(doc))]
#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec4Mask(pub(crate) XYZWU32);

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl_vec4mask!(Vec4Mask, __m128);

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
impl_vec4mask!(Vec4Mask, XYZWU32);
