#[cfg(feature = "num-traits")]
use num_traits::Float;

use crate::core::vector_traits::*;

use crate::{DVec2, DVec3, DVec4Mask};
use crate::{Vec2, Vec3, Vec3A, Vec4Mask, XYZW};
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

#[cfg(feature = "std")]
use std::iter::{Product, Sum};

use core::{cmp::Ordering, f32};

macro_rules! impl_vec4 {
    ($new:ident, $vec2:ident, $vec3:ident, $vec4:ident, $t:ty, $mask:ident, $inner:ident) => {
        impl Default for $vec4 {
            #[inline]
            fn default() -> Self {
                Self($inner::ZERO)
            }
        }

        impl PartialEq for $vec4 {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.cmpeq(*other).all()
            }
        }

        impl PartialOrd for $vec4 {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.as_ref().partial_cmp(other.as_ref())
            }
        }

        impl From<$vec4> for $inner {
            #[inline]
            fn from(t: $vec4) -> Self {
                t.0
            }
        }

        impl From<$inner> for $vec4 {
            #[inline]
            fn from(t: $inner) -> Self {
                Self(t)
            }
        }

        /// Creates a `$vec4`.
        #[inline]
        pub fn $new(x: $t, y: $t, z: $t, w: $t) -> $vec4 {
            $vec4::new(x, y, z, w)
        }

        impl $vec4 {
            /// Creates a new `$vec4`.
            #[inline]
            pub fn new(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self(Vector4::new(x, y, z, w))
            }

            /// Creates a `$vec4` with all elements set to `0.0`.
            #[inline]
            pub const fn zero() -> Self {
                Self($inner::ZERO)
            }

            /// Creates a `$vec4` with all elements set to `1.0`.
            #[inline]
            pub const fn one() -> Self {
                Self($inner::ONE)
            }

            /// Creates a `$vec4` with values `[x: 1.0, y: 0.0, z: 0.0, w: 0.0]`.
            #[inline]
            pub const fn unit_x() -> Self {
                Self(Vector4Consts::UNIT_X)
            }

            /// Creates a `$vec4` with values `[x: 0.0, y: 1.0, z: 0.0, w: 0.0]`.
            #[inline]
            pub const fn unit_y() -> Self {
                Self(Vector4Consts::UNIT_Y)
            }

            /// Creates a `$vec4` with values `[x: 0.0, y: 0.0, z: 1.0, w: 0.0]`.
            #[inline]
            pub const fn unit_z() -> Self {
                Self(Vector4Consts::UNIT_Z)
            }

            /// Creates a `$vec4` with values `[x: 0.0, y: 0.0, z: 0.0, w: 1.0]`.
            #[inline]
            pub const fn unit_w() -> Self {
                Self(Vector4Consts::UNIT_W)
            }

            /// Creates a `$vec4` with all elements set to `v`.
            #[inline]
            pub fn splat(v: $t) -> Self {
                Self($inner::splat(v))
            }

            /// Creates a `$vec4` from the elements in `if_true` and `if_false`, selecting which to use for
            /// each element of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false uses
            /// the element from `if_false`.
            #[inline]
            pub fn select(mask: $mask, if_true: $vec4, if_false: $vec4) -> $vec4 {
                Self($inner::select(mask.0, if_true.0, if_false.0))
            }

            /// Creates a `Vec3` from the `x`, `y` and `z` elements of `self`, discarding `w`.
            ///
            /// Truncation to `Vec3` may also be performed by using `self.xyz()` or `Vec3::from()`.
            ///
            /// To truncate to `Vec3A` use `Vec3A::from()`.
            #[inline]
            pub fn truncate(self) -> $vec3 {
                $vec3::new(self.x, self.y, self.z)
            }

            /// Computes the 4D dot product of `self` and `other`.
            #[inline]
            pub fn dot(self, other: Self) -> $t {
                FloatVector4::dot(self.0, other.0)
            }

            /// Computes the 4D length of `self`.
            #[inline]
            pub fn length(self) -> $t {
                FloatVector4::length(self.0)
            }

            /// Computes the squared 4D length of `self`.
            ///
            /// This is generally faster than `$vec4::length()` as it avoids a square
            /// root operation.
            #[inline]
            pub fn length_squared(self) -> $t {
                FloatVector4::dot(self.0, self.0)
            }

            /// Computes `1.0 / $vec4::length()`.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline]
            pub fn length_recip(self) -> $t {
                FloatVector4::length_recip(self.0)
            }

            /// Computes the Euclidean distance between two points in space.
            #[inline]
            pub fn distance(self, other: $vec4) -> $t {
                (self - other).length()
            }

            /// Compute the squared euclidean distance between two points in space.
            #[inline]
            pub fn distance_squared(self, other: $vec4) -> $t {
                (self - other).length_squared()
            }

            /// Returns `self` normalized to length 1.0.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline]
            pub fn normalize(self) -> Self {
                Self(FloatVector4::normalize(self.0))
            }

            /// Returns the vertical minimum of `self` and `other`.
            ///
            /// In other words, this computes
            /// `[x: min(x1, x2), y: min(y1, y2), z: min(z1, z2), w: min(w1, w2)]`,
            /// taking the minimum of each element individually.
            #[inline]
            pub fn min(self, other: Self) -> Self {
                Self(self.0.min(other.0))
            }

            /// Returns the vertical maximum of `self` and `other`.
            ///
            /// In other words, this computes
            /// `[x: max(x1, x2), y: max(y1, y2), z: max(z1, z2), w: max(w1, w2)]`,
            /// taking the maximum of each element individually.
            #[inline]
            pub fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }

            /// Returns the horizontal minimum of `self`'s elements.
            ///
            /// In other words, this computes `min(x, y, z, w)`.
            #[inline]
            pub fn min_element(self) -> $t {
                self.0.min_element()
            }

            /// Returns the horizontal maximum of `self`'s elements.
            ///
            /// In other words, this computes `max(x, y, z, w)`.
            #[inline]
            pub fn max_element(self) -> $t {
                self.0.max_element()
            }

            /// Performs a vertical `==` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 == x2, y1 == y2, z1 == z2, w1 == w2]`.
            #[inline]
            pub fn cmpeq(self, other: Self) -> $mask {
                $mask(self.0.cmpeq(other.0))
            }

            /// Performs a vertical `!=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 != x2, y1 != y2, z1 != z2, w1 != w2]`.
            #[inline]
            pub fn cmpne(self, other: Self) -> $mask {
                $mask(self.0.cmpne(other.0))
            }

            /// Performs a vertical `>=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 >= x2, y1 >= y2, z1 >= z2, w1 >= w2]`.
            #[inline]
            pub fn cmpge(self, other: Self) -> $mask {
                $mask(self.0.cmpge(other.0))
            }

            /// Performs a vertical `>` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 > x2, y1 > y2, z1 > z2, w1 > w2]`.
            #[inline]
            pub fn cmpgt(self, other: Self) -> $mask {
                $mask(self.0.cmpgt(other.0))
            }

            /// Performs a vertical `<=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 <= x2, y1 <= y2, z1 <= z2, w1 <= w2]`.
            #[inline]
            pub fn cmple(self, other: Self) -> $mask {
                $mask(self.0.cmple(other.0))
            }

            /// Performs a vertical `<` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 < x2, y1 < y2, z1 < z2, w1 < w2]`.
            #[inline]
            pub fn cmplt(self, other: Self) -> $mask {
                $mask(self.0.cmplt(other.0))
            }

            /// Creates a `$vec4` from the first four values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than four elements long.
            #[inline]
            pub fn from_slice_unaligned(slice: &[$t]) -> Self {
                Self(Vector4::from_slice_unaligned(slice))
            }

            /// Writes the elements of `self` to the first four elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than four elements long.
            #[inline]
            pub fn write_to_slice_unaligned(self, slice: &mut [$t]) {
                Vector4::write_to_slice_unaligned(self.0, slice)
            }

            /// Per element multiplication/addition of the three inputs: b + (self * a)
            #[inline]
            #[allow(dead_code)]
            pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
                Self(self.0.mul_add(a.0, b.0))
            }

            /// Returns a `$vec4` containing the absolute value of each element of `self`.
            #[inline]
            pub fn abs(self) -> Self {
                Self(self.0.abs())
            }

            /// Returns a `$vec4` containing the nearest integer to a number for each element of `self`.
            /// Round half-way cases away from 0.0.
            #[inline]
            pub fn round(self) -> Self {
                Self(self.0.round())
            }

            /// Returns a `$vec4` containing the largest integer less than or equal to a number for each
            /// element of `self`.
            #[inline]
            pub fn floor(self) -> Self {
                Self(self.0.floor())
            }

            /// Returns a `$vec4` containing the smallest integer greater than or equal to a number for each
            /// element of `self`.
            #[inline]
            pub fn ceil(self) -> Self {
                Self(self.0.ceil())
            }

            /// Returns a `$vec4` containing `e^self` (the exponential function) for each element of `self`.
            #[inline]
            pub fn exp(self) -> Self {
                Self::new(self.x.exp(), self.y.exp(), self.z.exp(), self.w.exp())
            }

            /// Returns a `$vec4` containing each element of `self` raised to the power of `n`.
            #[inline]
            pub fn powf(self, n: $t) -> Self {
                Self::new(
                    self.x.powf(n),
                    self.y.powf(n),
                    self.z.powf(n),
                    self.w.powf(n),
                )
            }

            /// Performs `is_nan` on each element of self, returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x.is_nan(), y.is_nan(), z.is_nan(), w.is_nan()]`.
            #[inline]
            pub fn is_nan_mask(self) -> $mask {
                $mask(self.0.is_nan())
            }

            /// Returns a `$vec4` with elements representing the sign of `self`.
            ///
            /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
            /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
            /// - `NAN` if the number is `NAN`
            #[inline]
            pub fn signum(self) -> Self {
                Self(self.0.signum())
            }

            /// Returns a `$vec4` containing the reciprocal `1.0/n` of each element of `self`.
            #[inline]
            pub fn recip(self) -> Self {
                Self(self.0.recip())
            }

            /// Performs a linear interpolation between `self` and `other` based on
            /// the value `s`.
            ///
            /// When `s` is `0.0`, the result will be equal to `self`.  When `s`
            /// is `1.0`, the result will be equal to `other`.
            #[inline]
            pub fn lerp(self, other: Self, s: $t) -> Self {
                self + ((other - self) * s)
            }

            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
            #[inline]
            pub fn is_finite(self) -> bool {
                // TODO: SIMD implementation
                self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            pub fn is_nan(self) -> bool {
                MaskVector4::all(FloatVector::is_nan(self.0))
            }

            /// Returns whether `self` is length `1.0` or not.
            ///
            /// Uses a precision threshold of `1e-6`.
            #[inline]
            pub fn is_normalized(self) -> bool {
                FloatVector4::is_normalized(self.0)
            }

            /// Returns true if the absolute difference of all elements between `self`
            /// and `other` is less than or equal to `max_abs_diff`.
            ///
            /// This can be used to compare if two `$vec4`'s contain similar elements. It
            /// works best when comparing with a known value. The `max_abs_diff` that
            /// should be used used depends on the values being compared against.
            ///
            /// For more on floating point comparisons see
            /// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            #[inline]
            pub fn abs_diff_eq(self, other: Self, max_abs_diff: $t) -> bool {
                FloatVector4::abs_diff_eq(self.0, other.0, max_abs_diff)
            }
        }

        impl AsRef<[$t; 4]> for $vec4 {
            #[inline]
            fn as_ref(&self) -> &[$t; 4] {
                unsafe { &*(self as *const Self as *const [$t; 4]) }
            }
        }

        impl AsMut<[$t; 4]> for $vec4 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$t; 4] {
                unsafe { &mut *(self as *mut Self as *mut [$t; 4]) }
            }
        }

        impl fmt::Debug for $vec4 {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                let a = self.as_ref();
                fmt.debug_tuple(stringify!($vec4))
                    .field(&a[0])
                    .field(&a[1])
                    .field(&a[2])
                    .field(&a[3])
                    .finish()
            }
        }

        impl fmt::Display for $vec4 {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                let a = self.as_ref();
                write!(fmt, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
            }
        }

        impl Div<$vec4> for $vec4 {
            type Output = Self;
            #[inline]
            fn div(self, other: Self) -> Self {
                Self(self.0.div(other.0))
            }
        }

        impl DivAssign<$vec4> for $vec4 {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                self.0 = self.0.div(other.0);
            }
        }

        impl Div<$t> for $vec4 {
            type Output = Self;
            #[inline]
            fn div(self, other: $t) -> Self {
                // TODO: add div by scalar to inner?
                Self(self.0.div($inner::splat(other)))
            }
        }

        impl DivAssign<$t> for $vec4 {
            #[inline]
            fn div_assign(&mut self, other: $t) {
                self.0 = self.0.div($inner::splat(other));
            }
        }

        impl Div<$vec4> for $t {
            type Output = $vec4;
            #[inline]
            fn div(self, other: $vec4) -> $vec4 {
                $vec4($inner::splat(self).div(other.0))
            }
        }

        impl Mul<$vec4> for $vec4 {
            type Output = Self;
            #[inline]
            fn mul(self, other: Self) -> Self {
                Self(self.0.mul(other.0))
            }
        }

        impl MulAssign<$vec4> for $vec4 {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                self.0 = self.0.mul(other.0);
            }
        }

        impl Mul<$t> for $vec4 {
            type Output = Self;
            #[inline]
            fn mul(self, other: $t) -> Self {
                Self(self.0.mul($inner::splat(other)))
            }
        }

        impl MulAssign<$t> for $vec4 {
            #[inline]
            fn mul_assign(&mut self, other: $t) {
                self.0 = self.0.mul($inner::splat(other));
            }
        }

        impl Mul<$vec4> for $t {
            type Output = $vec4;
            #[inline]
            fn mul(self, other: $vec4) -> $vec4 {
                $vec4($inner::splat(self).mul(other.0))
            }
        }

        impl Add for $vec4 {
            type Output = Self;
            #[inline]
            fn add(self, other: Self) -> Self {
                Self(self.0.add(other.0))
            }
        }

        impl AddAssign for $vec4 {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.0 = self.0.add(other.0);
            }
        }

        impl Sub for $vec4 {
            type Output = Self;
            #[inline]
            fn sub(self, other: Self) -> Self {
                Self(self.0.sub(other.0))
            }
        }

        impl SubAssign for $vec4 {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.0 = self.0.sub(other.0);
            }
        }

        impl Neg for $vec4 {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(self.0.neg())
            }
        }

        impl Index<usize> for $vec4 {
            type Output = $t;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl IndexMut<usize> for $vec4 {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut()[index]
            }
        }

        impl From<($t, $t, $t, $t)> for $vec4 {
            #[inline]
            fn from(t: ($t, $t, $t, $t)) -> Self {
                Self(Vector4::from_tuple(t))
            }
        }

        impl From<$vec4> for ($t, $t, $t, $t) {
            #[inline]
            fn from(v: $vec4) -> Self {
                Vector4::into_tuple(v.0)
            }
        }

        impl From<[$t; 4]> for $vec4 {
            #[inline]
            fn from(a: [$t; 4]) -> Self {
                Self(Vector4::from_array(a))
            }
        }

        impl From<$vec4> for [$t; 4] {
            #[inline]
            fn from(v: $vec4) -> Self {
                v.into_array()
            }
        }

        impl From<($vec3, $t)> for $vec4 {
            #[inline]
            fn from((v, w): ($vec3, $t)) -> Self {
                Self::new(v.x, v.y, v.z, w)
            }
        }

        impl From<$vec4> for $vec3 {
            /// Creates a `Vec3` from the `x`, `y` and `z` elements of the `$vec4`, discarding `z`.
            #[inline]
            fn from(v: $vec4) -> Self {
                Self(v.into_xyz())
            }
        }

        impl From<$vec4> for $vec2 {
            /// Creates a `Vec2` from the `x` and `y` elements of the `$vec4`, discarding `z`.
            #[inline]
            fn from(v: $vec4) -> Self {
                Self(v.into_xy())
            }
        }

        impl Deref for $vec4 {
            type Target = XYZW<$t>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                Vector4::deref(&self.0)
            }
        }

        impl DerefMut for $vec4 {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                Vector4::deref_mut(&mut self.0)
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Sum<&'a Self> for $vec4 {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::zero(), |a, &b| Self::add(a, b))
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Product<&'a Self> for $vec4 {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::one(), |a, &b| Self::mul(a, b))
            }
        }
    };
}

/// A 4-dimensional `f32` vector.
///
/// This type is 16 byte aligned unless the `scalar-math` feature is enabed.
#[cfg(doc)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
type XYZWF32 = XYZW<f32>;

#[cfg(not(doc))]
#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec4(pub(crate) XYZWF32);

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
impl_vec4!(vec4, Vec2, Vec3, Vec4, f32, Vec4Mask, XYZWF32);

#[cfg(not(doc))]
#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vec4(pub(crate) __m128);

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl_vec4!(vec4, Vec2, Vec3, Vec4, f32, Vec4Mask, __m128);

impl From<Vec4> for Vec3A {
    /// Creates a `Vec3A` from the `x`, `y` and `z` elements of the `$vec4`, discarding `z`.
    ///
    /// On architectures where SIMD is supported such as SSE2 on x86_64 this conversion is a noop.
    #[inline]
    fn from(v: Vec4) -> Self {
        Self(v.0.into())
    }
}

/// A 4-dimensional `f64` vector.
#[cfg(doc)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DVec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

type XYZWF64 = XYZW<f64>;

#[cfg(not(doc))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DVec4(pub(crate) XYZWF64);

impl_vec4!(dvec4, DVec2, DVec3, DVec4, f64, DVec4Mask, XYZWF64);

#[test]
fn test_vec4_private() {
    assert_eq!(
        vec4(1.0, 1.0, 1.0, 1.0).mul_add(vec4(0.5, 2.0, -4.0, 0.0), vec4(-1.0, -1.0, -1.0, -1.0)),
        vec4(-0.5, 1.0, -5.0, -1.0)
    );
}

#[cfg(test)]
mod tests {
    use super::{Vec3, vec4};

    #[test]
    fn from_vec3() {
        assert_eq!(vec4(1.0, 2.0, 3.0, 4.0), (Vec3::new(1.0, 2.0, 3.0), 4.0).into());
    }
}
