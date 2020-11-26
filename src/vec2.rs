#[cfg(feature = "num-traits")]
use num_traits::Float;

use crate::vector_traits::*;
// use crate::DVec2Mask;
use crate::{Vec2Mask, Vec3, XY};
#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::{cmp::Ordering, f32, ops::*};

#[cfg(feature = "std")]
use std::iter::{Product, Sum};

macro_rules! impl_vec2 {
    ($new:ident, $vec2:ident, $t:ty, $mask:ident, $inner:ident) => {
        impl Default for $vec2 {
            #[inline]
            fn default() -> Self {
                Self($inner::ZERO)
            }
        }

        impl PartialEq for $vec2 {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.cmpeq(*other).all()
            }
        }

        impl PartialOrd for $vec2 {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.as_ref().partial_cmp(other.as_ref())
            }
        }

        impl From<$vec2> for $inner {
            #[inline]
            fn from(t: $vec2) -> Self {
                t.0
            }
        }

        impl From<$inner> for $vec2 {
            #[inline]
            fn from(t: $inner) -> Self {
                Self(t)
            }
        }

        /// Creates a 2D vector.
        #[inline]
        pub fn $new(x: $t, y: $t) -> $vec2 {
            $vec2::new(x, y)
        }

        impl $vec2 {
            /// Performs `is_nan` on each element of self, returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x.is_nan(), y.is_nan()]`.
            #[inline]
            pub fn is_nan_mask(self) -> $mask {
                $mask(self.0.is_nan())
            }

            /// Returns a `$vec2` with elements representing the sign of `self`.
            ///
            /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
            /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
            /// - `NAN` if the number is `NAN`
            #[inline]
            pub fn signum(self) -> Self {
                Self(self.0.signum())
            }

            /// Returns a `$vec2` containing the reciprocal `1.0/n` of each element of `self`.
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

            /// Returns whether `self` is length `1.0` or not.
            ///
            /// Uses a precision threshold of `1e-6`.
            #[inline]
            pub fn is_normalized(self) -> bool {
                is_normalized!(self)
            }

            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.x.is_finite() && self.y.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            pub fn is_nan(self) -> bool {
                self.x.is_nan() || self.y.is_nan()
            }

            /// Returns true if the absolute difference of all elements between `self`
            /// and `other` is less than or equal to `max_abs_diff`.
            ///
            /// This can be used to compare if two `$vec2`'s contain similar elements. It
            /// works best when comparing with a known value. The `max_abs_diff` that
            /// should be used used depends on the values being compared against.
            ///
            /// For more on floating point comparisons see
            /// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            #[inline]
            pub fn abs_diff_eq(self, other: Self, max_abs_diff: $t) -> bool {
                abs_diff_eq!(self, other, max_abs_diff)
            }

            /// Creates a new `$vec2`.
            #[inline]
            pub fn new(x: $t, y: $t) -> $vec2 {
                Self(Vector2::new(x, y))
            }

            /// Creates a `$vec2` with all elements set to `0.0`.
            #[inline]
            pub const fn zero() -> $vec2 {
                Self($inner::ZERO)
            }

            /// Creates a `$vec2` with all elements set to `1.0`.
            #[inline]
            pub const fn one() -> $vec2 {
                Self($inner::ONE)
            }

            /// Creates a `$vec2` with values `[x: 1.0, y: 0.0]`.
            #[inline]
            pub const fn unit_x() -> $vec2 {
                Self($inner::UNIT_X)
            }

            /// Creates a `$vec2` with values `[x: 0.0, y: 1.0]`.
            #[inline]
            pub const fn unit_y() -> $vec2 {
                Self($inner::UNIT_Y)
            }

            /// Creates a `$vec2` with all elements set to `v`.
            #[inline]
            pub fn splat(v: $t) -> $vec2 {
                Self($inner::splat(v))
            }

            /// Creates a `$vec2` from the elements in `if_true` and `if_false`, selecting which to use for
            /// each element of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false uses
            /// the element from `if_false`.
            #[inline]
            pub fn select(mask: $mask, if_true: $vec2, if_false: $vec2) -> $vec2 {
                Self($inner::select(mask.0, if_true.0, if_false.0))
            }

            /// Creates a `$vec2` from `self` and the given `z` value.
            #[inline]
            pub fn extend(self, z: $t) -> Vec3 {
                // TODO: specify Vec3 type
                Vec3::new(self.x as f32, self.y as f32, z as f32)
            }

            /// Computes the dot product of `self` and `other`.
            #[inline]
            pub fn dot(self, other: $vec2) -> $t {
                FloatVector2::dot(self.0, other.0)
            }

            /// Computes the length of `self`.
            #[inline]
            pub fn length(self) -> $t {
                self.dot(self).sqrt()
            }

            /// Computes the squared length of `self`.
            ///
            /// This is generally faster than `$vec2::length()` as it avoids a square
            /// root operation.
            #[inline]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            /// Computes `1.0 / $vec2::length()`.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline]
            pub fn length_recip(self) -> $t {
                self.length().recip()
            }

            /// Computes the Euclidean distance between two points.
            #[inline]
            pub fn distance(self, other: $vec2) -> $t {
                (self - other).length()
            }

            /// Compute the squared Euclidean distance between two points.
            #[inline]
            pub fn distance_squared(self, other: $vec2) -> $t {
                (self - other).length_squared()
            }

            /// Returns `self` normalized to length 1.0.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline]
            pub fn normalize(self) -> $vec2 {
                self * self.length_recip()
            }

            /// Returns the vertical minimum of `self` and `other`.
            ///
            /// In other words, this computes
            /// `[x: min(x1, x2), y: min(y1, y2)]`,
            /// taking the minimum of each element individually.
            #[inline]
            pub fn min(self, other: $vec2) -> $vec2 {
                Self(self.0.min(other.0))
            }

            /// Returns the vertical maximum of `self` and `other`.
            ///
            /// In other words, this computes
            /// `[x: max(x1, x2), y: max(y1, y2)]`,
            /// taking the maximum of each element individually.
            #[inline]
            pub fn max(self, other: $vec2) -> $vec2 {
                Self(self.0.max(other.0))
            }

            /// Returns the horizontal minimum of `self`'s elements.
            ///
            /// In other words, this computes `min(x, y)`.
            #[inline]
            pub fn min_element(self) -> $t {
                self.0.min_element()
            }

            /// Returns the horizontal maximum of `self`'s elements.
            ///
            /// In other words, this computes `max(x, y)`.
            #[inline]
            pub fn max_element(self) -> $t {
                self.0.max_element()
            }

            /// Performs a vertical `==` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 == x2, y1 == y2]`.
            #[inline]
            pub fn cmpeq(self, other: $vec2) -> $mask {
                $mask(self.0.cmpeq(other.0))
            }

            /// Performs a vertical `!=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 != x2, y1 != y2]`.
            #[inline]
            pub fn cmpne(self, other: $vec2) -> $mask {
                $mask(self.0.cmpne(other.0))
            }

            /// Performs a vertical `>=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 >= x2, y1 >= y2]`.
            #[inline]
            pub fn cmpge(self, other: $vec2) -> $mask {
                $mask(self.0.cmpge(other.0))
            }

            /// Performs a vertical `>` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 > x2, y1 > y2]`.
            #[inline]
            pub fn cmpgt(self, other: $vec2) -> $mask {
                $mask(self.0.cmpgt(other.0))
            }

            /// Performs a vertical `<=` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 <= x2, y1 <= y2]`.
            #[inline]
            pub fn cmple(self, other: $vec2) -> $mask {
                $mask(self.0.cmple(other.0))
            }

            /// Performs a vertical `<` comparison between `self` and `other`,
            /// returning a `$mask` of the results.
            ///
            /// In other words, this computes `[x1 < x2, y1 < y2]`.
            #[inline]
            pub fn cmplt(self, other: $vec2) -> $mask {
                $mask(self.0.cmplt(other.0))
            }

            /// Creates a `$vec2` from the first two values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than two elements long.
            #[inline]
            pub fn from_slice_unaligned(slice: &[$t]) -> Self {
                Self($inner::from_slice_unaligned(slice))
            }

            /// Writes the elements of `self` to the first two elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than two elements long.
            #[inline]
            pub fn write_to_slice_unaligned(self, slice: &mut [$t]) {
                self.0.write_to_slice_unaligned(slice)
            }

            /// Returns a `$vec2` containing the absolute value of each element of `self`.
            #[inline]
            pub fn abs(self) -> Self {
                Self(self.0.abs())
            }

            /// Returns a `$vec2` containing the nearest integer to a number for each element of `self`.
            /// Round half-way cases away from 0.0.
            #[inline]
            pub fn round(self) -> Self {
                Self(self.0.round())
            }

            /// Returns a `$vec2` containing the largest integer less than or equal to a number for each
            /// element of `self`.
            #[inline]
            pub fn floor(self) -> Self {
                Self(self.0.floor())
            }

            /// Returns a `$vec2` containing the smallest integer greater than or equal to a number for each
            /// element of `self`.
            #[inline]
            pub fn ceil(self) -> Self {
                Self(self.0.ceil())
            }

            /// Returns a `$vec2` containing `e^self` (the exponential function) for each element of `self`.
            #[inline]
            pub fn exp(self) -> Self {
                Self::new(self.x.exp(), self.y.exp())
            }

            /// Returns a `$vec2` containing each element of `self` raised to the power of `n`.
            #[inline]
            pub fn powf(self, n: $t) -> Self {
                Self::new(self.x.powf(n), self.y.powf(n))
            }

            /// Returns a `$vec2` that is equal to `self` rotated by 90 degrees.
            #[inline]
            pub fn perp(self) -> Self {
                Self(self.0.perp())
            }

            /// The perpendicular dot product of the vector and `other`.
            #[inline]
            pub fn perp_dot(self, other: $vec2) -> $t {
                self.0.perp_dot(other.0)
            }

            /// Returns the angle between two vectors, in radians.
            ///
            /// The vectors do not need to be unit length, but this function does
            /// perform a `sqrt`.
            #[inline]
            pub fn angle_between(self, other: Self) -> $t {
                self.0.angle_between(other.0)
            }
        }

        impl fmt::Display for $vec2 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "[{}, {}]", self.x, self.y)
            }
        }

        impl Div<$vec2> for $vec2 {
            type Output = Self;
            #[inline]
            fn div(self, other: $vec2) -> Self {
                Self(self.0.div(other.0))
            }
        }

        impl DivAssign<$vec2> for $vec2 {
            #[inline]
            fn div_assign(&mut self, other: $vec2) {
                self.0 = self.0.div(other.0)
            }
        }

        impl Div<$t> for $vec2 {
            type Output = Self;
            #[inline]
            fn div(self, other: $t) -> Self {
                Self(self.0.div_scalar(other))
            }
        }

        impl DivAssign<$t> for $vec2 {
            #[inline]
            fn div_assign(&mut self, other: $t) {
                self.0 = self.div_scalar(other)
            }
        }

        impl Div<$vec2> for $t {
            type Output = $vec2;
            #[inline]
            fn div(self, other: $vec2) -> $vec2 {
                $vec2($inner::splat(self).div(other.0))
            }
        }

        impl Mul<$vec2> for $vec2 {
            type Output = Self;
            #[inline]
            fn mul(self, other: $vec2) -> Self {
                Self(self.0.mul(other.0))
            }
        }

        impl MulAssign<$vec2> for $vec2 {
            #[inline]
            fn mul_assign(&mut self, other: $vec2) {
                self.0 = self.0.mul(other.0)
            }
        }

        impl Mul<$t> for $vec2 {
            type Output = Self;
            #[inline]
            fn mul(self, other: $t) -> Self {
                Self(self.0.mul_scalar(other))
            }
        }

        impl MulAssign<$t> for $vec2 {
            #[inline]
            fn mul_assign(&mut self, other: $t) {
                self.0 = self.0.mul_scalar(other)
            }
        }

        impl Mul<$vec2> for $t {
            type Output = $vec2;
            #[inline]
            fn mul(self, other: $vec2) -> $vec2 {
                $vec2($inner::splat(self).mul(other.0))
            }
        }

        impl Add for $vec2 {
            type Output = Self;
            #[inline]
            fn add(self, other: Self) -> Self {
                Self(self.0.add(other.0))
            }
        }

        impl AddAssign for $vec2 {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.0 = self.0.add(other.0)
            }
        }

        impl Sub for $vec2 {
            type Output = Self;
            #[inline]
            fn sub(self, other: $vec2) -> Self {
                Self(self.0.sub(other.0))
            }
        }

        impl SubAssign for $vec2 {
            #[inline]
            fn sub_assign(&mut self, other: $vec2) {
                self.0 = self.0.sub(other.0)
            }
        }

        impl Neg for $vec2 {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(self.0.neg())
            }
        }

        impl AsRef<[$t; 2]> for $vec2 {
            #[inline]
            fn as_ref(&self) -> &[$t; 2] {
                unsafe { &*(self as *const $vec2 as *const [$t; 2]) }
            }
        }

        impl AsMut<[$t; 2]> for $vec2 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$t; 2] {
                unsafe { &mut *(self as *mut $vec2 as *mut [$t; 2]) }
            }
        }

        impl fmt::Debug for $vec2 {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_tuple(stringify!($vec2))
                    .field(&self.x)
                    .field(&self.y)
                    .finish()
            }
        }

        impl Index<usize> for $vec2 {
            type Output = $t;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl IndexMut<usize> for $vec2 {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_mut()[index]
            }
        }

        impl From<($t, $t)> for $vec2 {
            #[inline]
            fn from(t: ($t, $t)) -> Self {
                Self($inner::from_tuple(t))
            }
        }

        impl From<$vec2> for ($t, $t) {
            #[inline]
            fn from(v: $vec2) -> Self {
                v.0.into_tuple()
            }
        }

        impl From<[$t; 2]> for $vec2 {
            #[inline]
            fn from(a: [$t; 2]) -> Self {
                Self($inner::from_array(a))
            }
        }

        impl From<$vec2> for [$t; 2] {
            #[inline]
            fn from(v: $vec2) -> Self {
                v.0.into_array()
            }
        }

        impl Deref for $vec2 {
            type Target = $inner;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self as *const Self::Target) }
            }
        }

        impl DerefMut for $vec2 {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self as *mut Self::Target) }
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Sum<&'a Self> for $vec2 {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::zero(), |a, &b| Self::add(a, b))
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Product<&'a Self> for $vec2 {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::one(), |a, &b| Self::mul(a, b))
            }
        }
    };
}

type XYF32 = XY<f32>;
type XYF64 = XY<f64>;

#[cfg(not(doc))]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), repr(C))]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec2(pub(crate) XYF32);

#[cfg(not(doc))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DVec2(pub(crate) XYF64);

#[cfg(doc)]
#[derive(Clone, Copy)]
#[repr(C)]
/// A 2-dimensional vector.
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl_vec2!(vec2, Vec2, f32, Vec2Mask, XYF32);
impl_vec2!(dvec2, DVec2, f64, Vec2Mask, XYF64);
