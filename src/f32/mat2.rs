use crate::core::matrix_traits::*;
use crate::Vec2;
#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::ops::{Add, Deref, DerefMut, Mul, Sub};

#[cfg(feature = "std")]
use std::iter::{Product, Sum};

const ZERO: Mat2 = const_mat2!([0.0; 4]);
const IDENTITY: Mat2 = const_mat2!([1.0, 0.0], [0.0, 1.0]);

/// Creates a `Mat2` from two column vectors.
#[inline]
pub fn mat2(x_axis: Vec2, y_axis: Vec2) -> Mat2 {
    Mat2::from_cols(x_axis, y_axis)
}

/// A 2x2 column major matrix.
#[cfg(doc)]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Mat2 {
    pub x_axis: Vec2,
    pub y_axis: Vec2,
}

// #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
type InnerF32 = crate::core::storage::XYAxes<f32>;

// #[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
// type InnerF32 = __m128;

#[cfg(not(doc))]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Mat2(pub(crate) InnerF32);

impl Default for Mat2 {
    #[inline]
    fn default() -> Self {
        Self(InnerF32::IDENTITY)
    }
}

#[cfg(not(target_arch = "spirv"))]
impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
}

impl Mat2 {
    /// Creates a 2x2 matrix with all elements set to `0.0`.
    #[inline]
    pub const fn zero() -> Self {
        ZERO
    }

    /// Creates a 2x2 identity matrix.
    #[inline]
    pub const fn identity() -> Self {
        IDENTITY
    }

    /// Creates a 2x2 matrix from two column vectors.
    #[inline]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self(InnerF32::from_cols(x_axis.0, y_axis.0))
    }

    /// Creates a 2x2 matrix from a `[f32; 4]` stored in column major order.  If
    /// your data is stored in row major you will need to `transpose` the
    /// returned matrix.
    #[inline]
    pub fn from_cols_array(m: &[f32; 4]) -> Self {
        Self(InnerF32::from_cols_array(m))
    }

    /// Creates a `[f32; 4]` storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    pub fn to_cols_array(&self) -> [f32; 4] {
        self.0.to_cols_array()
    }

    /// Creates a 2x2 matrix from a `[[f32; 2]; 2]` stored in column major
    /// order.  If your data is in row major order you will need to `transpose`
    /// the returned matrix.
    #[inline]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Mat2(InnerF32::from_cols_array_2d(m))
    }

    /// Creates a `[[f32; 2]; 2]` storing data in column major order.
    /// If you require data in row major order `transpose` the matrix first.
    #[inline]
    pub fn to_cols_array_2d(&self) -> [[f32; 2]; 2] {
        self.0.to_cols_array_2d()
    }

    /// Creates a 2x2 matrix containing the given `scale` and rotation of
    /// `angle` (in radians).
    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        Self(InnerF32::from_scale_angle(scale.0, angle))
    }

    /// Creates a 2x2 matrix containing a rotation of `angle` (in radians).
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        Self(InnerF32::from_angle(angle))
    }

    /// Creates a 2x2 matrix containing the given non-uniform `scale`.
    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self(InnerF32::from_scale(scale.0))
    }

    // #[inline]
    // pub(crate) fn col(&self, index: usize) -> Vec2 {
    //     match index {
    //         0 => self.x_axis(),
    //         1 => self.y_axis(),
    //         _ => panic!(
    //             "index out of bounds: the len is 2 but the index is {}",
    //             index
    //         ),
    //     }
    // }

    // #[inline]
    // pub(crate) fn col_mut(&mut self, index: usize) -> &mut Vec2 {
    //     match index {
    //         0 => unsafe { &mut *(self.0.as_mut().as_mut_ptr() as *mut Vec2) },
    //         1 => unsafe { &mut *(self.0.as_mut()[2..].as_mut_ptr() as *mut Vec2) },
    //         _ => panic!(
    //             "index out of bounds: the len is 2 but the index is {}",
    //             index
    //         ),
    //     }
    // }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(&self) -> bool {
        // TODO
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(&self) -> Self {
        Self(self.0.transpose())
        // #[cfg(vec4_sse2)]
        // unsafe {
        //     let abcd = self.0.into();
        //     let acbd = _mm_shuffle_ps(abcd, abcd, 0b11_01_10_00);
        //     Self(acbd.into())
        // }
    }

    /// Returns the determinant of `self`.
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.0.determinant()
        // #[cfg(vec4_sse2)]
        // unsafe {
        //     let abcd = self.0.into();
        //     let dcba = _mm_shuffle_ps(abcd, abcd, 0b00_01_10_11);
        //     let prod = _mm_mul_ps(abcd, dcba);
        //     let det = _mm_sub_ps(prod, _mm_shuffle_ps(prod, prod, 0b01_01_01_01));
        //     _mm_cvtss_f32(det)
        // }
    }

    /// Returns the inverse of `self`.
    ///
    /// If the matrix is not invertible the returned matrix will be invalid.
    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
        // #[cfg(vec4_sse2)]
        // unsafe {
        //     const SIGN: __m128 = const_m128!([1.0, -1.0, -1.0, 1.0]);
        //     let abcd = self.0.into();
        //     let dcba = _mm_shuffle_ps(abcd, abcd, 0b00_01_10_11);
        //     let prod = _mm_mul_ps(abcd, dcba);
        //     let sub = _mm_sub_ps(prod, _mm_shuffle_ps(prod, prod, 0b01_01_01_01));
        //     let det = _mm_shuffle_ps(sub, sub, 0b00_00_00_00);
        //     let tmp = _mm_div_ps(SIGN, det);
        //     let dbca = _mm_shuffle_ps(abcd, abcd, 0b00_10_01_11);
        //     Self(_mm_mul_ps(dbca, tmp).into())
        // }
    }

    /// Transforms a `Vec2`.
    #[inline]
    pub fn mul_vec2(&self, other: Vec2) -> Vec2 {
        Vec2(self.0.mul_vector(other.0))
    }

    /// Multiplies two 2x2 matrices.
    #[inline]
    pub fn mul_mat2(&self, other: &Self) -> Self {
        Mat2(self.0.mul_matrix(&other.0))
    }

    /// Adds two 2x2 matrices.
    #[inline]
    pub fn add_mat2(&self, other: &Self) -> Self {
        Mat2(self.0.add_matrix(&other.0))
    }

    /// Subtracts two 2x2 matrices.
    #[inline]
    pub fn sub_mat2(&self, other: &Self) -> Self {
        Mat2(self.0.sub_matrix(&other.0))
    }

    /// Multiplies a 2x2 matrix by a scalar.
    #[inline]
    pub fn mul_scalar(&self, other: f32) -> Self {
        Mat2(self.0.mul_scalar(other))
    }

    /// Returns true if the absolute difference of all elements between `self`
    /// and `other` is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two `Mat2`'s contain similar elements. It
    /// works best when comparing with a known value. The `max_abs_diff` that
    /// should be used used depends on the values being compared against.
    ///
    /// For more on floating point comparisons see
    /// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    #[inline]
    pub fn abs_diff_eq(&self, other: &Self, max_abs_diff: f32) -> bool {
        self.0.abs_diff_eq(&other.0, max_abs_diff)
    }
}

impl fmt::Debug for Mat2 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Mat2")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}

impl AsRef<[f32; 4]> for Mat2 {
    #[inline]
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
}

impl AsMut<[f32; 4]> for Mat2 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
}

impl Add<Mat2> for Mat2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0.add_matrix(&other.0))
    }
}

impl Sub<Mat2> for Mat2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0.sub_matrix(&other.0))
    }
}

impl Mul<Mat2> for Mat2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self(self.0.mul_matrix(&other.0))
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, other: Vec2) -> Vec2 {
        Vec2(self.0.mul_vector(other.0))
    }
}

impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline]
    fn mul(self, other: Mat2) -> Mat2 {
        Mat2(other.0.mul_scalar(self))
    }
}

impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: f32) -> Self {
        Self(self.0.mul_scalar(other))
    }
}

impl Deref for Mat2 {
    type Target = crate::XYAxes;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl DerefMut for Mat2 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

#[cfg(feature = "std")]
impl<'a> Sum<&'a Self> for Mat2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(ZERO, |a, &b| Self::add(a, b))
    }
}

#[cfg(feature = "std")]
impl<'a> Product<&'a Self> for Mat2 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(IDENTITY, |a, &b| Self::mul(a, b))
    }
}
