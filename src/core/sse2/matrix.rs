#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{
    const_m128,
    core::{
        storage::{XYAxes, XY},
        traits::{
            matrix::{FloatMatrix2x2, Matrix, Matrix2x2, MatrixConsts},
            vector::{FloatVector4, Vector2},
        },
    },
};

impl MatrixConsts for __m128 {
    const IDENTITY: __m128 = const_m128!([1.0, 0.0, 0.0, 1.0]);
}

impl Matrix<f32> for __m128 {}

impl Matrix2x2<f32> for __m128 {
    #[inline(always)]
    fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        unsafe { _mm_set_ps(m11, m10, m01, m00) }
    }

    #[inline(always)]
    fn deref(&self) -> &XYAxes<f32> {
        unsafe { &*(self as *const Self as *const XYAxes<f32>) }
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYAxes<f32> {
        unsafe { &mut *(self as *mut Self as *mut XYAxes<f32>) }
    }

    // #[inline(always)]
    // fn to_cols_array(&self) -> [f32; 4] {
    //     // TODO: MaybeUninit
    //     [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    // }

    // #[inline(always)]
    // fn to_cols_array_2d(&self) -> [[f32; 2]; 2] {
    //     // TODO: MaybeUninit
    //     [
    //         [self.x_axis.x, self.x_axis.y],
    //         [self.y_axis.x, self.y_axis.y],
    //     ]
    // }

    // #[inline(always)]
    // fn from_scale(scale: XY<f32>) -> Self {
    //     Self::new(scale.x, 0.0, 0.0, scale.y)
    // }

    #[inline]
    fn determinant(&self) -> f32 {
        // self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
        unsafe {
            let abcd = *self;
            let dcba = _mm_shuffle_ps(abcd, abcd, 0b00_01_10_11);
            let prod = _mm_mul_ps(abcd, dcba);
            let det = _mm_sub_ps(prod, _mm_shuffle_ps(prod, prod, 0b01_01_01_01));
            _mm_cvtss_f32(det)
        }
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        unsafe { _mm_shuffle_ps(*self, *self, 0b11_01_10_00) }
    }

    #[inline]
    fn mul_vector(&self, other: XY<f32>) -> XY<f32> {
        // TODO: SSE2
        let m = self.deref();
        Vector2::new(
            (m.x_axis.x * other.x) + (m.y_axis.x * other.y),
            (m.x_axis.y * other.x) + (m.y_axis.y * other.y),
        )
    }

    #[inline]
    fn mul_matrix(&self, other: &Self) -> Self {
        // TODO: SSE2
        let other = other.deref();
        Self::from_cols(self.mul_vector(other.x_axis), self.mul_vector(other.y_axis))
    }

    #[inline]
    fn mul_scalar(&self, other: f32) -> Self {
        unsafe { _mm_mul_ps(*self, _mm_set_ps1(other)) }
    }

    #[inline]
    fn add_matrix(&self, other: &Self) -> Self {
        unsafe { _mm_add_ps(*self, *other) }
    }

    #[inline]
    fn sub_matrix(&self, other: &Self) -> Self {
        unsafe { _mm_sub_ps(*self, *other) }
    }
}

impl FloatMatrix2x2<f32> for __m128 {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: f32) -> bool {
        FloatVector4::abs_diff_eq(*self, *other, max_abs_diff)
    }

    #[inline]
    fn inverse(&self) -> Self {
        unsafe {
            const SIGN: __m128 = const_m128!([1.0, -1.0, -1.0, 1.0]);
            let abcd = *self;
            let dcba = _mm_shuffle_ps(abcd, abcd, 0b00_01_10_11);
            let prod = _mm_mul_ps(abcd, dcba);
            let sub = _mm_sub_ps(prod, _mm_shuffle_ps(prod, prod, 0b01_01_01_01));
            let det = _mm_shuffle_ps(sub, sub, 0b00_00_00_00);
            let tmp = _mm_div_ps(SIGN, det);
            let dbca = _mm_shuffle_ps(abcd, abcd, 0b00_10_01_11);
            _mm_mul_ps(dbca, tmp)
        }
    }
}
