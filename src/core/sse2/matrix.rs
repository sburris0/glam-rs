#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem::MaybeUninit;

use crate::{
    const_m128,
    core::{
        storage::{Align16, Vector2x2, Vector4x4, XY, XYZ, XYZW},
        traits::{
            matrix::{FloatMatrix2x2, FloatMatrix4x4, Matrix, Matrix2x2, Matrix4x4, MatrixConst},
            projection::ProjectionMatrix,
            vector::{FloatVector, FloatVector4, Vector, Vector4, Vector4Const},
        },
    },
};

// __m128 as a Matrix2x2
impl MatrixConst for __m128 {
    const ZERO: __m128 = const_m128!([0.0, 0.0, 0.0, 0.0]);
    const IDENTITY: __m128 = const_m128!([1.0, 0.0, 0.0, 1.0]);
}

impl Matrix<f32> for __m128 {}

impl Matrix2x2<f32, XY<f32>> for __m128 {
    #[inline(always)]
    fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        unsafe { _mm_set_ps(m11, m10, m01, m00) }
    }

    #[inline(always)]
    fn from_cols(x_axis: XY<f32>, y_axis: XY<f32>) -> Self {
        Matrix2x2::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline(always)]
    fn deref(&self) -> &Vector2x2<XY<f32>> {
        unsafe { &*(self as *const Self as *const Vector2x2<XY<f32>>) }
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Vector2x2<XY<f32>> {
        unsafe { &mut *(self as *mut Self as *mut Vector2x2<XY<f32>>) }
    }

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
        unsafe {
            let abcd = *self;
            let xxyy = _mm_set_ps(other.y, other.y, other.x, other.x);
            let axbxcydy = _mm_mul_ps(abcd, xxyy);
            let cydyaxbx = _mm_shuffle_ps(axbxcydy, axbxcydy, 0b01_00_11_10);
            let result = _mm_add_ps(axbxcydy, cydyaxbx);
            let mut out: MaybeUninit<Align16<XY<f32>>> = MaybeUninit::uninit();
            _mm_store_ps(out.as_mut_ptr() as *mut f32, result);
            out.assume_init().0
        }
    }

    #[inline]
    fn mul_matrix(&self, other: &Self) -> Self {
        // TODO: SSE2
        let other = Matrix2x2::deref(other);
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

impl FloatMatrix2x2<f32, XY<f32>> for __m128 {
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

impl MatrixConst for Vector4x4<__m128> {
    const ZERO: Vector4x4<__m128> = Vector4x4 {
        x_axis: __m128::ZERO,
        y_axis: __m128::ZERO,
        z_axis: __m128::ZERO,
        w_axis: __m128::ZERO,
    };
    const IDENTITY: Vector4x4<__m128> = Vector4x4 {
        x_axis: __m128::UNIT_X,
        y_axis: __m128::UNIT_Y,
        z_axis: __m128::UNIT_Z,
        w_axis: __m128::UNIT_W,
    };
}

impl Matrix<f32> for Vector4x4<__m128> {}

impl Matrix4x4<f32, __m128> for Vector4x4<__m128> {
    #[inline(always)]
    fn from_cols(x_axis: __m128, y_axis: __m128, z_axis: __m128, w_axis: __m128) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    #[inline(always)]
    fn x_axis(&self) -> &__m128 {
        &self.x_axis
    }

    #[inline(always)]
    fn y_axis(&self) -> &__m128 {
        &self.y_axis
    }

    #[inline(always)]
    fn z_axis(&self) -> &__m128 {
        &self.z_axis
    }

    #[inline(always)]
    fn w_axis(&self) -> &__m128 {
        &self.w_axis
    }

    #[inline(always)]
    fn deref(&self) -> &Vector4x4<XYZW<f32>> {
        unsafe { &*(self as *const Self as *const Vector4x4<XYZW<f32>>) }
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Vector4x4<XYZW<f32>> {
        unsafe { &mut *(self as *mut Self as *mut Vector4x4<XYZW<f32>>) }
    }

    fn determinant(&self) -> f32 {
        unsafe {
            // Based on https://github.com/g-truc/glm `glm_mat4_determinant`
            let swp2a = _mm_shuffle_ps(self.z_axis, self.z_axis, 0b00_01_01_10);
            let swp3a = _mm_shuffle_ps(self.w_axis, self.w_axis, 0b11_10_11_11);
            let swp2b = _mm_shuffle_ps(self.z_axis, self.z_axis, 0b11_10_11_11);
            let swp3b = _mm_shuffle_ps(self.w_axis, self.w_axis, 0b00_10_01_10);
            let swp2c = _mm_shuffle_ps(self.z_axis, self.z_axis, 0b00_00_01_10);
            let swp3c = _mm_shuffle_ps(self.w_axis, self.w_axis, 0b01_10_00_00);

            let mula = _mm_mul_ps(swp2a, swp3a);
            let mulb = _mm_mul_ps(swp2b, swp3b);
            let mulc = _mm_mul_ps(swp2c, swp3c);
            let sube = _mm_sub_ps(mula, mulb);
            let subf = _mm_sub_ps(_mm_movehl_ps(mulc, mulc), mulc);

            let subfaca = _mm_shuffle_ps(sube, sube, 0b10_01_00_00);
            let swpfaca = _mm_shuffle_ps(self.y_axis, self.y_axis, 0b00_00_00_01);
            let mulfaca = _mm_mul_ps(swpfaca, subfaca);

            let subtmpb = _mm_shuffle_ps(sube, subf, 0b00_00_11_01);
            let subfacb = _mm_shuffle_ps(subtmpb, subtmpb, 0b11_01_01_00);
            let swpfacb = _mm_shuffle_ps(self.y_axis, self.y_axis, 0b01_01_10_10);
            let mulfacb = _mm_mul_ps(swpfacb, subfacb);

            let subres = _mm_sub_ps(mulfaca, mulfacb);
            let subtmpc = _mm_shuffle_ps(sube, subf, 0b01_00_10_10);
            let subfacc = _mm_shuffle_ps(subtmpc, subtmpc, 0b11_11_10_00);
            let swpfacc = _mm_shuffle_ps(self.y_axis, self.y_axis, 0b10_11_11_11);
            let mulfacc = _mm_mul_ps(swpfacc, subfacc);

            let addres = _mm_add_ps(subres, mulfacc);
            let detcof = _mm_mul_ps(addres, _mm_setr_ps(1.0, -1.0, 1.0, -1.0));

            self.x_axis.dot(detcof)
        }
    }

    fn transpose(&self) -> Self {
        unsafe {
            // Based on https://github.com/microsoft/DirectXMath `XMMatrixTranspose`
            let tmp0 = _mm_shuffle_ps(self.x_axis, self.y_axis, 0b01_00_01_00);
            let tmp1 = _mm_shuffle_ps(self.x_axis, self.y_axis, 0b11_10_11_10);
            let tmp2 = _mm_shuffle_ps(self.z_axis, self.w_axis, 0b01_00_01_00);
            let tmp3 = _mm_shuffle_ps(self.z_axis, self.w_axis, 0b11_10_11_10);

            Self {
                x_axis: _mm_shuffle_ps(tmp0, tmp2, 0b10_00_10_00),
                y_axis: _mm_shuffle_ps(tmp0, tmp2, 0b11_01_11_01),
                z_axis: _mm_shuffle_ps(tmp1, tmp3, 0b10_00_10_00),
                w_axis: _mm_shuffle_ps(tmp1, tmp3, 0b11_01_11_01),
            }
        }
    }
}

impl FloatMatrix4x4<f32, __m128> for Vector4x4<__m128> {
    type SIMDVector3 = __m128;

    fn inverse(&self) -> Self {
        unsafe {
            // Based on https://github.com/g-truc/glm `glm_mat4_inverse`
            let fac0 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b11_11_11_11);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b10_10_10_10);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b10_10_10_10);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b11_11_11_11);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let fac1 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b11_11_11_11);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b01_01_01_01);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b01_01_01_01);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b11_11_11_11);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let fac2 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b10_10_10_10);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b01_01_01_01);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b01_01_01_01);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b10_10_10_10);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let fac3 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b11_11_11_11);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b00_00_00_00);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b00_00_00_00);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b11_11_11_11);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let fac4 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b10_10_10_10);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b00_00_00_00);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b00_00_00_00);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b10_10_10_10);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let fac5 = {
                let swp0a = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b01_01_01_01);
                let swp0b = _mm_shuffle_ps(self.w_axis, self.z_axis, 0b00_00_00_00);

                let swp00 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b00_00_00_00);
                let swp01 = _mm_shuffle_ps(swp0a, swp0a, 0b10_00_00_00);
                let swp02 = _mm_shuffle_ps(swp0b, swp0b, 0b10_00_00_00);
                let swp03 = _mm_shuffle_ps(self.z_axis, self.y_axis, 0b01_01_01_01);

                let mul00 = _mm_mul_ps(swp00, swp01);
                let mul01 = _mm_mul_ps(swp02, swp03);
                _mm_sub_ps(mul00, mul01)
            };
            let sign_a = _mm_set_ps(1.0, -1.0, 1.0, -1.0);
            let sign_b = _mm_set_ps(-1.0, 1.0, -1.0, 1.0);

            let temp0 = _mm_shuffle_ps(self.y_axis, self.x_axis, 0b00_00_00_00);
            let vec0 = _mm_shuffle_ps(temp0, temp0, 0b10_10_10_00);

            let temp1 = _mm_shuffle_ps(self.y_axis, self.x_axis, 0b01_01_01_01);
            let vec1 = _mm_shuffle_ps(temp1, temp1, 0b10_10_10_00);

            let temp2 = _mm_shuffle_ps(self.y_axis, self.x_axis, 0b10_10_10_10);
            let vec2 = _mm_shuffle_ps(temp2, temp2, 0b10_10_10_00);

            let temp3 = _mm_shuffle_ps(self.y_axis, self.x_axis, 0b11_11_11_11);
            let vec3 = _mm_shuffle_ps(temp3, temp3, 0b10_10_10_00);

            let mul00 = _mm_mul_ps(vec1, fac0);
            let mul01 = _mm_mul_ps(vec2, fac1);
            let mul02 = _mm_mul_ps(vec3, fac2);
            let sub00 = _mm_sub_ps(mul00, mul01);
            let add00 = _mm_add_ps(sub00, mul02);
            let inv0 = _mm_mul_ps(sign_b, add00);

            let mul03 = _mm_mul_ps(vec0, fac0);
            let mul04 = _mm_mul_ps(vec2, fac3);
            let mul05 = _mm_mul_ps(vec3, fac4);
            let sub01 = _mm_sub_ps(mul03, mul04);
            let add01 = _mm_add_ps(sub01, mul05);
            let inv1 = _mm_mul_ps(sign_a, add01);

            let mul06 = _mm_mul_ps(vec0, fac1);
            let mul07 = _mm_mul_ps(vec1, fac3);
            let mul08 = _mm_mul_ps(vec3, fac5);
            let sub02 = _mm_sub_ps(mul06, mul07);
            let add02 = _mm_add_ps(sub02, mul08);
            let inv2 = _mm_mul_ps(sign_b, add02);

            let mul09 = _mm_mul_ps(vec0, fac2);
            let mul10 = _mm_mul_ps(vec1, fac4);
            let mul11 = _mm_mul_ps(vec2, fac5);
            let sub03 = _mm_sub_ps(mul09, mul10);
            let add03 = _mm_add_ps(sub03, mul11);
            let inv3 = _mm_mul_ps(sign_a, add03);

            let row0 = _mm_shuffle_ps(inv0, inv1, 0b00_00_00_00);
            let row1 = _mm_shuffle_ps(inv2, inv3, 0b00_00_00_00);
            let row2 = _mm_shuffle_ps(row0, row1, 0b10_00_10_00);

            let dot0 = self.x_axis.dot(row2);
            glam_assert!(dot0 != 0.0);

            let rcp0 = _mm_set1_ps(dot0.recip());

            Self {
                x_axis: _mm_mul_ps(inv0, rcp0),
                y_axis: _mm_mul_ps(inv1, rcp0),
                z_axis: _mm_mul_ps(inv2, rcp0),
                w_axis: _mm_mul_ps(inv3, rcp0),
            }
        }
    }

    #[inline]
    fn transform_point3(&self, other: XYZ<f32>) -> XYZ<f32> {
        // SIMDFloatMatrix4x4::transform_point3_simd(self, other.into()).into()
        self.transform_float4_as_point3(other.into()).into()
    }

    #[inline]
    fn transform_vector3(&self, other: XYZ<f32>) -> XYZ<f32> {
        // SIMDFloatMatrix4x4::transform_vector3_simd(self, other.into()).into()
        self.transform_float4_as_vector3(other.into()).into()
    }

    #[inline]
    fn transform_float4_as_point3(&self, other: __m128) -> __m128 {
        let mut res = self.x_axis.mul(other.splat_x());
        res = self.y_axis.mul_add(other.splat_y(), res);
        res = self.z_axis.mul_add(other.splat_z(), res);
        res = self.w_axis.add(res);
        res = res.mul(res.splat_w().recip());
        res
    }

    #[inline]
    fn transform_float4_as_vector3(&self, other: __m128) -> __m128 {
        let mut res = self.x_axis.mul(other.splat_x());
        res = self.y_axis.mul_add(other.splat_y(), res);
        res = self.z_axis.mul_add(other.splat_z(), res);
        res
    }
}

impl ProjectionMatrix<f32, __m128> for Vector4x4<__m128> {}
