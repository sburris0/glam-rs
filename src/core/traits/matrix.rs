use crate::core::{
    storage::{XYZWx4, XYZx3, XYx2, XY, XYZ, XYZW},
    traits::{
        scalar::{FloatEx, NumEx},
        vector::*,
    },
};

pub trait MatrixConst {
    const ZERO: Self;
    const IDENTITY: Self;
}

pub trait Matrix<T: NumEx>: Sized + Copy + Clone {}

pub trait Matrix2x2<T: NumEx>: Matrix<T> {
    fn new(m00: T, m01: T, m10: T, m11: T) -> Self;

    fn deref(&self) -> &XYx2<T>;
    fn deref_mut(&mut self) -> &mut XYx2<T>;

    #[inline(always)]
    fn from_cols(x_axis: XY<T>, y_axis: XY<T>) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline(always)]
    fn from_cols_array(m: &[T; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array(&self) -> [T; 4] {
        let m = self.deref();
        [m.x_axis.x, m.x_axis.y,
         m.y_axis.x, m.y_axis.y]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols_array_2d(m: &[[T; 2]; 2]) -> Self {
        Self::new(
            m[0][0], m[0][1],
            m[1][0], m[1][1])
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array_2d(&self) -> [[T; 2]; 2] {
        let m = self.deref();
        [[m.x_axis.x, m.x_axis.y],
         [m.y_axis.x, m.y_axis.y]]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_scale(scale: XY<T>) -> Self {
        Self::new(
            scale.x, T::ZERO,
            T::ZERO, scale.y)
    }

    fn determinant(&self) -> T;
    fn transpose(&self) -> Self;
    fn mul_vector(&self, other: XY<T>) -> XY<T>;
    fn mul_matrix(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: T) -> Self;
    fn add_matrix(&self, other: &Self) -> Self;
    fn sub_matrix(&self, other: &Self) -> Self;
}

pub trait FloatMatrix2x2<T: FloatEx>: Matrix2x2<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool;

    #[rustfmt::skip]
    #[inline]
    fn from_scale_angle(scale: XY<T>, angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(
            cos * scale.x, sin * scale.x,
            -sin * scale.y, cos * scale.y)
    }

    #[rustfmt::skip]
    #[inline]
    fn from_angle(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(
            cos, sin,
            -sin, cos)
    }

    fn inverse(&self) -> Self;
    // fn is_finite(&self) -> bool;
}

pub trait Matrix3x3<T: NumEx>: Matrix<T> {
    fn new(m00: T, m01: T, m02: T, m10: T, m11: T, m12: T, m20: T, m21: T, m22: T) -> Self;

    fn deref(&self) -> &XYZx3<T>;
    fn deref_mut(&mut self) -> &mut XYZx3<T>;

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols(x_axis: XYZ<T>, y_axis: XYZ<T>, z_axis: XYZ<T>) -> Self {
        Self::new(
            x_axis.x, x_axis.y, x_axis.z,
            y_axis.x, y_axis.y, y_axis.z,
            z_axis.x, z_axis.y, z_axis.z,
        )
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols_array(m: &[T; 9]) -> Self {
        Self::new(
            m[0], m[1], m[2],
            m[3], m[4], m[5],
            m[6], m[7], m[8])
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array(&self) -> [T; 9] {
        let m = self.deref();
        [
            m.x_axis.x, m.x_axis.y, m.x_axis.z,
            m.y_axis.x, m.y_axis.y, m.y_axis.z,
            m.z_axis.x, m.z_axis.y, m.z_axis.z,
        ]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols_array_2d(m: &[[T; 3]; 3]) -> Self {
        Self::new(
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
        )
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array_2d(&self) -> [[T; 3]; 3] {
        let m = self.deref();
        [
            [m.x_axis.x, m.x_axis.y, m.x_axis.z],
            [m.y_axis.x, m.y_axis.y, m.y_axis.z],
            [m.z_axis.x, m.z_axis.y, m.z_axis.z],
        ]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_scale(scale: XYZ<T>) -> Self {
        Self::new(
            scale.x, T::ZERO, T::ZERO,
            T::ZERO, scale.y, T::ZERO,
            T::ZERO, T::ZERO, scale.z,
        )
    }

    fn determinant(&self) -> T;
    fn transpose(&self) -> Self;
    fn mul_vector(&self, other: XYZ<T>) -> XYZ<T>;
    fn mul_matrix(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: T) -> Self;
    fn add_matrix(&self, other: &Self) -> Self;
    fn sub_matrix(&self, other: &Self) -> Self;
}

pub trait FloatMatrix3x3<T: FloatEx>: Matrix3x3<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool;

    #[rustfmt::skip]
    #[inline]
    fn from_scale_angle_translation(scale: XY<T>, angle: T, translation: XY<T>) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(
            cos * scale.x, sin * scale.x, T::ZERO,
            -sin * scale.y, cos * scale.y, T::ZERO,
            translation.x, translation.y, T::ONE)
    }

    #[rustfmt::skip]
    #[inline]
    fn from_axis_angle(axis: XYZ<T>, angle: T) -> Self {
        glam_assert!(axis.is_normalized());
        let (sin, cos) = angle.sin_cos();
        let (xsin, ysin, zsin) = axis.mul_scalar(sin).into_tuple();
        let (x2, y2, z2) = axis.mul(axis).into_tuple();
        let omc = T::ONE - cos;
        let xyomc = axis.x * axis.y * omc;
        let xzomc = axis.x * axis.z * omc;
        let yzomc = axis.y * axis.z * omc;
        Self::new(
            x2 * omc + cos, xyomc + zsin, xzomc - ysin,
            xyomc - zsin, y2 * omc + cos, yzomc + xsin,
            xzomc + ysin, yzomc - xsin, z2 * omc + cos,
        )
    }

    #[rustfmt::skip]
    #[inline]
    fn from_quaternion(rotation: XYZW<T>) -> Self {
        glam_assert!(rotation.is_normalized());
        let x2 = rotation.x + rotation.x;
        let y2 = rotation.y + rotation.y;
        let z2 = rotation.z + rotation.z;
        let xx = rotation.x * x2;
        let xy = rotation.x * y2;
        let xz = rotation.x * z2;
        let yy = rotation.y * y2;
        let yz = rotation.y * z2;
        let zz = rotation.z * z2;
        let wx = rotation.w * x2;
        let wy = rotation.w * y2;
        let wz = rotation.w * z2;

        Self::new(
            T::ONE - (yy + zz), xy + wz, xz - wy,
            xy - wz, T::ONE - (xx + zz), yz + wx,
            xz + wy, yz - wx, T::ONE - (xx + yy))
    }

    #[rustfmt::skip]
    #[inline]
    fn from_rotation_x(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::new(
            T::ONE, T::ZERO, T::ZERO,
            T::ZERO, cosa, sina,
            T::ZERO, -sina, cosa)
    }

    #[rustfmt::skip]
    #[inline]
    fn from_rotation_y(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::new(
            cosa, T::ZERO, -sina,
            T::ZERO, T::ONE, T::ZERO,
            sina, T::ZERO, cosa)
    }

    #[rustfmt::skip]
    #[inline]
    fn from_rotation_z(angle: T) -> Self {
        let (sina, cosa) = angle.sin_cos();
        Self::new(
            cosa, sina, T::ZERO,
            -sina, cosa, T::ZERO,
            T::ZERO, T::ZERO, T::ONE)
    }

    #[rustfmt::skip]
    #[inline]
    fn from_scale(scale: XYZ<T>) -> Self {
        // TODO: should have a affine 2D scale and a 3d scale?
        // Do not panic as long as any component is non-zero
        glam_assert!(scale.cmpne(XYZ::ZERO).any());
        Self::new(
            scale.x, T::ZERO, T::ZERO,
            T::ZERO, scale.y, T::ZERO,
            T::ZERO, T::ZERO, scale.z,
        )
    }

    fn inverse(&self) -> Self;
}

pub trait Matrix4x4<T: NumEx>: Matrix<T> {
    #[rustfmt::skip]
    fn new(
        m00: T, m01: T, m02: T, m03: T,
        m10: T, m11: T, m12: T, m13: T,
        m20: T, m21: T, m22: T, m23: T,
        m30: T, m31: T, m32: T, m33: T,
        ) -> Self;

    fn deref(&self) -> &XYZWx4<T>;
    fn deref_mut(&mut self) -> &mut XYZWx4<T>;

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols(x_axis: XYZW<T>, y_axis: XYZW<T>, z_axis: XYZW<T>, w_axis: XYZW<T>) -> Self {
        Self::new(
            x_axis.x, x_axis.y, x_axis.z, x_axis.w,
            y_axis.x, y_axis.y, y_axis.z, y_axis.w,
            z_axis.x, z_axis.y, z_axis.z, z_axis.w,
            w_axis.x, w_axis.y, w_axis.z, w_axis.w,
        )
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_cols_array(m: &[T; 16]) -> Self {
        Self::new(
             m[0],  m[1],  m[2],  m[3],
             m[4],  m[5],  m[6],  m[7],
             m[8],  m[9], m[10], m[11],
            m[12], m[13], m[14], m[15])
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array(&self) -> [T; 16] {
        let m = self.deref();
        [
            m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w,
            m.y_axis.x, m.y_axis.y, m.y_axis.z, m.y_axis.w,
            m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w,
            m.w_axis.x, m.w_axis.y, m.w_axis.z, m.w_axis.w,
        ]
    }

    #[inline(always)]
    fn from_cols_array_2d(m: &[[T; 4]; 4]) -> Self {
        Self::new(
            m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0],
            m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3],
        )
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn to_cols_array_2d(&self) -> [[T; 4]; 4] {
        let m = self.deref();
        [
            [m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w],
            [m.y_axis.x, m.y_axis.y, m.y_axis.z, m.y_axis.w],
            [m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w],
            [m.w_axis.x, m.w_axis.y, m.w_axis.z, m.w_axis.w],
        ]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn from_scale(scale: XYZW<T>) -> Self {
        Self::new(
            scale.x, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, scale.y, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, scale.z, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, scale.w,
        )
    }

    fn determinant(&self) -> T;
    fn transpose(&self) -> Self;
    fn mul_vector(&self, other: XYZW<T>) -> XYZW<T>;
    fn mul_matrix(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: T) -> Self;
    fn add_matrix(&self, other: &Self) -> Self;
    fn sub_matrix(&self, other: &Self) -> Self;
}

pub trait FloatMatrix4x4<T: FloatEx>: Matrix4x4<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool;

    fn inverse(&self) -> Self;
}
