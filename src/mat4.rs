#[allow(unused_imports)]
use num_traits::Float;

use crate::core::traits::{
    matrix::{FloatMatrix4x4, Matrix4x4, MatrixConst},
    projection::ProjectionMatrix,
};
use crate::{Quat, Vec3, Vec3A, Vec3ASwizzles, Vec4, Vec4Swizzles};
#[cfg(all(vec4_sse2, target_arch = "x86"))]
use core::arch::x86::*;
#[cfg(all(vec4_sse2, target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::{
    cmp::Ordering,
    ops::{Add, Deref, DerefMut, Mul, Sub},
};

#[cfg(feature = "std")]
use std::iter::{Product, Sum};

macro_rules! impl_mat4 {
    ($new:ident, $mat4:ident, $vec4:ident, $vec3:ident, $quat:ident, $t:ty, $inner:ident) => {
        /// Creates a `$mat4` from four column vectors.
        #[inline(always)]
        pub fn $new(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> $mat4 {
            $mat4::from_cols(x_axis, y_axis, z_axis, w_axis)
        }

        impl Default for $mat4 {
            #[inline(always)]
            fn default() -> Self {
                Self::identity()
            }
        }

        impl PartialEq for $mat4 {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_ref().eq(other.as_ref())
            }
        }

        impl PartialOrd for $mat4 {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.as_ref().partial_cmp(other.as_ref())
            }
        }

        impl $mat4 {
            /// Creates a 4x4 matrix with all elements set to `0.0`.
            #[inline(always)]
            pub const fn zero() -> Self {
                Self($inner::ZERO)
            }

            /// Creates a 4x4 identity matrix.
            #[inline(always)]
            pub const fn identity() -> Self {
                Self($inner::IDENTITY)
            }

            /// Creates a 4x4 matrix from four column vectors.
            #[inline(always)]
            pub fn from_cols(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> Self {
                Self($inner::from_cols(x_axis.0, y_axis.0, z_axis.0, w_axis.0))
            }

            /// Creates a 4x4 matrix from a `[$t; 16]` stored in column major order.
            /// If your data is stored in row major you will need to `transpose` the
            /// returned matrix.
            #[inline(always)]
            pub fn from_cols_array(m: &[$t; 16]) -> Self {
                Self($inner::from_cols_array(m))
            }

            /// Creates a `[$t; 16]` storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline(always)]
            pub fn to_cols_array(&self) -> [$t; 16] {
                *self.as_ref()
            }

            /// Creates a 4x4 matrix from a `[[$t; 4]; 4]` stored in column major
            /// order.  If your data is in row major order you will need to `transpose`
            /// the returned matrix.
            #[inline(always)]
            pub fn from_cols_array_2d(m: &[[$t; 4]; 4]) -> Self {
                Self($inner::from_cols_array_2d(m))
            }

            /// Creates a `[[$t; 4]; 4]` storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline(always)]
            pub fn to_cols_array_2d(&self) -> [[$t; 4]; 4] {
                self.0.to_cols_array_2d()
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `scale`,
            /// `rotation` and `translation`.
            #[inline(always)]
            pub fn from_scale_rotation_translation(
                scale: $vec3,
                rotation: $quat,
                translation: $vec3,
            ) -> Self {
                Self($inner::from_scale_quaternion_translation(
                    scale.0,
                    rotation.0,
                    translation.0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `translation`.
            #[inline(always)]
            pub fn from_rotation_translation(rotation: $quat, translation: $vec3) -> Self {
                Self($inner::from_quaternion_translation(
                    rotation.0,
                    translation.0,
                ))
            }

            /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is expected to
            /// be a 4x4 homogeneous transformation matrix otherwise the output will be invalid.
            #[inline(always)]
            pub fn to_scale_rotation_translation(&self) -> ($vec3, $quat, $vec3) {
                let (scale, rotation, translation) = self.0.to_scale_quaternion_translation();
                ($vec3(scale), $quat(rotation), $vec3(translation))
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `rotation`.
            #[inline(always)]
            pub fn from_quat(rotation: $quat) -> Self {
                Self($inner::from_quaternion(rotation.0))
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `translation`.
            #[inline(always)]
            pub fn from_translation(translation: $vec3) -> Self {
                Self($inner::from_translation(translation.0))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around a normalized rotation `axis` of `angle` (in radians).
            #[inline(always)]
            pub fn from_axis_angle(axis: $vec3, angle: $t) -> Self {
                Self($inner::from_axis_angle(axis.0, angle))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the given Euler angles (in radians).
            #[inline]
            pub fn from_rotation_ypr(yaw: $t, pitch: $t, roll: $t) -> Self {
                let quat = $quat::from_rotation_ypr(yaw, pitch, roll);
                Self::from_quat(quat)
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the x axis of `angle` (in radians).
            #[inline(always)]
            pub fn from_rotation_x(angle: $t) -> Self {
                Self($inner::from_rotation_x(angle))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the y axis of `angle` (in radians).
            #[inline(always)]
            pub fn from_rotation_y(angle: $t) -> Self {
                Self($inner::from_rotation_y(angle))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the z axis of `angle` (in radians).
            #[inline(always)]
            pub fn from_rotation_z(angle: $t) -> Self {
                Self($inner::from_rotation_z(angle))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing the given
            /// non-uniform `scale`.
            #[inline(always)]
            pub fn from_scale(scale: $vec3) -> Self {
                Self($inner::from_scale(scale.0))
            }

            // #[inline]
            // pub(crate) fn col(&self, index: usize) -> $vec4 {
            //     match index {
            //         0 => self.x_axis,
            //         1 => self.y_axis,
            //         2 => self.z_axis,
            //         3 => self.w_axis,
            //         _ => panic!(
            //             "index out of bounds: the len is 4 but the index is {}",
            //             index
            //         ),
            //     }
            // }

            // #[inline]
            // pub(crate) fn col_mut(&mut self, index: usize) -> &mut $vec4 {
            //     match index {
            //         0 => &mut self.x_axis,
            //         1 => &mut self.y_axis,
            //         2 => &mut self.z_axis,
            //         3 => &mut self.w_axis,
            //         _ => panic!(
            //             "index out of bounds: the len is 4 but the index is {}",
            //             index
            //         ),
            //     }
            // }

            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
            #[inline]
            pub fn is_finite(&self) -> bool {
                self.x_axis.is_finite()
                    && self.y_axis.is_finite()
                    && self.z_axis.is_finite()
                    && self.w_axis.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            pub fn is_nan(&self) -> bool {
                self.x_axis.is_nan()
                    || self.y_axis.is_nan()
                    || self.z_axis.is_nan()
                    || self.w_axis.is_nan()
            }

            /// Returns the transpose of `self`.
            #[inline(always)]
            pub fn transpose(&self) -> Self {
                Self(self.0.transpose())
                // #[cfg(vec4_f32)]
                // {
                //     let (m00, m01, m02, m03) = self.x_axis.into();
                //     let (m10, m11, m12, m13) = self.y_axis.into();
                //     let (m20, m21, m22, m23) = self.z_axis.into();
                //     let (m30, m31, m32, m33) = self.w_axis.into();

                //     Self {
                //         x_axis: $vec4::new(m00, m10, m20, m30),
                //         y_axis: $vec4::new(m01, m11, m21, m31),
                //         z_axis: $vec4::new(m02, m12, m22, m32),
                //         w_axis: $vec4::new(m03, m13, m23, m33),
                //     }
                // }
            }

            /// Returns the determinant of `self`.
            #[inline(always)]
            pub fn determinant(&self) -> $t {
                self.0.determinant()
                // #[cfg(vec4_f32)]
                // {
                //     let (m00, m01, m02, m03) = self.x_axis.into();
                //     let (m10, m11, m12, m13) = self.y_axis.into();
                //     let (m20, m21, m22, m23) = self.z_axis.into();
                //     let (m30, m31, m32, m33) = self.w_axis.into();

                //     let a2323 = m22 * m33 - m23 * m32;
                //     let a1323 = m21 * m33 - m23 * m31;
                //     let a1223 = m21 * m32 - m22 * m31;
                //     let a0323 = m20 * m33 - m23 * m30;
                //     let a0223 = m20 * m32 - m22 * m30;
                //     let a0123 = m20 * m31 - m21 * m30;

                //     m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
                //         - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
                //         + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
                //         - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
                // }
            }

            /// Returns the inverse of `self`.
            ///
            /// If the matrix is not invertible the returned matrix will be invalid.
            #[inline(always)]
            pub fn inverse(&self) -> Self {
                Self(self.0.inverse())
                // #[cfg(vec4_f32)]
                // {
                //     let (m00, m01, m02, m03) = self.x_axis.into();
                //     let (m10, m11, m12, m13) = self.y_axis.into();
                //     let (m20, m21, m22, m23) = self.z_axis.into();
                //     let (m30, m31, m32, m33) = self.w_axis.into();

                //     let coef00 = m22 * m33 - m32 * m23;
                //     let coef02 = m12 * m33 - m32 * m13;
                //     let coef03 = m12 * m23 - m22 * m13;

                //     let coef04 = m21 * m33 - m31 * m23;
                //     let coef06 = m11 * m33 - m31 * m13;
                //     let coef07 = m11 * m23 - m21 * m13;

                //     let coef08 = m21 * m32 - m31 * m22;
                //     let coef10 = m11 * m32 - m31 * m12;
                //     let coef11 = m11 * m22 - m21 * m12;

                //     let coef12 = m20 * m33 - m30 * m23;
                //     let coef14 = m10 * m33 - m30 * m13;
                //     let coef15 = m10 * m23 - m20 * m13;

                //     let coef16 = m20 * m32 - m30 * m22;
                //     let coef18 = m10 * m32 - m30 * m12;
                //     let coef19 = m10 * m22 - m20 * m12;

                //     let coef20 = m20 * m31 - m30 * m21;
                //     let coef22 = m10 * m31 - m30 * m11;
                //     let coef23 = m10 * m21 - m20 * m11;

                //     let fac0 = $vec4::new(coef00, coef00, coef02, coef03);
                //     let fac1 = $vec4::new(coef04, coef04, coef06, coef07);
                //     let fac2 = $vec4::new(coef08, coef08, coef10, coef11);
                //     let fac3 = $vec4::new(coef12, coef12, coef14, coef15);
                //     let fac4 = $vec4::new(coef16, coef16, coef18, coef19);
                //     let fac5 = $vec4::new(coef20, coef20, coef22, coef23);

                //     let vec0 = $vec4::new(m10, m00, m00, m00);
                //     let vec1 = $vec4::new(m11, m01, m01, m01);
                //     let vec2 = $vec4::new(m12, m02, m02, m02);
                //     let vec3 = $vec4::new(m13, m03, m03, m03);

                //     let inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
                //     let inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
                //     let inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
                //     let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

                //     let sign_a = $vec4::new(1.0, -1.0, 1.0, -1.0);
                //     let sign_b = $vec4::new(-1.0, 1.0, -1.0, 1.0);

                //     let inverse = Self {
                //         x_axis: inv0 * sign_a,
                //         y_axis: inv1 * sign_b,
                //         z_axis: inv2 * sign_a,
                //         w_axis: inv3 * sign_b,
                //     };

                //     let col0 = $vec4::new(
                //         inverse.x_axis.x,
                //         inverse.y_axis.x,
                //         inverse.z_axis.x,
                //         inverse.w_axis.x,
                //     );

                //     let dot0 = self.x_axis * col0;
                //     let dot1 = dot0.x + dot0.y + dot0.z + dot0.w;

                //     glam_assert!(dot1 != 0.0);

                //     let rcp_det = 1.0 / dot1;
                //     inverse * rcp_det
                // }
            }

            /// Creates a left-handed view matrix using a camera position, an up direction, and a focal
            /// point.
            #[inline(always)]
            pub fn look_at_lh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
                Self($inner::look_at_lh(eye.0, center.0, up.0))
            }

            /// Creates a right-handed view matrix using a camera position, an up direction, and a focal
            /// point.
            #[inline(always)]
            pub fn look_at_rh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
                Self($inner::look_at_rh(eye.0, center.0, up.0))
            }

            /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
            /// This is the same as the OpenGL `gluPerspective` function.
            /// See https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
            #[inline(always)]
            pub fn perspective_rh_gl(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                Self($inner::perspective_rh_gl(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                    z_far,
                ))
            }

            /// Creates a left-handed perspective projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn perspective_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                Self($inner::perspective_lh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                    z_far,
                ))
            }

            /// Creates a right-handed perspective projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn perspective_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                Self($inner::perspective_rh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                    z_far,
                ))
            }

            /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn perspective_infinite_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                Self($inner::perspective_infinite_lh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                ))
            }

            /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn perspective_infinite_reverse_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                Self($inner::perspective_infinite_reverse_lh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                ))
            }

            /// Creates an infinite right-handed perspective projection matrix with
            /// [0,1] depth range.
            #[inline(always)]
            pub fn perspective_infinite_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                Self($inner::perspective_infinite_rh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                ))
            }

            /// Creates an infinite reverse right-handed perspective projection matrix
            /// with [0,1] depth range.
            #[inline(always)]
            pub fn perspective_infinite_reverse_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                Self($inner::perspective_infinite_reverse_rh(
                    fov_y_radians,
                    aspect_ratio,
                    z_near,
                ))
            }

            /// Creates a right-handed orthographic projection matrix with [-1,1] depth
            /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
            /// See
            /// https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
            #[inline(always)]
            pub fn orthographic_rh_gl(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                Self($inner::orthographic_rh_gl(
                    left, right, bottom, top, near, far,
                ))
            }

            /// Creates a left-handed orthographic projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn orthographic_lh(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                Self($inner::orthographic_lh(left, right, bottom, top, near, far))
            }

            /// Creates a right-handed orthographic projection matrix with [0,1] depth range.
            #[inline(always)]
            pub fn orthographic_rh(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                Self($inner::orthographic_rh(left, right, bottom, top, near, far))
            }

            /// Transforms a 4D vector.
            #[inline(always)]
            pub fn mul_vec4(&self, other: $vec4) -> $vec4 {
                $vec4(self.0.mul_vector(&other.0))
            }

            /// Multiplies two 4x4 matrices.
            #[inline(always)]
            pub fn mul_mat4(&self, other: &Self) -> Self {
                Self(self.0.mul_matrix(&other.0))
            }

            /// Adds two 4x4 matrices.
            #[inline(always)]
            pub fn add_mat4(&self, other: &Self) -> Self {
                Self(self.0.add_matrix(&other.0))
            }

            /// Subtracts two 4x4 matrices.
            #[inline(always)]
            pub fn sub_mat4(&self, other: &Self) -> Self {
                Self(self.0.sub_matrix(&other.0))
            }

            /// Multiplies this matrix by a scalar value.
            #[inline(always)]
            pub fn mul_scalar(&self, other: $t) -> Self {
                Self(self.0.mul_scalar(other))
            }

            /// Transforms the given `$vec3` as 3D point.
            ///
            /// This is the equivalent of multiplying the `$vec3` as a `$vec4` where `w` is `1.0`.
            #[inline]
            pub fn transform_point3(&self, other: $vec3) -> $vec3 {
                $vec3::from(self.transform_point3a(Vec3A::from(other)))
            }

            /// Transforms the given `Vec3A` as 3D point.
            ///
            /// This is the equivalent of multiplying the `Vec3A` as a `$vec4` where `w` is `1.0`.
            #[inline]
            pub fn transform_point3a(&self, other: Vec3A) -> Vec3A {
                let mut res = self.x_axis.mul(other.xxxx());
                res = self.y_axis.mul_add(other.yyyy(), res);
                res = self.z_axis.mul_add(other.zzzz(), res);
                res = self.w_axis.add(res);
                res = res.mul(res.wwww().recip());
                Vec3A::from(res)
            }

            /// Transforms the give `$vec3` as 3D vector.
            ///
            /// This is the equivalent of multiplying the `$vec3` as a `$vec4` where `w` is `0.0`.
            #[inline]
            pub fn transform_vector3(&self, other: $vec3) -> $vec3 {
                $vec3::from(self.transform_vector3a(Vec3A::from(other)))
            }

            /// Transforms the give `Vec3A` as 3D vector.
            ///
            /// This is the equivalent of multiplying the `Vec3A` as a `$vec4` where `w` is `0.0`.
            #[inline]
            pub fn transform_vector3a(&self, other: Vec3A) -> Vec3A {
                let mut res = self.x_axis.mul(other.xxxx());
                res = self.y_axis.mul_add(other.yyyy(), res);
                res = self.z_axis.mul_add(other.zzzz(), res);
                Vec3A::from(res)
            }

            /// Returns true if the absolute difference of all elements between `self` and `other` is less
            /// than or equal to `max_abs_diff`.
            ///
            /// This can be used to compare if two `$mat4`'s contain similar elements. It works best when
            /// comparing with a known value. The `max_abs_diff` that should be used used depends on the
            /// values being compared against.
            ///
            /// For more on floating point comparisons see
            /// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            #[inline]
            pub fn abs_diff_eq(&self, other: Self, max_abs_diff: $t) -> bool {
                self.x_axis.abs_diff_eq(other.x_axis, max_abs_diff)
                    && self.y_axis.abs_diff_eq(other.y_axis, max_abs_diff)
                    && self.z_axis.abs_diff_eq(other.z_axis, max_abs_diff)
                    && self.w_axis.abs_diff_eq(other.w_axis, max_abs_diff)
            }
        }

        impl AsRef<[$t; 16]> for $mat4 {
            #[inline]
            fn as_ref(&self) -> &[$t; 16] {
                unsafe { &*(self as *const Self as *const [$t; 16]) }
            }
        }

        impl AsMut<[$t; 16]> for $mat4 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$t; 16] {
                unsafe { &mut *(self as *mut Self as *mut [$t; 16]) }
            }
        }

        impl Add<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self.add_mat4(&other)
            }
        }

        impl Sub<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                self.sub_mat4(&other)
            }
        }

        impl Mul<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                self.mul_mat4(&other)
            }
        }

        impl Mul<$vec4> for $mat4 {
            type Output = $vec4;
            #[inline(always)]
            fn mul(self, other: $vec4) -> $vec4 {
                self.mul_vec4(other)
            }
        }

        impl Mul<$mat4> for $t {
            type Output = $mat4;
            #[inline(always)]
            fn mul(self, other: $mat4) -> $mat4 {
                other.mul_scalar(self)
            }
        }

        impl Mul<$t> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn mul(self, other: $t) -> Self {
                self.mul_scalar(other)
            }
        }

        impl Deref for $mat4 {
            type Target = crate::core::storage::Vector4x4<$vec4>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self as *const Self::Target) }
            }
        }

        impl DerefMut for $mat4 {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self as *mut Self::Target) }
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl fmt::Debug for $mat4 {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct(stringify!($mat4))
                    .field("x_axis", &self.x_axis)
                    .field("y_axis", &self.y_axis)
                    .field("z_axis", &self.z_axis)
                    .field("w_axis", &self.w_axis)
                    .finish()
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl fmt::Display for $mat4 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    f,
                    "[{}, {}, {}, {}]",
                    self.x_axis, self.y_axis, self.z_axis, self.w_axis
                )
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Sum<&'a Self> for $mat4 {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::zero(), |a, &b| Self::add(a, b))
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Product<&'a Self> for $mat4 {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::identity(), |a, &b| Self::mul(a, b))
            }
        }
    };
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
type InnerF32 = crate::core::storage::Vector4x4<__m128>;

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
type InnerF32 = crate::core::storage::Vector4x4<XYZW<f32>>;

/// A 4x4 column major matrix.
///
/// This type is 16 byte aligned.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat4(pub(crate) InnerF32);

impl_mat4!(mat4, Mat4, Vec4, Vec3, Quat, f32, InnerF32);
