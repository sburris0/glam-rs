#[allow(unused_imports)]
use num_traits::Float;

use crate::core::traits::matrix::{FloatMatrix4x4, Matrix4x4, MatrixConst};
use crate::{Mat3, Quat, Vec3, Vec3A, Vec3ASwizzles, Vec4, Vec4Swizzles};
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
        #[inline]
        pub fn $new(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> $mat4 {
            $mat4::from_cols(x_axis, y_axis, z_axis, w_axis)
        }

        #[inline]
        fn quat_to_axes(rotation: $quat) -> ($vec4, $vec4, $vec4) {
            glam_assert!(rotation.is_normalized());
            let (x, y, z, w) = rotation.into();
            let x2 = x + x;
            let y2 = y + y;
            let z2 = z + z;
            let xx = x * x2;
            let xy = x * y2;
            let xz = x * z2;
            let yy = y * y2;
            let yz = y * z2;
            let zz = z * z2;
            let wx = w * x2;
            let wy = w * y2;
            let wz = w * z2;

            let x_axis = $vec4::new(1.0 - (yy + zz), xy + wz, xz - wy, 0.0);
            let y_axis = $vec4::new(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0);
            let z_axis = $vec4::new(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0);
            (x_axis, y_axis, z_axis)
        }

        impl Default for $mat4 {
            #[inline]
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
            #[inline]
            pub const fn zero() -> Self {
                Self($inner::ZERO)
            }

            /// Creates a 4x4 identity matrix.
            #[inline]
            pub const fn identity() -> Self {
                Self($inner::IDENTITY)
            }

            /// Creates a 4x4 matrix from four column vectors.
            #[inline]
            pub fn from_cols(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> Self {
                Self($inner::from_cols(x_axis.0, y_axis.0, z_axis.0, w_axis.0))
            }

            /// Creates a 4x4 matrix from a `[$t; 16]` stored in column major order.
            /// If your data is stored in row major you will need to `transpose` the
            /// returned matrix.
            #[inline]
            pub fn from_cols_array(m: &[$t; 16]) -> Self {
                Self($inner::from_cols_array(m))
            }

            /// Creates a `[$t; 16]` storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            pub fn to_cols_array(&self) -> [$t; 16] {
                *self.as_ref()
            }

            /// Creates a 4x4 matrix from a `[[$t; 4]; 4]` stored in column major
            /// order.  If your data is in row major order you will need to `transpose`
            /// the returned matrix.
            #[inline]
            pub fn from_cols_array_2d(m: &[[$t; 4]; 4]) -> Self {
                Self($inner::from_cols_array_2d(m))
            }

            /// Creates a `[[$t; 4]; 4]` storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            pub fn to_cols_array_2d(&self) -> [[$t; 4]; 4] {
                self.0.to_cols_array_2d()
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `scale`,
            /// `rotation` and `translation`.
            #[inline]
            pub fn from_scale_rotation_translation(
                scale: $vec3,
                rotation: $quat,
                translation: $vec3,
            ) -> Self {
                glam_assert!(rotation.is_normalized());
                let (x_axis, y_axis, z_axis) = quat_to_axes(rotation);
                let (scale_x, scale_y, scale_z) = scale.into();
                Self($inner::from_cols(
                    x_axis.mul(scale_x).0,
                    y_axis.mul(scale_y).0,
                    z_axis.mul(scale_z).0,
                    translation.extend(1.0).0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `translation`.
            #[inline]
            pub fn from_rotation_translation(rotation: $quat, translation: $vec3) -> Self {
                glam_assert!(rotation.is_normalized());
                let (x_axis, y_axis, z_axis) = quat_to_axes(rotation);
                Self($inner::from_cols(
                    x_axis.0,
                    y_axis.0,
                    z_axis.0,
                    translation.extend(1.0).0,
                ))
            }

            /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is expected to
            /// be a 4x4 homogeneous transformation matrix otherwise the output will be invalid.
            pub fn to_scale_rotation_translation(&self) -> ($vec3, $quat, $vec3) {
                let det = self.determinant();
                glam_assert!(det != 0.0);

                let scale = Vec3A::new(
                    self.x_axis.length() * det.signum(),
                    self.y_axis.length(),
                    self.z_axis.length(),
                );
                glam_assert!(scale.cmpne(Vec3A::zero()).all());

                let inv_scale = scale.recip();

                let rotation = $quat::from_rotation_mat3(&Mat3::from_cols(
                    $vec3::from(Vec3A::from(self.x_axis) * inv_scale.xxx()),
                    $vec3::from(Vec3A::from(self.y_axis) * inv_scale.yyy()),
                    $vec3::from(Vec3A::from(self.z_axis) * inv_scale.zzz()),
                ));

                let translation = self.w_axis.xyz();
                let scale = $vec3::from(scale);

                (scale, rotation, translation)
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `rotation`.
            #[inline]
            pub fn from_quat(rotation: $quat) -> Self {
                glam_assert!(rotation.is_normalized());
                let (x_axis, y_axis, z_axis) = quat_to_axes(rotation);
                Self($inner::from_cols(
                    x_axis.0,
                    y_axis.0,
                    z_axis.0,
                    $vec4::unit_w().0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix from the given `translation`.
            #[inline]
            pub fn from_translation(translation: $vec3) -> Self {
                Self($inner::from_cols(
                    $vec4::unit_x().0,
                    $vec4::unit_y().0,
                    $vec4::unit_z().0,
                    translation.extend(1.0).0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around a normalized rotation `axis` of `angle` (in radians).
            #[inline]
            pub fn from_axis_angle(axis: $vec3, angle: $t) -> Self {
                glam_assert!(axis.is_normalized());
                let (sin, cos) = angle.sin_cos();
                let (x, y, z) = axis.into();
                let (xsin, ysin, zsin) = (axis * sin).into();
                let (x2, y2, z2) = (axis * axis).into();
                let omc = 1.0 - cos;
                let xyomc = x * y * omc;
                let xzomc = x * z * omc;
                let yzomc = y * z * omc;
                Self($inner::from_cols(
                    $vec4::new(x2 * omc + cos, xyomc + zsin, xzomc - ysin, 0.0).0,
                    $vec4::new(xyomc - zsin, y2 * omc + cos, yzomc + xsin, 0.0).0,
                    $vec4::new(xzomc + ysin, yzomc - xsin, z2 * omc + cos, 0.0).0,
                    $vec4::unit_w().0,
                ))
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
            #[inline]
            pub fn from_rotation_x(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self($inner::from_cols(
                    $vec4::unit_x().0,
                    $vec4::new(0.0, cosa, sina, 0.0).0,
                    $vec4::new(0.0, -sina, cosa, 0.0).0,
                    $vec4::unit_w().0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the y axis of `angle` (in radians).
            #[inline]
            pub fn from_rotation_y(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self($inner::from_cols(
                    $vec4::new(cosa, 0.0, -sina, 0.0).0,
                    $vec4::unit_y().0,
                    $vec4::new(sina, 0.0, cosa, 0.0).0,
                    $vec4::unit_w().0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing a rotation
            /// around the z axis of `angle` (in radians).
            #[inline]
            pub fn from_rotation_z(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self($inner::from_cols(
                    $vec4::new(cosa, sina, 0.0, 0.0).0,
                    $vec4::new(-sina, cosa, 0.0, 0.0).0,
                    $vec4::unit_z().0,
                    $vec4::unit_w().0,
                ))
            }

            /// Creates a 4x4 homogeneous transformation matrix containing the given
            /// non-uniform `scale`.
            #[inline]
            pub fn from_scale(scale: $vec3) -> Self {
                // Do not panic as long as any component is non-zero
                glam_assert!(scale.cmpne($vec3::zero()).any());
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

            /// Creates a left-handed view matrix using a camera position, an up direction, and a camera
            /// direction.
            #[inline]
            // TODO: make public at some point
            fn look_to_lh(eye: Vec3A, dir: Vec3A, up: Vec3A) -> Self {
                let f = dir.normalize();
                let s = up.cross(f).normalize();
                let u = f.cross(s);
                let (fx, fy, fz) = f.into();
                let (sx, sy, sz) = s.into();
                let (ux, uy, uz) = u.into();
                $mat4::from_cols(
                    $vec4::new(sx, ux, fx, 0.0),
                    $vec4::new(sy, uy, fy, 0.0),
                    $vec4::new(sz, uz, fz, 0.0),
                    $vec4::new(-s.dot(eye), -u.dot(eye), -f.dot(eye), 1.0),
                )
            }

            /// Creates a left-handed view matrix using a camera position, an up direction, and a focal
            /// point.
            #[inline]
            pub fn look_at_lh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
                let eye = Vec3A::from(eye);
                let center = Vec3A::from(center);
                let up = Vec3A::from(up);
                glam_assert!(up.is_normalized());
                $mat4::look_to_lh(eye, center - eye, up)
            }

            /// Creates a right-handed view matrix using a camera position, an up direction, and a focal
            /// point.
            #[inline]
            pub fn look_at_rh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
                let eye = Vec3A::from(eye);
                let center = Vec3A::from(center);
                let up = Vec3A::from(up);
                glam_assert!(up.is_normalized());
                $mat4::look_to_lh(eye, eye - center, up)
            }

            /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
            /// This is the same as the OpenGL `gluPerspective` function.
            /// See https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
            pub fn perspective_rh_gl(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                let inv_length = 1.0 / (z_near - z_far);
                let f = 1.0 / (0.5 * fov_y_radians).tan();
                let a = f / aspect_ratio;
                let b = (z_near + z_far) * inv_length;
                let c = (2.0 * z_near * z_far) * inv_length;
                $mat4::from_cols(
                    $vec4::new(a, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, f, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, b, -1.0),
                    $vec4::new(0.0, 0.0, c, 0.0),
                )
            }

            /// Creates a left-handed perspective projection matrix with [0,1] depth range.
            pub fn perspective_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                glam_assert!(z_near > 0.0 && z_far > 0.0);
                let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                let r = z_far / (z_far - z_near);
                $mat4::from_cols(
                    $vec4::new(w, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, h, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, r, 1.0),
                    $vec4::new(0.0, 0.0, -r * z_near, 0.0),
                )
            }

            /// Creates a right-handed perspective projection matrix with [0,1] depth range.
            pub fn perspective_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
                z_far: $t,
            ) -> Self {
                glam_assert!(z_near > 0.0 && z_far > 0.0);
                let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                let r = z_far / (z_near - z_far);
                $mat4::from_cols(
                    $vec4::new(w, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, h, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, r, -1.0),
                    $vec4::new(0.0, 0.0, r * z_near, 0.0),
                )
            }

            /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
            pub fn perspective_infinite_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                glam_assert!(z_near > 0.0);
                let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                $mat4::from_cols(
                    $vec4::new(w, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, h, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, 1.0, 1.0),
                    $vec4::new(0.0, 0.0, -z_near, 0.0),
                )
            }

            /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
            pub fn perspective_infinite_reverse_lh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                glam_assert!(z_near > 0.0);
                let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
                let h = cos_fov / sin_fov;
                let w = h / aspect_ratio;
                $mat4::from_cols(
                    $vec4::new(w, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, h, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, 0.0, 1.0),
                    $vec4::new(0.0, 0.0, z_near, 0.0),
                )
            }

            /// Creates an infinite right-handed perspective projection matrix with
            /// [0,1] depth range.
            pub fn perspective_infinite_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                let f = 1.0 / (0.5 * fov_y_radians).tan();
                $mat4::from_cols(
                    $vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, f, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, -1.0, -1.0),
                    $vec4::new(0.0, 0.0, -z_near, 0.0),
                )
            }

            /// Creates an infinite reverse right-handed perspective projection matrix
            /// with [0,1] depth range.
            pub fn perspective_infinite_reverse_rh(
                fov_y_radians: $t,
                aspect_ratio: $t,
                z_near: $t,
            ) -> Self {
                let f = 1.0 / (0.5 * fov_y_radians).tan();
                $mat4::from_cols(
                    $vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, f, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, 0.0, -1.0),
                    $vec4::new(0.0, 0.0, z_near, 0.0),
                )
            }

            /// Creates a right-handed orthographic projection matrix with [-1,1] depth
            /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
            /// See
            /// https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
            pub fn orthographic_rh_gl(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                let a = 2.0 / (right - left);
                let b = 2.0 / (top - bottom);
                let c = -2.0 / (far - near);
                let tx = -(right + left) / (right - left);
                let ty = -(top + bottom) / (top - bottom);
                let tz = -(far + near) / (far - near);

                $mat4::from_cols(
                    $vec4::new(a, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, b, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, c, 0.0),
                    $vec4::new(tx, ty, tz, 1.0),
                )
            }

            /// Creates a left-handed orthographic projection matrix with [0,1] depth range.
            pub fn orthographic_lh(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                let rcp_width = 1.0 / (right - left);
                let rcp_height = 1.0 / (top - bottom);
                let r = 1.0 / (far - near);
                $mat4::from_cols(
                    $vec4::new(rcp_width + rcp_width, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, rcp_height + rcp_height, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, r, 0.0),
                    $vec4::new(
                        -(left + right) * rcp_width,
                        -(top + bottom) * rcp_height,
                        -r * near,
                        1.0,
                    ),
                )
            }

            /// Creates a right-handed orthographic projection matrix with [0,1] depth range.
            pub fn orthographic_rh(
                left: $t,
                right: $t,
                bottom: $t,
                top: $t,
                near: $t,
                far: $t,
            ) -> Self {
                let rcp_width = 1.0 / (right - left);
                let rcp_height = 1.0 / (top - bottom);
                let r = 1.0 / (near - far);
                $mat4::from_cols(
                    $vec4::new(rcp_width + rcp_width, 0.0, 0.0, 0.0),
                    $vec4::new(0.0, rcp_height + rcp_height, 0.0, 0.0),
                    $vec4::new(0.0, 0.0, r, 0.0),
                    $vec4::new(
                        -(left + right) * rcp_width,
                        -(top + bottom) * rcp_height,
                        r * near,
                        1.0,
                    ),
                )
            }

            /// Transforms a 4D vector.
            #[inline]
            pub fn mul_vec4(&self, other: $vec4) -> $vec4 {
                $vec4(self.0.mul_vector(&other.0))
            }

            /// Multiplies two 4x4 matrices.
            #[inline]
            pub fn mul_mat4(&self, other: &Self) -> Self {
                Self(self.0.mul_matrix(&other.0))
            }

            /// Adds two 4x4 matrices.
            #[inline]
            pub fn add_mat4(&self, other: &Self) -> Self {
                Self(self.0.add_matrix(&other.0))
            }

            /// Subtracts two 4x4 matrices.
            #[inline]
            pub fn sub_mat4(&self, other: &Self) -> Self {
                Self(self.0.sub_matrix(&other.0))
            }

            /// Multiplies this matrix by a scalar value.
            #[inline]
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
            #[inline]
            fn add(self, other: Self) -> Self {
                self.add_mat4(&other)
            }
        }

        impl Sub<$mat4> for $mat4 {
            type Output = Self;
            #[inline]
            fn sub(self, other: Self) -> Self {
                self.sub_mat4(&other)
            }
        }

        impl Mul<$mat4> for $mat4 {
            type Output = Self;
            #[inline]
            fn mul(self, other: Self) -> Self {
                self.mul_mat4(&other)
            }
        }

        impl Mul<$vec4> for $mat4 {
            type Output = $vec4;
            #[inline]
            fn mul(self, other: $vec4) -> $vec4 {
                self.mul_vec4(other)
            }
        }

        impl Mul<$mat4> for $t {
            type Output = $mat4;
            #[inline]
            fn mul(self, other: $mat4) -> $mat4 {
                other.mul_scalar(self)
            }
        }

        impl Mul<$t> for $mat4 {
            type Output = Self;
            #[inline]
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
