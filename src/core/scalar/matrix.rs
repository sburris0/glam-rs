#[cfg(feature = "glam-assert")]
use crate::core::traits::vector::MaskVector3;
use crate::core::{
    storage::{XYZx3, XYx2, XY, XYZ},
    traits::{
        matrix::{FloatMatrix2x2, FloatMatrix3x3, Matrix, Matrix2x2, Matrix3x3, MatrixConst},
        scalar::{FloatEx, NumEx},
        vector::{
            FloatVector, FloatVector2, FloatVector3, Vector, Vector2, Vector2Const, Vector3,
            Vector3Const, VectorConst,
        },
    },
};

impl<T: NumEx> MatrixConst for XYx2<T> {
    const ZERO: Self = Self {
        x_axis: XY::ZERO,
        y_axis: XY::ZERO,
    };
    const IDENTITY: Self = Self {
        x_axis: XY::UNIT_X,
        y_axis: XY::UNIT_Y,
    };
}

impl<T: NumEx> Matrix<T> for XYx2<T> {}

impl<T: NumEx> Matrix2x2<T> for XYx2<T> {
    #[inline(always)]
    fn new(m00: T, m01: T, m10: T, m11: T) -> Self {
        Self {
            x_axis: Vector2::new(m00, m01),
            y_axis: Vector2::new(m10, m11),
        }
    }

    #[inline(always)]
    fn deref(&self) -> &XYx2<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYx2<T> {
        self
    }

    #[inline]
    fn determinant(&self) -> T {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        Self::new(self.x_axis.x, self.y_axis.x, self.x_axis.y, self.y_axis.y)
    }

    #[inline]
    fn mul_vector(&self, other: XY<T>) -> XY<T> {
        Vector2::new(
            (self.x_axis.x * other.x) + (self.y_axis.x * other.y),
            (self.x_axis.y * other.x) + (self.y_axis.y * other.y),
        )
    }

    #[inline]
    fn mul_matrix(&self, other: &Self) -> Self {
        Self::from_cols(self.mul_vector(other.x_axis), self.mul_vector(other.y_axis))
    }

    #[inline]
    fn mul_scalar(&self, other: T) -> Self {
        Self::from_cols(self.x_axis.mul_scalar(other), self.y_axis.mul_scalar(other))
    }

    #[inline]
    fn add_matrix(&self, other: &Self) -> Self {
        Self::from_cols(self.x_axis.add(other.x_axis), self.y_axis.add(other.y_axis))
    }

    #[inline]
    fn sub_matrix(&self, other: &Self) -> Self {
        Self::from_cols(self.x_axis.sub(other.x_axis), self.y_axis.sub(other.y_axis))
    }
}

impl<T: FloatEx> FloatMatrix2x2<T> for XYx2<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool {
        self.x_axis.abs_diff_eq(other.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(other.y_axis, max_abs_diff)
    }

    // #[inline]
    // fn is_finite(&self) -> bool {
    //     self.x_axis.is_finite() && self.y_axis.is_finite()
    // }

    #[inline]
    fn inverse(&self) -> Self {
        let inv_det = {
            let det = self.determinant();
            glam_assert!(det != T::ZERO);
            det.recip()
        };
        Self::new(
            self.y_axis.y * inv_det,
            self.x_axis.y * -inv_det,
            self.y_axis.x * -inv_det,
            self.x_axis.x * inv_det,
        )
    }
}

impl<T: NumEx> MatrixConst for XYZx3<T> {
    const ZERO: Self = Self {
        x_axis: XYZ::ZERO,
        y_axis: XYZ::ZERO,
        z_axis: XYZ::ZERO,
    };
    const IDENTITY: Self = Self {
        x_axis: XYZ::UNIT_X,
        y_axis: XYZ::UNIT_Y,
        z_axis: XYZ::UNIT_Z,
    };
}

impl<T: NumEx> Matrix<T> for XYZx3<T> {}

impl<T: NumEx> Matrix3x3<T> for XYZx3<T> {
    fn new(m00: T, m01: T, m02: T, m10: T, m11: T, m12: T, m20: T, m21: T, m22: T) -> Self {
        Self {
            x_axis: XYZ {
                x: m00,
                y: m01,
                z: m02,
            },
            y_axis: XYZ {
                x: m10,
                y: m11,
                z: m12,
            },
            z_axis: XYZ {
                x: m20,
                y: m21,
                z: m22,
            },
        }
    }

    #[inline(always)]
    fn deref(&self) -> &XYZx3<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZx3<T> {
        self
    }

    #[inline]
    fn determinant(&self) -> T {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        Self::new(
            self.x_axis.x,
            self.y_axis.x,
            self.z_axis.x,
            self.x_axis.y,
            self.y_axis.y,
            self.z_axis.y,
            self.x_axis.z,
            self.y_axis.z,
            self.z_axis.z,
        )
    }

    #[inline]
    fn mul_vector(&self, other: XYZ<T>) -> XYZ<T> {
        let mut res = self.x_axis.mul_scalar(other.x);
        res = self.y_axis.mul_scalar(other.y).add(res);
        res = self.z_axis.mul_scalar(other.z).add(res);
        res
    }

    #[inline]
    fn mul_matrix(&self, other: &Self) -> Self {
        Self::from_cols(
            self.mul_vector(other.x_axis),
            self.mul_vector(other.y_axis),
            self.mul_vector(other.z_axis),
        )
    }

    #[inline]
    fn mul_scalar(&self, other: T) -> Self {
        Self::from_cols(
            self.x_axis.mul_scalar(other),
            self.y_axis.mul_scalar(other),
            self.z_axis.mul_scalar(other),
        )
    }

    #[inline]
    fn add_matrix(&self, other: &Self) -> Self {
        Self::from_cols(
            self.x_axis.add(other.x_axis),
            self.y_axis.add(other.y_axis),
            self.z_axis.add(other.z_axis),
        )
    }

    #[inline]
    fn sub_matrix(&self, other: &Self) -> Self {
        Self::from_cols(
            self.x_axis.sub(other.x_axis),
            self.y_axis.sub(other.y_axis),
            self.z_axis.sub(other.z_axis),
        )
    }
}

impl<T: FloatEx> FloatMatrix3x3<T> for XYZx3<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool {
        self.x_axis.abs_diff_eq(other.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(other.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(other.z_axis, max_abs_diff)
    }

    fn inverse(&self) -> Self {
        let tmp0 = self.y_axis.cross(self.z_axis);
        let tmp1 = self.z_axis.cross(self.x_axis);
        let tmp2 = self.x_axis.cross(self.y_axis);
        let det = self.z_axis.dot_into_vec(tmp2);
        glam_assert!(det.cmpne(XYZ::ZERO).all());
        let inv_det = det.recip();
        // TODO: Work out if it's possible to get rid of the transpose
        Self::from_cols(tmp0.mul(inv_det), tmp1.mul(inv_det), tmp2.mul(inv_det)).transpose()
    }
}
