use crate::core::{
    storage::{XYAxes, XY},
    traits::{
        matrix::{FloatMatrix2x2, Matrix, Matrix2x2, MatrixConsts},
        scalar::{Float, Num},
        vector::{FloatVector2, Vector, Vector2, Vector2Consts, VectorConsts},
    },
};

impl<T: Num> MatrixConsts for XYAxes<T> {
    const ZERO: Self = Self {
        x_axis: XY::ZERO,
        y_axis: XY::ZERO,
    };
    const IDENTITY: Self = Self {
        x_axis: XY::UNIT_X,
        y_axis: XY::UNIT_Y,
    };
}

impl<T: Num> Matrix<T> for XYAxes<T> {}

impl<T: Num> Matrix2x2<T> for XYAxes<T> {
    #[inline(always)]
    fn new(m00: T, m01: T, m10: T, m11: T) -> Self {
        Self {
            x_axis: Vector2::new(m00, m01),
            y_axis: Vector2::new(m10, m11),
        }
    }

    #[inline(always)]
    fn deref(&self) -> &XYAxes<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYAxes<T> {
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

impl<T: Float> FloatMatrix2x2<T> for XYAxes<T> {
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
