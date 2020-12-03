use crate::core::{
    storage::{XYAxes, XY},
    traits::scalar::{Float, Num},
};

pub trait MatrixConsts {
    const ZERO: Self;
    const IDENTITY: Self;
}

pub trait Matrix<T: Num>: Sized + Copy + Clone {}

pub trait Matrix2x2<T: Num>: Matrix<T> {
    fn new(m00: T, m01: T, m10: T, m11: T) -> Self;

    fn deref(&self) -> &XYAxes<T>;
    fn deref_mut(&mut self) -> &mut XYAxes<T>;

    #[inline(always)]
    fn from_cols(x_axis: XY<T>, y_axis: XY<T>) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline(always)]
    fn from_cols_array(m: &[T; 4]) -> Self {
        Self::new(m[0], m[1], m[2], m[3])
    }

    #[inline(always)]
    fn to_cols_array(&self) -> [T; 4] {
        let m = self.deref();
        [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y]
    }

    #[inline(always)]
    fn from_cols_array_2d(m: &[[T; 2]; 2]) -> Self {
        Self::new(m[0][0], m[0][1], m[1][0], m[1][1])
    }

    #[inline(always)]
    fn to_cols_array_2d(&self) -> [[T; 2]; 2] {
        let m = self.deref();
        [[m.x_axis.x, m.x_axis.y], [m.y_axis.x, m.y_axis.y]]
    }

    #[inline(always)]
    fn from_scale(scale: XY<T>) -> Self {
        Self::new(scale.x, T::ZERO, T::ZERO, scale.y)
    }

    fn determinant(&self) -> T;
    fn transpose(&self) -> Self;
    fn mul_vector(&self, other: XY<T>) -> XY<T>;
    fn mul_matrix(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: T) -> Self;
    fn add_matrix(&self, other: &Self) -> Self;
    fn sub_matrix(&self, other: &Self) -> Self;
}

pub trait FloatMatrix2x2<T: Float>: Matrix2x2<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool;

    #[inline]
    fn from_scale_angle(scale: XY<T>, angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos * scale.x, sin * scale.x, -sin * scale.y, cos * scale.y)
    }

    #[inline]
    fn from_angle(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(cos, sin, -sin, cos)
    }

    fn inverse(&self) -> Self;
    // fn is_finite(&self) -> bool;
}
