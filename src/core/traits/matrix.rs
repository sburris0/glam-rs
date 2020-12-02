use crate::core::{storage::XY, traits::scalar::Float};

pub trait MatrixConsts {
    const IDENTITY: Self;
}

pub trait Matrix<T>: Sized + Copy + Clone {}

pub trait Matrix2x2<T>: Matrix<T> {
    fn from_cols(x_axis: XY<T>, y_axis: XY<T>) -> Self;
    fn from_cols_array(m: &[T; 4]) -> Self;
    fn to_cols_array(&self) -> [T; 4];
    fn from_cols_array_2d(m: &[[T; 2]; 2]) -> Self;
    fn to_cols_array_2d(&self) -> [[T; 2]; 2];
    fn from_scale(scale: XY<T>) -> Self;
    fn determinant(&self) -> T;
    fn transpose(&self) -> Self;
    fn add_matrix(&self, other: &Self) -> Self;
    fn mul_vector(&self, other: XY<T>) -> XY<T>;
    fn mul_matrix(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: T) -> Self;
    fn sub_matrix(&self, other: &Self) -> Self;
}

pub trait FloatMatrix2x2<T: Float>: Matrix2x2<T> {
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: T) -> bool;
    fn from_scale_angle(scale: XY<T>, angle: T) -> Self;
    fn from_angle(angle: T) -> Self;
    fn inverse(&self) -> Self;
    // fn is_finite(&self) -> bool;
}

mod scalar {
    use crate::core::{
        storage::{XYAxes, XY},
        traits::{
            matrix::{FloatMatrix2x2, Matrix, Matrix2x2, MatrixConsts},
            scalar::{Float, Num},
            vector::{FloatVector2, Vector, Vector2, Vector2Consts},
        },
    };

    impl<T: Num> MatrixConsts for XYAxes<T> {
        const IDENTITY: Self = Self {
            x_axis: XY::UNIT_X,
            y_axis: XY::UNIT_Y,
        };
    }

    impl<T: Num> Matrix<T> for XYAxes<T> {}

    impl<T: Num> Matrix2x2<T> for XYAxes<T> {
        #[inline(always)]
        fn from_cols(x_axis: XY<T>, y_axis: XY<T>) -> Self {
            Self { x_axis, y_axis }
        }

        #[inline(always)]
        fn from_cols_array(m: &[T; 4]) -> Self {
            Self::from_cols(Vector2::new(m[0], m[1]), Vector2::new(m[2], m[3]))
        }

        #[inline(always)]
        fn to_cols_array(&self) -> [T; 4] {
            [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
        }

        #[inline(always)]
        fn from_cols_array_2d(m: &[[T; 2]; 2]) -> Self {
            Self::from_cols(
                Vector2::new(m[0][0], m[0][1]),
                Vector2::new(m[1][0], m[1][1]),
            )
        }

        #[inline(always)]
        fn to_cols_array_2d(&self) -> [[T; 2]; 2] {
            [
                [self.x_axis.x, self.x_axis.y],
                [self.y_axis.x, self.y_axis.y],
            ]
        }

        #[inline(always)]
        fn from_scale(scale: XY<T>) -> Self {
            Self::from_cols(
                Vector2::new(scale.x, T::ZERO),
                Vector2::new(T::ZERO, scale.y),
            )
        }

        #[inline]
        fn determinant(&self) -> T {
            self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
        }

        #[inline(always)]
        fn transpose(&self) -> Self {
            Self::from_cols(
                Vector2::new(self.x_axis.x, self.y_axis.x),
                Vector2::new(self.x_axis.y, self.y_axis.y),
            )
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
        fn from_scale_angle(scale: XY<T>, angle: T) -> Self {
            let (sin, cos) = angle.sin_cos();
            Self {
                x_axis: Vector2::new(cos * scale.x, sin * scale.x),
                y_axis: Vector2::new(-sin * scale.y, cos * scale.y),
            }
        }

        #[inline]
        fn from_angle(angle: T) -> Self {
            let (sin, cos) = angle.sin_cos();
            Self {
                x_axis: Vector2::new(cos, sin),
                y_axis: Vector2::new(-sin, cos),
            }
        }

        #[inline]
        fn inverse(&self) -> Self {
            let inv_det = {
                let det = self.determinant();
                glam_assert!(det != T::ZERO);
                det.recip()
            };
            Self {
                x_axis: Vector2::new(self.y_axis.y * inv_det, self.x_axis.y * -inv_det),
                y_axis: Vector2::new(self.y_axis.x * -inv_det, self.x_axis.x * inv_det),
            }
        }
    }
}
