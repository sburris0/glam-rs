use super::scalar_traits::Float;
use super::storage::{XY, XYZ, XYZW};

pub trait MaskVectorConsts: Sized {
    const FALSE: Self;
}

pub trait MaskVector: MaskVectorConsts {
    fn bitand(self, other: Self) -> Self;
    fn bitor(self, other: Self) -> Self;
    fn not(self) -> Self;
}

pub trait MaskVector2: MaskVector {
    fn new(x: bool, y: bool) -> Self;
    fn bitmask(self) -> u32;
    fn any(self) -> bool;
    fn all(self) -> bool;
}

pub trait MaskVector3: MaskVector {
    fn new(x: bool, y: bool, z: bool) -> Self;
    fn bitmask(self) -> u32;
    fn any(self) -> bool;
    fn all(self) -> bool;
}

pub trait MaskVector4: MaskVector {
    fn new(x: bool, y: bool, z: bool, w: bool) -> Self;
    fn bitmask(self) -> u32;
    fn any(self) -> bool;
    fn all(self) -> bool;
}

pub trait VectorConsts {
    const ZERO: Self;
    const ONE: Self;
}

pub trait Vector2Consts: VectorConsts {
    const UNIT_X: Self;
    const UNIT_Y: Self;
}

pub trait Vector3Consts: VectorConsts {
    const UNIT_X: Self;
    const UNIT_Y: Self;
    const UNIT_Z: Self;
}

pub trait Vector4Consts: VectorConsts {
    const UNIT_X: Self;
    const UNIT_Y: Self;
    const UNIT_Z: Self;
    const UNIT_W: Self;
}

pub trait Vector<T>: Sized + Copy + Clone {
    type Mask;

    fn splat(s: T) -> Self;

    fn select(mask: Self::Mask, a: Self, b: Self) -> Self;

    fn cmpeq(self, other: Self) -> Self::Mask;
    fn cmpne(self, other: Self) -> Self::Mask;
    fn cmpge(self, other: Self) -> Self::Mask;
    fn cmpgt(self, other: Self) -> Self::Mask;
    fn cmple(self, other: Self) -> Self::Mask;
    fn cmplt(self, other: Self) -> Self::Mask;

    fn add(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sub(self, other: Self) -> Self;

    fn scale(self, other: T) -> Self {
        self.mul_scalar(other)
    }

    fn mul_scalar(self, other: T) -> Self;
    fn div_scalar(self, other: T) -> Self;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    fn min_element(self) -> T;
    fn max_element(self) -> T;
}

pub trait Vector2<T>: Vector<T> {
    fn new(x: T, y: T) -> Self;
    fn from_slice_unaligned(slice: &[T]) -> Self;
    fn write_to_slice_unaligned(self, slice: &mut [T]);
    fn deref(&self) -> &XY<T>;
    fn deref_mut(&mut self) -> &mut XY<T>;
    fn into_xyz(self, z: T) -> XYZ<T>;
    fn into_xyzw(self, z: T, w: T) -> XYZW<T>;
    fn from_array(a: [T; 2]) -> Self;
    fn into_array(self) -> [T; 2];
    fn from_tuple(t: (T, T)) -> Self;
    fn into_tuple(self) -> (T, T);
}

pub trait Vector3<T>: Vector<T> {
    fn new(x: T, y: T, z: T) -> Self;
    fn from_slice_unaligned(slice: &[T]) -> Self;
    fn write_to_slice_unaligned(self, slice: &mut [T]);
    fn deref(&self) -> &XYZ<T>;
    fn deref_mut(&mut self) -> &mut XYZ<T>;
    fn into_xy(self) -> XY<T>;
    fn into_xyzw(self, w: T) -> XYZW<T>;
    fn from_array(a: [T; 3]) -> Self;
    fn into_array(self) -> [T; 3];
    fn from_tuple(t: (T, T, T)) -> Self;
    fn into_tuple(self) -> (T, T, T);
}

pub trait Vector4<T>: Vector<T> {
    fn new(x: T, y: T, z: T, w: T) -> Self;
    fn from_slice_unaligned(slice: &[T]) -> Self;
    fn write_to_slice_unaligned(self, slice: &mut [T]);
    fn deref(&self) -> &XYZW<T>;
    fn deref_mut(&mut self) -> &mut XYZW<T>;
    fn into_xy(self) -> XY<T>;
    fn into_xyz(self) -> XYZ<T>;
    fn from_array(a: [T; 4]) -> Self;
    fn into_array(self) -> [T; 4];
    fn from_tuple(t: (T, T, T, T)) -> Self;
    fn into_tuple(self) -> (T, T, T, T);
}

pub trait FloatVector<T: Float>: Vector<T> {
    fn abs(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn is_nan(self) -> Self::Mask;
    fn neg(self) -> Self;
    fn recip(self) -> Self;
    fn round(self) -> Self;
    fn signum(self) -> Self;
}

pub trait FloatVector2<T: Float>: FloatVector<T> + Vector2<T> {
    fn dot(self, other: Self) -> T;

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }

    #[inline]
    fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    fn length_recip(self) -> T {
        self.length().recip()
    }

    #[inline]
    fn normalize(self) -> Self {
        self.mul_scalar(self.length_recip())
    }

    #[inline]
    fn length_squared(self) -> T {
        self.dot(self)
    }

    #[inline]
    fn is_normalized(self) -> bool {
        // TODO: do something with epsilon
        (self.length_squared() - T::ONE).abs() <= T::from_f64(1e-6)
    }

    fn abs_diff_eq(self, other: Self, max_abs_diff: T) -> bool
    where
        <Self as Vector<T>>::Mask: MaskVector2,
    {
        self.sub(other).abs().cmple(Self::splat(max_abs_diff)).all()
    }

    fn perp(self) -> Self;

    fn perp_dot(self, other: Self) -> T;

    #[inline]
    fn angle_between(self, other: Self) -> T {
        let angle = (self.dot(other) / (self.length_squared() * other.length_squared()).sqrt())
            .acos_approx();

        if self.perp_dot(other) < T::ZERO {
            -angle
        } else {
            angle
        }
    }
}

pub trait FloatVector3<T: Float>: FloatVector<T> + Vector3<T> {
    fn dot(self, other: Self) -> T;
    fn cross(self, other: Self) -> Self;

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }

    #[inline]
    fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    fn length_recip(self) -> T {
        self.length().recip()
    }

    #[inline]
    fn normalize(self) -> Self {
        self.mul_scalar(self.length_recip())
    }

    #[inline]
    fn length_squared(self) -> T {
        self.dot(self)
    }

    #[inline]
    fn is_normalized(self) -> bool {
        // TODO: do something with epsilon
        (self.length_squared() - T::ONE).abs() <= T::from_f64(1e-6)
    }

    fn abs_diff_eq(self, other: Self, max_abs_diff: T) -> bool
    where
        <Self as Vector<T>>::Mask: MaskVector3,
    {
        self.sub(other).abs().cmple(Self::splat(max_abs_diff)).all()
    }

    fn angle_between(self, other: Self) -> T {
        self.dot(other)
            .div(self.length_squared().mul(other.length_squared()).sqrt())
            .acos_approx()
    }
}

pub trait FloatVector4<T: Float>: FloatVector<T> + Vector4<T> {
    fn dot(self, other: Self) -> T;

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }

    #[inline]
    fn length(self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    fn length_recip(self) -> T {
        self.length().recip()
    }

    #[inline]
    fn normalize(self) -> Self {
        self.mul_scalar(self.length_recip())
    }

    #[inline]
    fn length_squared(self) -> T {
        self.dot(self)
    }

    #[inline]
    fn is_normalized(self) -> bool {
        // TODO: do something with epsilon
        (self.length_squared() - T::ONE).abs() <= T::from_f64(1e-6)
    }

    fn abs_diff_eq(self, other: Self, max_abs_diff: T) -> bool
    where
        <Self as Vector<T>>::Mask: MaskVector4,
    {
        self.sub(other).abs().cmple(Self::splat(max_abs_diff)).all()
    }
}

pub trait Quaternion<T: Float>: FloatVector4<T> {
    fn from_axis_angle(axis: XYZ<T>, angle: T) -> Self {
        glam_assert!(axis.is_normalized());
        let (s, c) = (angle * T::HALF).sin_cos();
        let v = axis.mul_scalar(s);
        Self::new(v.x, v.y, v.z, c)
    }

    fn from_rotation_ypr(yaw: T, pitch: T, roll: T) -> Self {
        // Self::from_rotation_y(yaw) * Self::from_rotation_x(pitch) * Self::from_rotation_z(roll)
        let (y0, w0) = (yaw * T::HALF).sin_cos();
        let (x1, w1) = (pitch * T::HALF).sin_cos();
        let (z2, w2) = (roll * T::HALF).sin_cos();

        let x3 = w0 * x1;
        let y3 = y0 * w1;
        let z3 = -y0 * x1;
        let w3 = w0 * w1;

        let x4 = x3 * w2 + y3 * z2;
        let y4 = -x3 * z2 + y3 * w2;
        let z4 = w3 * z2 + z3 * w2;
        let w4 = w3 * w2 - z3 * z2;

        Self::new(x4, y4, z4, w4)
    }

    #[inline]
    fn from_rotation_x(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos();
        Self::new(s, T::ZERO, T::ZERO, c)
    }

    #[inline]
    fn from_rotation_y(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos();
        Self::new(T::ZERO, s, T::ZERO, c)
    }

    #[inline]
    fn from_rotation_z(angle: T) -> Self {
        let (s, c) = (angle * T::HALF).sin_cos();
        Self::new(T::ZERO, T::ZERO, s, c)
    }

    fn to_axis_angle(self) -> (XYZ<T>, T) {
        // const EPSILON: f32 = 1.0e-8;
        // const EPSILON_SQUARED: f32 = EPSILON * EPSILON;
        let (x, y, z, w) = Vector4::into_tuple(self);
        let angle = w.acos_approx() * T::TWO;
        let scale_sq = (T::ONE - w * w).max(T::ZERO);
        // TODO: constants for epslions?
        if scale_sq >= T::from_f32(1.0e-8 * 1.0e-8) {
            (XYZ { x, y, z }.mul_scalar(scale_sq.sqrt().recip()), angle)
        } else {
            (Vector3Consts::UNIT_X, angle)
        }
    }

    fn is_near_identity(self) -> bool {
        // Based on https://github.com/nfrechette/rtm `rtm::quat_near_identity`
        let threshold_angle = T::from_f64(0.002_847_144_6);
        // Because of floating point precision, we cannot represent very small rotations.
        // The closest f32 to 1.0 that is not 1.0 itself yields:
        // 0.99999994.acos() * 2.0  = 0.000690533954 rad
        //
        // An error threshold of 1.e-6 is used by default.
        // (1.0 - 1.e-6).acos() * 2.0 = 0.00284714461 rad
        // (1.0 - 1.e-7).acos() * 2.0 = 0.00097656250 rad
        //
        // We don't really care about the angle value itself, only if it's close to 0.
        // This will happen whenever quat.w is close to 1.0.
        // If the quat.w is close to -1.0, the angle will be near 2*PI which is close to
        // a negative 0 rotation. By forcing quat.w to be positive, we'll end up with
        // the shortest path.
        let positive_w_angle = self.deref().w.abs().acos_approx() * T::TWO;
        positive_w_angle < threshold_angle
    }

    fn conjugate(self) -> Self;
    fn lerp(self, end: Self, s: T) -> Self;
    fn slerp(self, end: Self, s: T) -> Self;
    fn mul_quaternion(self, other: Self) -> Self;
    // fn rotate_vector<T: FloatVector3>(self, other: T) -> T;
}
