use super::scalar::FloatEx;
use crate::core::storage::{XY, XYZ, XYZW};

pub trait MaskVectorConst: Sized {
    const FALSE: Self;
}

pub trait MaskVector: MaskVectorConst {
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

pub trait VectorConst {
    const ZERO: Self;
    const ONE: Self;
}

pub trait Vector2Const: VectorConst {
    const UNIT_X: Self;
    const UNIT_Y: Self;
}

pub trait Vector3Const: VectorConst {
    const UNIT_X: Self;
    const UNIT_Y: Self;
    const UNIT_Z: Self;
}

pub trait Vector4Const: VectorConst {
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
}

pub trait Vector2<T>: Vector<T> + Vector2Const {
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

    fn min_element(self) -> T;
    fn max_element(self) -> T;

    fn dot(self, other: Self) -> T;

    #[inline(always)]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }
}

pub trait Vector3<T>: Vector<T> + Vector3Const {
    fn new(x: T, y: T, z: T) -> Self;
    fn from_slice_unaligned(slice: &[T]) -> Self;
    fn write_to_slice_unaligned(self, slice: &mut [T]);
    fn deref(&self) -> &XYZ<T>;
    fn deref_mut(&mut self) -> &mut XYZ<T>;

    #[inline(always)]
    fn from_xy(v2: XY<T>, z: T) -> Self {
        Self::new(v2.x, v2.y, z)
    }

    #[inline(always)]
    fn from_xyzw(v4: XYZW<T>) -> Self {
        Self::new(v4.x, v4.y, v4.z)
    }

    fn into_xy(self) -> XY<T>;

    fn into_xyzw(self, w: T) -> XYZW<T>;
    fn from_array(a: [T; 3]) -> Self;
    fn into_array(self) -> [T; 3];
    fn from_tuple(t: (T, T, T)) -> Self;
    fn into_tuple(self) -> (T, T, T);

    fn min_element(self) -> T;
    fn max_element(self) -> T;

    fn dot(self, other: Self) -> T;

    #[inline(always)]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }

    fn cross(self, other: Self) -> Self;
}

pub trait Vector4<T>: Vector<T> + Vector4Const {
    fn new(x: T, y: T, z: T, w: T) -> Self;
    fn splat_x(self) -> Self;
    fn splat_y(self) -> Self;
    fn splat_z(self) -> Self;
    fn splat_w(self) -> Self;
    fn from_slice_unaligned(slice: &[T]) -> Self;
    fn write_to_slice_unaligned(self, slice: &mut [T]);
    fn deref(&self) -> &XYZW<T>;
    fn deref_mut(&mut self) -> &mut XYZW<T>;

    #[inline(always)]
    fn from_xy(v2: XY<T>, z: T, w: T) -> Self {
        Self::new(v2.x, v2.y, z, w)
    }

    #[inline(always)]
    fn from_xyz(v3: XYZ<T>, w: T) -> Self {
        Self::new(v3.x, v3.y, v3.z, w)
    }

    fn into_xy(self) -> XY<T>;
    fn into_xyz(self) -> XYZ<T>;
    fn from_array(a: [T; 4]) -> Self;
    fn into_array(self) -> [T; 4];
    fn from_tuple(t: (T, T, T, T)) -> Self;
    fn into_tuple(self) -> (T, T, T, T);

    fn min_element(self) -> T;
    fn max_element(self) -> T;

    fn dot(self, other: Self) -> T;

    #[inline(always)]
    fn dot_into_vec(self, other: Self) -> Self {
        Self::splat(self.dot(other))
    }
}

pub trait FloatVector<T: FloatEx>: Vector<T> {
    fn abs(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn neg(self) -> Self;
    fn recip(self) -> Self;
    fn round(self) -> Self;
    fn signum(self) -> Self;
}

pub trait FloatVector2<T: FloatEx>: FloatVector<T> + Vector2<T> {
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_nan_mask(self) -> Self::Mask;

    #[inline(always)]
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

    #[inline(always)]
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

pub trait FloatVector3<T: FloatEx>: FloatVector<T> + Vector3<T> {
    fn is_nan_mask(self) -> Self::Mask;

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

    #[inline(always)]
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

pub trait FloatVector4<T: FloatEx>: FloatVector<T> + Vector4<T> {
    fn is_nan_mask(self) -> Self::Mask;

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

    #[inline(always)]
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
