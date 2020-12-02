use crate::core::{
    storage::{XY, XYZ, XYZW},
    traits::{quaternion::Quaternion, scalar::*, vector::*},
};

impl<T> XY<T> {
    #[inline(always)]
    fn map<D, F>(self, f: F) -> XY<D>
    where
        F: Fn(T) -> D,
    {
        XY {
            x: f(self.x),
            y: f(self.y),
        }
    }

    #[inline(always)]
    fn map2<D, F>(self, other: Self, f: F) -> XY<D>
    where
        F: Fn(T, T) -> D,
    {
        XY {
            x: f(self.x, other.x),
            y: f(self.y, other.y),
        }
    }

    #[inline(always)]
    fn map3<D, F>(self, a: Self, b: Self, f: F) -> XY<D>
    where
        F: Fn(T, T, T) -> D,
    {
        XY {
            x: f(self.x, a.x, b.x),
            y: f(self.y, a.y, b.y),
        }
    }
}

impl<T> XYZ<T> {
    #[inline(always)]
    fn map<D, F>(self, f: F) -> XYZ<D>
    where
        F: Fn(T) -> D,
    {
        XYZ {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }

    #[inline(always)]
    fn map2<D, F>(self, other: Self, f: F) -> XYZ<D>
    where
        F: Fn(T, T) -> D,
    {
        XYZ {
            x: f(self.x, other.x),
            y: f(self.y, other.y),
            z: f(self.z, other.z),
        }
    }

    #[inline(always)]
    fn map3<D, F>(self, a: Self, b: Self, f: F) -> XYZ<D>
    where
        F: Fn(T, T, T) -> D,
    {
        XYZ {
            x: f(self.x, a.x, b.x),
            y: f(self.y, a.y, b.y),
            z: f(self.z, a.z, b.z),
        }
    }
}

impl<T> XYZW<T> {
    #[inline(always)]
    fn map<D, F>(self, f: F) -> XYZW<D>
    where
        F: Fn(T) -> D,
    {
        XYZW {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
            w: f(self.w),
        }
    }

    #[inline(always)]
    fn map2<D, F>(self, other: Self, f: F) -> XYZW<D>
    where
        F: Fn(T, T) -> D,
    {
        XYZW {
            x: f(self.x, other.x),
            y: f(self.y, other.y),
            z: f(self.z, other.z),
            w: f(self.w, other.w),
        }
    }

    #[inline(always)]
    fn map3<D, F>(self, a: Self, b: Self, f: F) -> XYZW<D>
    where
        F: Fn(T, T, T) -> D,
    {
        XYZW {
            x: f(self.x, a.x, b.x),
            y: f(self.y, a.y, b.y),
            z: f(self.z, a.z, b.z),
            w: f(self.w, a.w, b.w),
        }
    }
}

impl MaskVectorConsts for XY<u32> {
    const FALSE: Self = Self { x: 0, y: 0 };
}

impl MaskVectorConsts for XYZ<u32> {
    const FALSE: Self = Self { x: 0, y: 0, z: 0 };
}

impl MaskVectorConsts for XYZW<u32> {
    const FALSE: Self = Self {
        x: 0,
        y: 0,
        z: 0,
        w: 0,
    };
}

impl MaskVector for XY<u32> {
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.map2(other, |a, b| a & b)
    }

    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.map2(other, |a, b| a | b)
    }

    #[inline]
    fn not(self) -> Self {
        self.map(|a| !a)
    }
}

impl MaskVector for XYZ<u32> {
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.map2(other, |a, b| a & b)
    }

    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.map2(other, |a, b| a | b)
    }

    #[inline]
    fn not(self) -> Self {
        self.map(|a| !a)
    }
}

impl MaskVector for XYZW<u32> {
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.map2(other, |a, b| a & b)
    }

    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.map2(other, |a, b| a | b)
    }

    #[inline]
    fn not(self) -> Self {
        self.map(|a| !a)
    }
}

impl MaskVector2 for XY<u32> {
    #[inline(always)]
    fn new(x: bool, y: bool) -> Self {
        Self {
            x: MaskConsts::MASK[x as usize],
            y: MaskConsts::MASK[y as usize],
        }
    }

    #[inline]
    fn bitmask(self) -> u32 {
        (self.x as u32 & 0x1) | (self.y as u32 & 0x1) << 1
    }

    #[inline]
    fn any(self) -> bool {
        ((self.x | self.y) & 0x1) != 0
    }

    #[inline]
    fn all(self) -> bool {
        ((self.x & self.y) & 0x1) != 0
    }
}

impl MaskVector3 for XYZ<u32> {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool) -> Self {
        // A SSE2 mask can be any bit pattern but for the `Vec3Mask` implementation of select
        // we expect either 0 or 0xff_ff_ff_ff. This should be a safe assumption as this type
        // can only be created via this function or by `Vec3` methods.
        Self {
            x: MaskConsts::MASK[x as usize],
            y: MaskConsts::MASK[y as usize],
            z: MaskConsts::MASK[z as usize],
        }
    }

    #[inline]
    fn bitmask(self) -> u32 {
        (self.x & 0x1) | (self.y & 0x1) << 1 | (self.z & 0x1) << 2
    }

    #[inline]
    fn any(self) -> bool {
        ((self.x | self.y | self.z) & 0x1) != 0
    }

    #[inline]
    fn all(self) -> bool {
        ((self.x & self.y & self.z) & 0x1) != 0
    }
}

impl MaskVector4 for XYZW<u32> {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        // A SSE2 mask can be any bit pattern but for the `Vec4Mask` implementation of select
        // we expect either 0 or 0xff_ff_ff_ff. This should be a safe assumption as this type
        // can only be created via this function or by `Vec4` methods.
        Self {
            x: MaskConsts::MASK[x as usize],
            y: MaskConsts::MASK[y as usize],
            z: MaskConsts::MASK[z as usize],
            w: MaskConsts::MASK[w as usize],
        }
    }

    #[inline]
    fn bitmask(self) -> u32 {
        (self.x & 0x1) | (self.y & 0x1) << 1 | (self.z & 0x1) << 2 | (self.w & 0x1) << 3
    }

    #[inline]
    fn any(self) -> bool {
        ((self.x | self.y | self.z | self.w) & 0x1) != 0
    }

    #[inline]
    fn all(self) -> bool {
        ((self.x & self.y & self.z & self.w) & 0x1) != 0
    }
}

impl<T: Num> VectorConsts for XY<T> {
    const ZERO: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
    };
    const ONE: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ONE,
    };
}

impl<T: Num> Vector2Consts for XY<T> {
    const UNIT_X: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ZERO,
    };
    const UNIT_Y: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ONE,
    };
}

impl<T: Num> VectorConsts for XYZ<T> {
    const ZERO: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ZERO,
    };
    const ONE: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ONE,
        z: <T as NumConsts>::ONE,
    };
}
impl<T: Num> Vector3Consts for XYZ<T> {
    const UNIT_X: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ZERO,
    };
    const UNIT_Y: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ONE,
        z: <T as NumConsts>::ZERO,
    };
    const UNIT_Z: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ONE,
    };
}

impl<T: Num> VectorConsts for XYZW<T> {
    const ZERO: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ZERO,
        w: <T as NumConsts>::ZERO,
    };
    const ONE: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ONE,
        z: <T as NumConsts>::ONE,
        w: <T as NumConsts>::ONE,
    };
}
impl<T: Num> Vector4Consts for XYZW<T> {
    const UNIT_X: Self = Self {
        x: <T as NumConsts>::ONE,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ZERO,
        w: <T as NumConsts>::ZERO,
    };
    const UNIT_Y: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ONE,
        z: <T as NumConsts>::ZERO,
        w: <T as NumConsts>::ZERO,
    };
    const UNIT_Z: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ONE,
        w: <T as NumConsts>::ZERO,
    };
    const UNIT_W: Self = Self {
        x: <T as NumConsts>::ZERO,
        y: <T as NumConsts>::ZERO,
        z: <T as NumConsts>::ZERO,
        w: <T as NumConsts>::ONE,
    };
}

impl<T: Num> Vector<T> for XY<T> {
    type Mask = XY<u32>;

    #[inline]
    fn splat(s: T) -> Self {
        Self { x: s, y: s }
    }

    #[inline]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        Self {
            x: if mask.x != 0 { if_true.x } else { if_false.x },
            y: if mask.y != 0 { if_true.y } else { if_false.y },
        }
    }

    #[inline]
    fn cmpeq(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.eq(&b) as usize])
    }

    #[inline]
    fn cmpne(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ne(&b) as usize])
    }

    #[inline]
    fn cmpge(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ge(&b) as usize])
    }

    #[inline]
    fn cmpgt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.gt(&b) as usize])
    }

    #[inline]
    fn cmple(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.le(&b) as usize])
    }

    #[inline]
    fn cmplt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.lt(&b) as usize])
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        self.map2(other, |a, b| a + b)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        self.map2(other, |a, b| a / b)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.map2(other, |a, b| a * b)
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self.map3(b, c, |a, b, c| a * b + c)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.map2(other, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(self, other: T) -> Self {
        self.map(|a| a * other)
    }

    #[inline]
    fn div_scalar(self, other: T) -> Self {
        self.map(|a| a / other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.map2(other, |a, b| a.min(b))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.map2(other, |a, b| a.max(b))
    }
}

impl<T: Num> Vector<T> for XYZ<T> {
    type Mask = XYZ<u32>;

    #[inline]
    fn splat(s: T) -> Self {
        Self { x: s, y: s, z: s }
    }

    #[inline]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        Self {
            x: if mask.x != 0 { if_true.x } else { if_false.x },
            y: if mask.y != 0 { if_true.y } else { if_false.y },
            z: if mask.z != 0 { if_true.z } else { if_false.z },
        }
    }

    #[inline]
    fn cmpeq(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.eq(&b) as usize])
    }

    #[inline]
    fn cmpne(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ne(&b) as usize])
    }

    #[inline]
    fn cmpge(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ge(&b) as usize])
    }

    #[inline]
    fn cmpgt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.gt(&b) as usize])
    }

    #[inline]
    fn cmple(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.le(&b) as usize])
    }

    #[inline]
    fn cmplt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.lt(&b) as usize])
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        self.map2(other, |a, b| a + b)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        self.map2(other, |a, b| a / b)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.map2(other, |a, b| a * b)
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self.map3(b, c, |a, b, c| a * b + c)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.map2(other, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(self, other: T) -> Self {
        self.map(|a| a * other)
    }

    #[inline]
    fn div_scalar(self, other: T) -> Self {
        self.map(|a| a / other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.map2(other, |a, b| a.min(b))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.map2(other, |a, b| a.max(b))
    }
}

impl<T: Num> Vector<T> for XYZW<T> {
    type Mask = XYZW<u32>;

    #[inline]
    fn splat(s: T) -> Self {
        Self {
            x: s,
            y: s,
            z: s,
            w: s,
        }
    }

    #[inline]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        Self {
            x: if mask.x != 0 { if_true.x } else { if_false.x },
            y: if mask.y != 0 { if_true.y } else { if_false.y },
            z: if mask.z != 0 { if_true.z } else { if_false.z },
            w: if mask.w != 0 { if_true.w } else { if_false.w },
        }
    }

    #[inline]
    fn cmpeq(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.eq(&b) as usize])
    }

    #[inline]
    fn cmpne(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ne(&b) as usize])
    }

    #[inline]
    fn cmpge(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.ge(&b) as usize])
    }

    #[inline]
    fn cmpgt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.gt(&b) as usize])
    }

    #[inline]
    fn cmple(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.le(&b) as usize])
    }

    #[inline]
    fn cmplt(self, other: Self) -> Self::Mask {
        self.map2(other, |a, b| MaskConsts::MASK[a.lt(&b) as usize])
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        self.map2(other, |a, b| a + b)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        self.map2(other, |a, b| a / b)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.map2(other, |a, b| a * b)
    }

    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self.map3(b, c, |a, b, c| a * b + c)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.map2(other, |a, b| a - b)
    }

    #[inline]
    fn mul_scalar(self, other: T) -> Self {
        self.map(|a| a * other)
    }

    #[inline]
    fn div_scalar(self, other: T) -> Self {
        self.map(|a| a / other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        self.map2(other, |a, b| a.min(b))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        self.map2(other, |a, b| a.max(b))
    }
}

impl<T: Num> Vector2<T> for XY<T> {
    #[inline(always)]
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[T]) -> Self {
        Self {
            x: slice[0],
            y: slice[1],
        }
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [T]) {
        slice[0] = self.x;
        slice[1] = self.y;
    }

    #[inline(always)]
    fn deref(&self) -> &XY<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XY<T> {
        self
    }

    #[inline(always)]
    fn into_xyz(self, z: T) -> XYZ<T> {
        XYZ {
            x: self.x,
            y: self.y,
            z,
        }
    }

    #[inline(always)]
    fn into_xyzw(self, z: T, w: T) -> XYZW<T> {
        XYZW {
            x: self.x,
            y: self.y,
            z,
            w,
        }
    }

    #[inline(always)]
    fn from_array(a: [T; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    #[inline(always)]
    fn into_array(self) -> [T; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    fn from_tuple(t: (T, T)) -> Self {
        Self::new(t.0, t.1)
    }

    #[inline(always)]
    fn into_tuple(self) -> (T, T) {
        (self.x, self.y)
    }

    #[inline]
    fn min_element(self) -> T {
        self.x.min(self.y)
    }

    #[inline]
    fn max_element(self) -> T {
        self.x.max(self.y)
    }

    #[inline]
    fn dot(self, other: Self) -> T {
        (self.x * other.x) + (self.y * other.y)
    }
}

impl<T: Num> Vector3<T> for XYZ<T> {
    #[inline(always)]
    fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[T]) -> Self {
        Self {
            x: slice[0],
            y: slice[1],
            z: slice[2],
        }
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [T]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
    }

    #[inline(always)]
    fn deref(&self) -> &XYZ<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZ<T> {
        self
    }

    #[inline(always)]
    fn into_xy(self) -> XY<T> {
        XY {
            x: self.x,
            y: self.y,
        }
    }

    #[inline(always)]
    fn into_xyzw(self, w: T) -> XYZW<T> {
        XYZW {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    #[inline(always)]
    fn from_array(a: [T; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    #[inline(always)]
    fn into_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }

    #[inline(always)]
    fn from_tuple(t: (T, T, T)) -> Self {
        Self::new(t.0, t.1, t.2)
    }

    #[inline(always)]
    fn into_tuple(self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }

    #[inline]
    fn min_element(self) -> T {
        self.x.min(self.y.min(self.z))
    }

    #[inline]
    fn max_element(self) -> T {
        self.x.max(self.y.max(self.z))
    }

    #[inline]
    fn dot(self, other: Self) -> T {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }
}

impl<T: Num> Vector4<T> for XYZW<T> {
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[T]) -> Self {
        Self {
            x: slice[0],
            y: slice[1],
            z: slice[2],
            w: slice[3],
        }
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [T]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    #[inline(always)]
    fn deref(&self) -> &XYZW<T> {
        self
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZW<T> {
        self
    }

    #[inline(always)]
    fn into_xy(self) -> XY<T> {
        XY {
            x: self.x,
            y: self.y,
        }
    }

    #[inline(always)]
    fn into_xyz(self) -> XYZ<T> {
        XYZ {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    #[inline(always)]
    fn from_array(a: [T; 4]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
            w: a[3],
        }
    }

    #[inline(always)]
    fn into_array(self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }

    #[inline(always)]
    fn from_tuple(t: (T, T, T, T)) -> Self {
        Self::new(t.0, t.1, t.2, t.3)
    }

    #[inline(always)]
    fn into_tuple(self) -> (T, T, T, T) {
        (self.x, self.y, self.z, self.w)
    }

    #[inline]
    fn min_element(self) -> T {
        self.x.min(self.y.min(self.z.min(self.w)))
    }

    #[inline]
    fn max_element(self) -> T {
        self.x.max(self.y.max(self.z.min(self.w)))
    }

    #[inline]
    fn dot(self, other: Self) -> T {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z) + (self.w * other.w)
    }
}

impl<T: Float> FloatVector<T> for XY<T> {
    #[inline]
    fn abs(self) -> Self {
        self.map(Float::abs)
    }

    #[inline]
    fn floor(self) -> Self {
        self.map(Float::floor)
    }

    #[inline]
    fn ceil(self) -> Self {
        self.map(Float::ceil)
    }

    #[inline]
    fn round(self) -> Self {
        self.map(Float::round)
    }

    #[inline]
    fn neg(self) -> Self {
        self.map(|a| a.neg())
    }

    #[inline]
    fn recip(self) -> Self {
        self.map(Float::recip)
    }

    #[inline]
    fn signum(self) -> Self {
        self.map(Float::signum)
    }
}

impl<T: Float> FloatVector<T> for XYZ<T> {
    #[inline]
    fn abs(self) -> Self {
        self.map(Float::abs)
    }

    #[inline]
    fn floor(self) -> Self {
        self.map(Float::floor)
    }

    #[inline]
    fn ceil(self) -> Self {
        self.map(Float::ceil)
    }

    #[inline]
    fn round(self) -> Self {
        self.map(Float::round)
    }

    #[inline]
    fn neg(self) -> Self {
        self.map(|a| a.neg())
    }

    #[inline]
    fn recip(self) -> Self {
        self.map(Float::recip)
    }

    #[inline]
    fn signum(self) -> Self {
        self.map(Float::signum)
    }
}

impl<T: Float> FloatVector<T> for XYZW<T> {
    #[inline]
    fn abs(self) -> Self {
        self.map(Float::abs)
    }

    #[inline]
    fn floor(self) -> Self {
        self.map(Float::floor)
    }

    #[inline]
    fn ceil(self) -> Self {
        self.map(Float::ceil)
    }

    #[inline]
    fn round(self) -> Self {
        self.map(Float::round)
    }

    #[inline]
    fn neg(self) -> Self {
        self.map(|a| a.neg())
    }

    #[inline]
    fn recip(self) -> Self {
        self.map(Float::recip)
    }

    #[inline]
    fn signum(self) -> Self {
        self.map(Float::signum)
    }
}

impl<T: Float> FloatVector2<T> for XY<T> {
    #[inline]
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    #[inline]
    fn is_nan_mask(self) -> Self::Mask {
        self.map(|a| MaskConsts::MASK[a.is_nan() as usize])
    }

    #[inline]
    fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    #[inline]
    fn perp_dot(self, other: Self) -> T {
        (self.x * other.y) - (self.y * other.x)
    }
}

impl<T: Float> FloatVector3<T> for XYZ<T> {
    #[inline]
    fn is_nan_mask(self) -> Self::Mask {
        self.map(|a| MaskConsts::MASK[a.is_nan() as usize])
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - other.y * self.z,
            y: self.z * other.x - other.z * self.x,
            z: self.x * other.y - other.x * self.y,
        }
    }
}

impl<T: Float> FloatVector4<T> for XYZW<T> {
    #[inline]
    fn is_nan_mask(self) -> Self::Mask {
        self.map(|a| MaskConsts::MASK[a.is_nan() as usize])
    }
}

impl<T: Float> Quaternion<T> for XYZW<T> {
    // fn from_axis_angle(axis: XYZ<T>, angle: T) -> Self {
    // }

    // fn to_axis_angle(self) -> (XYZ<T>, T) {
    // }

    //fn is_near_identity(self) -> bool {
    //    // Based on https://github.com/nfrechette/rtm `rtm::quat_near_identity`
    //    const THRESHOLD_ANGLE: f32 = 0.002_847_144_6;
    //    // Because of floating point precision, we cannot represent very small rotations.
    //    // The closest f32 to 1.0 that is not 1.0 itself yields:
    //    // 0.99999994.acos() * 2.0  = 0.000690533954 rad
    //    //
    //    // An error threshold of 1.e-6 is used by default.
    //    // (1.0 - 1.e-6).acos() * 2.0 = 0.00284714461 rad
    //    // (1.0 - 1.e-7).acos() * 2.0 = 0.00097656250 rad
    //    //
    //    // We don't really care about the angle value itself, only if it's close to 0.
    //    // This will happen whenever quat.w is close to 1.0.
    //    // If the quat.w is close to -1.0, the angle will be near 2*PI which is close to
    //    // a negative 0 rotation. By forcing quat.w to be positive, we'll end up with
    //    // the shortest path.
    //    let positive_w_angle = self.w.abs().acos_approx() * 2.0;
    //    positive_w_angle < THRESHOLD_ANGLE
    //}

    fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    fn lerp(self, end: Self, s: T) -> Self {
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        let start = self;
        let end = end;
        let dot = start.dot(end);
        let bias = if dot >= T::ZERO { T::ONE } else { T::NEG_ONE };
        let interpolated = start.add(end.mul_scalar(bias).sub(start).mul_scalar(s));
        interpolated.normalize()
    }

    fn slerp(self, end: Self, s: T) -> Self {
        // http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/

        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        let dot = self.dot(end);

        if dot > T::from_f32(0.9995) {
            // assumes lerp returns a normalized quaternion
            self.lerp(end, s)
        } else {
            // assumes scalar_acos clamps the input to [-1.0, 1.0]
            let theta = dot.acos_approx();
            let scale1 = (theta * (T::ONE - s)).sin();
            let scale2 = (theta * s).sin();
            let theta_sin = theta.sin();

            self.mul_scalar(scale1)
                .add(end.mul_scalar(scale2))
                .mul_scalar(theta_sin.recip())
        }
    }

    fn mul_vector3(self, other: XYZ<T>) -> XYZ<T> {
        glam_assert!(FloatVector4::is_normalized(self));
        let w = self.w;
        let b = XYZ {
            x: self.x,
            y: self.y,
            z: self.z,
        };
        let b2 = b.dot(b);
        other
            .mul_scalar(w * w - b2)
            .add(b.mul_scalar(other.dot(b) * T::TWO))
            .add(b.cross(other).mul_scalar(w * T::TWO))
    }

    fn mul_quaternion(self, other: Self) -> Self {
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(other));
        let (x0, y0, z0, w0) = self.into_tuple();
        let (x1, y1, z1, w1) = other.into_tuple();
        Self::new(
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        )
    }
}

impl<T> From<XYZW<T>> for XYZ<T> {
    #[inline(always)]
    fn from(v: XYZW<T>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl<T> From<XYZW<T>> for XY<T> {
    #[inline(always)]
    fn from(v: XYZW<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl<T> From<XYZ<T>> for XY<T> {
    #[inline(always)]
    fn from(v: XYZ<T>) -> Self {
        Self { x: v.x, y: v.y }
    }
}
