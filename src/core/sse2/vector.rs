#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::const_m128;
use crate::core::{
    storage::{Align16, XY, XYZ, XYZW},
    traits::{quaternion::Quaternion, scalar::*, vector::*},
};
use core::mem::MaybeUninit;

#[repr(C)]
union UnionCast {
    pub m128: __m128,
    pub m128i: __m128i,
    pub f32x4: [f32; 4],
    pub i32x4: [i32; 4],
    pub u32x4: [u32; 4],
}

macro_rules! _ps_const_ty {
    ($name:ident, $field:ident, $x:expr) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        const $name: UnionCast = UnionCast {
            $field: [$x, $x, $x, $x],
        };
    };

    ($name:ident, $field:ident, $x:expr, $y:expr, $z:expr, $w:expr) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        const $name: UnionCast = UnionCast {
            $field: [$x, $y, $z, $w],
        };
    };
}

_ps_const_ty!(PS_INV_SIGN_MASK, u32x4, !0x8000_0000);
_ps_const_ty!(PS_SIGN_MASK, u32x4, 0x8000_0000);
_ps_const_ty!(PS_NO_FRACTION, f32x4, 8388608.0);

_ps_const_ty!(PS_NEGATIVE_ZERO, u32x4, 0x80000000);
_ps_const_ty!(PS_PI, f32x4, core::f32::consts::PI);
_ps_const_ty!(PS_HALF_PI, f32x4, core::f32::consts::FRAC_PI_2);
_ps_const_ty!(
    PS_SIN_COEFFICIENTS0,
    f32x4,
    -0.16666667,
    0.0083333310,
    -0.00019840874,
    2.7525562e-06
);
_ps_const_ty!(
    PS_SIN_COEFFICIENTS1,
    f32x4,
    -2.3889859e-08,
    -0.16665852,    /*Est1*/
    0.0083139502,   /*Est2*/
    -0.00018524670  /*Est3*/
);
_ps_const_ty!(PS_ONE, f32x4, 1.0);
_ps_const_ty!(PS_TWO_PI, f32x4, core::f32::consts::PI * 2.0);
_ps_const_ty!(PS_RECIPROCAL_TWO_PI, f32x4, 0.159154943);

impl MaskVectorConst for __m128 {
    const FALSE: __m128 = const_m128!([0.0; 4]);
}

impl MaskVector for __m128 {
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        unsafe { _mm_and_ps(self, other) }
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        unsafe { _mm_or_ps(self, other) }
    }

    #[inline]
    fn not(self) -> Self {
        unsafe { _mm_andnot_ps(self, _mm_set_ps1(f32::from_bits(0xff_ff_ff_ff))) }
    }
}

impl MaskVector3 for __m128 {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool) -> Self {
        // A SSE2 mask can be any bit pattern but for the `MaskVector3` implementation of select we
        // expect either 0 or 0xff_ff_ff_ff. This should be a safe assumption as this type can only
        // be created via this function or by `Vector3` methods.

        unsafe {
            _mm_set_ps(
                0.0,
                f32::from_bits(MaskConst::MASK[z as usize]),
                f32::from_bits(MaskConst::MASK[y as usize]),
                f32::from_bits(MaskConst::MASK[x as usize]),
            )
        }
    }

    #[inline(always)]
    fn bitmask(self) -> u32 {
        unsafe { (_mm_movemask_ps(self) as u32) & 0x7 }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { (_mm_movemask_ps(self) & 0x7) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { (_mm_movemask_ps(self) & 0x7) == 0x7 }
    }
}

impl MaskVector4 for __m128 {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        // A SSE2 mask can be any bit pattern but for the `Vec4Mask` implementation of select we
        // expect either 0 or 0xff_ff_ff_ff. This should be a safe assumption as this type can only
        // be created via this function or by `Vec4` methods.

        const MASK: [u32; 2] = [0, 0xff_ff_ff_ff];
        unsafe {
            _mm_set_ps(
                f32::from_bits(MASK[w as usize]),
                f32::from_bits(MASK[z as usize]),
                f32::from_bits(MASK[y as usize]),
                f32::from_bits(MASK[x as usize]),
            )
        }
    }
    #[inline(always)]
    fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(self) as u32 }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self) == 0xf }
    }
}

/// Calculates the vector 3 dot product and returns answer in x lane of __m128.
#[inline]
unsafe fn dot3_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let x2_y2_z2_w2 = _mm_mul_ps(lhs, rhs);
    let y2_0_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_00_01);
    let z2_0_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_00_10);
    let x2y2_0_0_0 = _mm_add_ss(x2_y2_z2_w2, y2_0_0_0);
    _mm_add_ss(x2y2_0_0_0, z2_0_0_0)
}

/// Calculates the vector 4 dot product and returns answer in x lane of __m128.
#[inline]
unsafe fn dot4_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let x2_y2_z2_w2 = _mm_mul_ps(lhs, rhs);
    let z2_w2_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_11_10);
    let x2z2_y2w2_0_0 = _mm_add_ps(x2_y2_z2_w2, z2_w2_0_0);
    let y2w2_0_0_0 = _mm_shuffle_ps(x2z2_y2w2_0_0, x2z2_y2w2_0_0, 0b00_00_00_01);
    _mm_add_ps(x2z2_y2w2_0_0, y2w2_0_0_0)
}

impl VectorConst for __m128 {
    const ZERO: __m128 = const_m128!([0.0; 4]);
    const ONE: __m128 = const_m128!([1.0; 4]);
}

impl Vector3Const for __m128 {
    const UNIT_X: __m128 = const_m128!([1.0, 0.0, 0.0, 0.0]);
    const UNIT_Y: __m128 = const_m128!([0.0, 1.0, 0.0, 0.0]);
    const UNIT_Z: __m128 = const_m128!([0.0, 0.0, 1.0, 0.0]);
}

impl Vector4Const for __m128 {
    const UNIT_X: __m128 = const_m128!([1.0, 0.0, 0.0, 0.0]);
    const UNIT_Y: __m128 = const_m128!([0.0, 1.0, 0.0, 0.0]);
    const UNIT_Z: __m128 = const_m128!([0.0, 0.0, 1.0, 0.0]);
    const UNIT_W: __m128 = const_m128!([0.0, 0.0, 0.0, 1.0]);
}

impl Vector<f32> for __m128 {
    type Mask = __m128;

    #[inline(always)]
    fn splat(s: f32) -> Self {
        unsafe { _mm_set_ps1(s) }
    }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        unsafe { _mm_or_ps(_mm_andnot_ps(mask, if_false), _mm_and_ps(if_true, mask)) }
    }

    #[inline(always)]
    fn cmpeq(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmpeq_ps(self, other) }
    }

    #[inline(always)]
    fn cmpne(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmpneq_ps(self, other) }
    }

    #[inline(always)]
    fn cmpge(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmpge_ps(self, other) }
    }

    #[inline(always)]
    fn cmpgt(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmpgt_ps(self, other) }
    }

    #[inline(always)]
    fn cmple(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmple_ps(self, other) }
    }

    #[inline(always)]
    fn cmplt(self, other: Self) -> Self::Mask {
        unsafe { _mm_cmplt_ps(self, other) }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        unsafe { _mm_add_ps(self, other) }
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        unsafe { _mm_div_ps(self, other) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        unsafe { _mm_mul_ps(self, other) }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            _mm_fmadd_ps(a, b, c)
        }

        #[cfg(not(target_feature = "fma"))]
        unsafe {
            _mm_add_ps(_mm_mul_ps(self, a), b)
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        unsafe { _mm_sub_ps(self, other) }
    }

    #[inline(always)]
    fn mul_scalar(self, other: f32) -> Self {
        unsafe { _mm_mul_ps(self, _mm_set_ps1(other)) }
    }

    #[inline(always)]
    fn div_scalar(self, other: f32) -> Self {
        unsafe { _mm_div_ps(self, _mm_set_ps1(other)) }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { _mm_min_ps(self, other) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { _mm_max_ps(self, other) }
    }
}

impl Vector3<f32> for __m128 {
    #[inline(always)]
    fn new(x: f32, y: f32, z: f32) -> Self {
        unsafe { _mm_set_ps(0.0, z, y, x) }
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[f32]) -> Self {
        Vector3::new(slice[0], slice[1], slice[2])
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        let xyz = Vector3::deref(&self);
        slice[0] = xyz.x;
        slice[1] = xyz.y;
        slice[2] = xyz.z;
    }

    #[inline(always)]
    fn deref(&self) -> &XYZ<f32> {
        unsafe { &*(self as *const Self as *const XYZ<f32>) }
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZ<f32> {
        unsafe { &mut *(self as *mut Self as *mut XYZ<f32>) }
    }

    #[inline]
    fn into_xy(self) -> XY<f32> {
        let mut out: MaybeUninit<Align16<XY<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline]
    fn into_xyzw(self, w: f32) -> XYZW<f32> {
        unsafe {
            let mut t = _mm_move_ss(self, _mm_set_ss(w));
            t = _mm_shuffle_ps(t, t, 0b00_10_01_00);
            // TODO: need a SIMD path
            *Vector4::deref(&_mm_move_ss(t, self))
        }
    }

    #[inline(always)]
    fn from_array(a: [f32; 3]) -> Self {
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }

    #[inline]
    fn into_array(self) -> [f32; 3] {
        let mut out: MaybeUninit<Align16<[f32; 3]>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline(always)]
    fn from_tuple(t: (f32, f32, f32)) -> Self {
        unsafe { _mm_set_ps(0.0, t.2, t.1, t.0) }
    }

    #[inline]
    fn into_tuple(self) -> (f32, f32, f32) {
        let mut out: MaybeUninit<Align16<(f32, f32, f32)>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline]
    fn min_element(self) -> f32 {
        unsafe {
            let v = self;
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b01_01_10_10));
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    fn max_element(self) -> f32 {
        unsafe {
            let v = self;
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_10_10));
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    fn dot(self, other: Self) -> f32 {
        unsafe { _mm_cvtss_f32(dot3_in_x(self, other)) }
    }

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        unsafe {
            let dot_in_x = dot3_in_x(self, other);
            _mm_shuffle_ps(dot_in_x, dot_in_x, 0b00_00_00_00)
        }
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        unsafe {
            // x  <-  a.y*b.z - a.z*b.y
            // y  <-  a.z*b.x - a.x*b.z
            // z  <-  a.x*b.y - a.y*b.x
            // We can save a shuffle by grouping it in this wacky order:
            // (self.zxy() * other - self * other.zxy()).zxy()
            let lhszxy = _mm_shuffle_ps(self, self, 0b01_01_00_10);
            let rhszxy = _mm_shuffle_ps(other, other, 0b01_01_00_10);
            let lhszxy_rhs = _mm_mul_ps(lhszxy, other);
            let rhszxy_lhs = _mm_mul_ps(rhszxy, self);
            let sub = _mm_sub_ps(lhszxy_rhs, rhszxy_lhs);
            _mm_shuffle_ps(sub, sub, 0b01_01_00_10)
        }
    }
}

impl Vector4<f32> for __m128 {
    #[inline(always)]
    fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe { _mm_set_ps(w, z, y, x) }
    }

    #[inline(always)]
    fn splat_x(self) -> Self {
        unsafe { _mm_shuffle_ps(self, self, 0b00_00_00_00) }
    }

    #[inline(always)]
    fn splat_y(self) -> Self {
        unsafe { _mm_shuffle_ps(self, self, 0b01_01_01_01) }
    }

    #[inline(always)]
    fn splat_z(self) -> Self {
        unsafe { _mm_shuffle_ps(self, self, 0b10_10_10_10) }
    }

    #[inline(always)]
    fn splat_w(self) -> Self {
        unsafe { _mm_shuffle_ps(self, self, 0b11_11_11_11) }
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4);
        unsafe { _mm_loadu_ps(slice.as_ptr()) }
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        unsafe {
            assert!(slice.len() >= 4);
            _mm_storeu_ps(slice.as_mut_ptr(), self);
        }
    }

    #[inline(always)]
    fn deref(&self) -> &XYZW<f32> {
        unsafe { &*(self as *const Self as *const XYZW<f32>) }
    }

    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZW<f32> {
        unsafe { &mut *(self as *mut Self as *mut XYZW<f32>) }
    }

    #[inline]
    fn into_xy(self) -> XY<f32> {
        let mut out: MaybeUninit<Align16<XY<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline]
    fn into_xyz(self) -> XYZ<f32> {
        let mut out: MaybeUninit<Align16<XYZ<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline(always)]
    fn from_array(a: [f32; 4]) -> Self {
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }

    #[inline]
    fn into_array(self) -> [f32; 4] {
        let mut out: MaybeUninit<Align16<[f32; 4]>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline(always)]
    fn from_tuple(t: (f32, f32, f32, f32)) -> Self {
        unsafe { _mm_set_ps(t.3, t.2, t.1, t.0) }
    }

    #[inline]
    fn into_tuple(self) -> (f32, f32, f32, f32) {
        let mut out: MaybeUninit<Align16<(f32, f32, f32, f32)>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, self);
            out.assume_init().0
        }
    }

    #[inline]
    fn min_element(self) -> f32 {
        unsafe {
            let v = self;
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b00_00_11_10));
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    fn max_element(self) -> f32 {
        unsafe {
            let v = self;
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_11_10));
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    fn dot(self, other: Self) -> f32 {
        unsafe { _mm_cvtss_f32(dot4_in_x(self, other)) }
    }

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        unsafe {
            let dot_in_x = dot4_in_x(self, other);
            _mm_shuffle_ps(dot_in_x, dot_in_x, 0b00_00_00_00)
        }
    }
}

impl FloatVector<f32> for __m128 {
    #[inline]
    fn abs(self) -> Self {
        unsafe { _mm_and_ps(self, _mm_castsi128_ps(_mm_set1_epi32(0x7f_ff_ff_ff))) }
    }

    #[inline]
    fn round(self) -> Self {
        unsafe {
            // Based on https://github.com/microsoft/DirectXMath `XMVectorRound`
            let sign = _mm_and_ps(self, PS_SIGN_MASK.m128);
            let s_magic = _mm_or_ps(PS_NO_FRACTION.m128, sign);
            let r1 = _mm_add_ps(self, s_magic);
            let r1 = _mm_sub_ps(r1, s_magic);
            let r2 = _mm_and_ps(self, PS_INV_SIGN_MASK.m128);
            let mask = _mm_cmple_ps(r2, PS_NO_FRACTION.m128);
            let r2 = _mm_andnot_ps(mask, self);
            let r1 = _mm_and_ps(r1, mask);
            _mm_xor_ps(r1, r2)
        }
    }

    #[inline]
    fn floor(self) -> Self {
        unsafe {
            // Based on https://github.com/microsoft/DirectXMath `XMVectorFloor`
            // To handle NAN, INF and numbers greater than 8388608, use masking
            let test = _mm_and_si128(_mm_castps_si128(self), PS_INV_SIGN_MASK.m128i);
            let test = _mm_cmplt_epi32(test, PS_NO_FRACTION.m128i);
            // Truncate
            let vint = _mm_cvttps_epi32(self);
            let result = _mm_cvtepi32_ps(vint);
            let larger = _mm_cmpgt_ps(result, self);
            // 0 -> 0, 0xffffffff -> -1.0f
            let larger = _mm_cvtepi32_ps(_mm_castps_si128(larger));
            let result = _mm_add_ps(result, larger);
            // All numbers less than 8388608 will use the round to int
            let result = _mm_and_ps(result, _mm_castsi128_ps(test));
            // All others, use the ORIGINAL value
            let test = _mm_andnot_si128(test, _mm_castps_si128(self));
            _mm_or_ps(result, _mm_castsi128_ps(test))
        }
    }

    #[inline]
    fn ceil(self) -> Self {
        unsafe {
            // Based on https://github.com/microsoft/DirectXMath `XMVectorCeil`
            // To handle NAN, INF and numbers greater than 8388608, use masking
            let test = _mm_and_si128(_mm_castps_si128(self), PS_INV_SIGN_MASK.m128i);
            let test = _mm_cmplt_epi32(test, PS_NO_FRACTION.m128i);
            // Truncate
            let vint = _mm_cvttps_epi32(self);
            let result = _mm_cvtepi32_ps(vint);
            let smaller = _mm_cmplt_ps(result, self);
            // 0 -> 0, 0xffffffff -> -1.0f
            let smaller = _mm_cvtepi32_ps(_mm_castps_si128(smaller));
            let result = _mm_sub_ps(result, smaller);
            // All numbers less than 8388608 will use the round to int
            let result = _mm_and_ps(result, _mm_castsi128_ps(test));
            // All others, use the ORIGINAL value
            let test = _mm_andnot_si128(test, _mm_castps_si128(self));
            _mm_or_ps(result, _mm_castsi128_ps(test))
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        unsafe { _mm_div_ps(Self::ONE, self) }
    }

    #[inline]
    fn signum(self) -> Self {
        const NEG_ONE: __m128 = const_m128!([-1.0; 4]);
        let mask = self.cmpge(Self::ZERO);
        let result = Self::select(mask, Self::ONE, NEG_ONE);
        let mask = unsafe { _mm_cmpunord_ps(self, self) };
        Self::select(mask, self, result)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { _mm_sub_ps(Self::ZERO, self) }
    }
}

impl FloatVector3<f32> for __m128 {
    #[inline(always)]
    fn is_nan_mask(self) -> Self::Mask {
        unsafe { _mm_cmpunord_ps(self, self) }
    }

    #[inline]
    fn length(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self, self);
            _mm_cvtss_f32(_mm_sqrt_ps(dot))
        }
    }

    #[inline]
    fn length_recip(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self, self);
            // _mm_rsqrt_ps is lower precision
            _mm_cvtss_f32(_mm_div_ps(Self::ONE, _mm_sqrt_ps(dot)))
        }
    }

    #[inline]
    fn normalize(self) -> Self {
        unsafe {
            let dot = Vector3::dot_into_vec(self, self);
            _mm_div_ps(self, _mm_sqrt_ps(dot))
        }
    }
}

impl FloatVector4<f32> for __m128 {
    #[inline(always)]
    fn is_nan_mask(self) -> Self::Mask {
        unsafe { _mm_cmpunord_ps(self, self) }
    }

    #[inline]
    fn length(self) -> f32 {
        unsafe {
            let dot = dot4_in_x(self, self);
            _mm_cvtss_f32(_mm_sqrt_ps(dot))
        }
    }

    #[inline]
    fn length_recip(self) -> f32 {
        unsafe {
            let dot = dot4_in_x(self, self);
            // _mm_rsqrt_ps is lower precision
            _mm_cvtss_f32(_mm_div_ps(Self::ONE, _mm_sqrt_ps(dot)))
        }
    }

    #[inline]
    fn normalize(self) -> Self {
        unsafe {
            let dot = Vector4::dot_into_vec(self, self);
            _mm_div_ps(self, _mm_sqrt_ps(dot))
        }
    }
}

impl Quaternion<f32> for __m128 {
    #[inline(always)]
    fn conjugate(self) -> Self {
        const SIGN: __m128 = const_m128!([-0.0, -0.0, -0.0, 0.0]);
        unsafe { _mm_xor_ps(self, SIGN) }
    }

    fn lerp(self, end: Self, s: f32) -> Self {
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        unsafe {
            const NEG_ZERO: __m128 = const_m128!([-0.0; 4]);
            let start = self;
            let end = end;
            let dot = Vector4::dot_into_vec(start, end);
            // Calculate the bias, if the dot product is positive or zero, there is no bias
            // but if it is negative, we want to flip the 'end' rotation XYZW components
            let bias = _mm_and_ps(dot, NEG_ZERO);
            let interpolated = _mm_add_ps(
                _mm_mul_ps(_mm_sub_ps(_mm_xor_ps(end, bias), start), _mm_set_ps1(s)),
                start,
            );
            FloatVector4::normalize(interpolated)
        }
    }

    fn slerp(self, end: Self, s: f32) -> Self {
        // http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        const DOT_THRESHOLD: f32 = 0.9995;

        let dot = Vector4::dot(self, end);

        if dot > DOT_THRESHOLD {
            // assumes lerp returns a normalized quaternion
            self.lerp(end, s)
        } else {
            // assumes scalar_acos clamps the input to [-1.0, 1.0]
            let theta = dot.acos_approx();

            let x = 1.0 - s;
            let y = s;
            let z = 1.0;

            unsafe {
                let tmp = _mm_mul_ps(_mm_set_ps1(theta), _mm_set_ps(0.0, z, y, x));
                let tmp = m128_sin(tmp);

                let scale1 = _mm_shuffle_ps(tmp, tmp, 0b00_00_00_00);
                let scale2 = _mm_shuffle_ps(tmp, tmp, 0b01_01_01_01);
                let theta_sin = _mm_shuffle_ps(tmp, tmp, 0b10_10_10_10);

                let theta_sin_recip = _mm_rcp_ps(theta_sin);

                self.mul(scale1).add(end.mul(scale2)).mul(theta_sin_recip)
            }
        }
    }

    fn mul_quaternion(self, other: Self) -> Self {
        unsafe {
            // Based on https://github.com/nfrechette/rtm `rtm::quat_mul`
            let lhs = self;
            let rhs = other;

            const CONTROL_WZYX: __m128 = const_m128!([1.0, -1.0, 1.0, -1.0]);
            const CONTROL_ZWXY: __m128 = const_m128!([1.0, 1.0, -1.0, -1.0]);
            const CONTROL_YXWZ: __m128 = const_m128!([-1.0, 1.0, 1.0, -1.0]);

            let r_xxxx = _mm_shuffle_ps(lhs, lhs, 0b00_00_00_00);
            let r_yyyy = _mm_shuffle_ps(lhs, lhs, 0b01_01_01_01);
            let r_zzzz = _mm_shuffle_ps(lhs, lhs, 0b10_10_10_10);
            let r_wwww = _mm_shuffle_ps(lhs, lhs, 0b11_11_11_11);

            let lxrw_lyrw_lzrw_lwrw = _mm_mul_ps(r_wwww, rhs);
            let l_wzyx = _mm_shuffle_ps(rhs, rhs, 0b00_01_10_11);

            let lwrx_lzrx_lyrx_lxrx = _mm_mul_ps(r_xxxx, l_wzyx);
            let l_zwxy = _mm_shuffle_ps(l_wzyx, l_wzyx, 0b10_11_00_01);

            let lwrx_nlzrx_lyrx_nlxrx = _mm_mul_ps(lwrx_lzrx_lyrx_lxrx, CONTROL_WZYX);

            let lzry_lwry_lxry_lyry = _mm_mul_ps(r_yyyy, l_zwxy);
            let l_yxwz = _mm_shuffle_ps(l_zwxy, l_zwxy, 0b00_01_10_11);

            let lzry_lwry_nlxry_nlyry = _mm_mul_ps(lzry_lwry_lxry_lyry, CONTROL_ZWXY);

            let lyrz_lxrz_lwrz_lzrz = _mm_mul_ps(r_zzzz, l_yxwz);
            let result0 = _mm_add_ps(lxrw_lyrw_lzrw_lwrw, lwrx_nlzrx_lyrx_nlxrx);

            let nlyrz_lxrz_lwrz_wlzrz = _mm_mul_ps(lyrz_lxrz_lwrz_lzrz, CONTROL_YXWZ);
            let result1 = _mm_add_ps(lzry_lwry_nlxry_nlyry, nlyrz_lxrz_lwrz_wlzrz);
            _mm_add_ps(result0, result1)
        }
    }

    fn mul_vector3(self, other: XYZ<f32>) -> XYZ<f32> {
        glam_assert!(FloatVector4::is_normalized(self));
        unsafe {
            const TWO: __m128 = const_m128!([2.0; 4]);
            let other = _mm_set_ps(0.0, other.z, other.y, other.x);
            let w = _mm_shuffle_ps(self, self, 0b11_11_11_11);
            let b = self;
            let b2 = Vector3::dot_into_vec(b, b);
            other
                .mul(w.mul(w).sub(b2))
                .add(b.mul(Vector3::dot_into_vec(other, b).mul(TWO)))
                .add(b.cross(other).mul(w.mul(TWO)))
                .into()
        }
    }
}

impl From<__m128> for XYZW<f32> {
    #[inline]
    fn from(v: __m128) -> XYZW<f32> {
        let mut out: MaybeUninit<Align16<XYZW<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, v);
            out.assume_init().0
        }
    }
}

impl From<__m128> for XYZ<f32> {
    #[inline]
    fn from(v: __m128) -> XYZ<f32> {
        let mut out: MaybeUninit<Align16<XYZ<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, v);
            out.assume_init().0
        }
    }
}

impl From<__m128> for XY<f32> {
    #[inline]
    fn from(v: __m128) -> XY<f32> {
        let mut out: MaybeUninit<Align16<XY<f32>>> = MaybeUninit::uninit();
        unsafe {
            _mm_store_ps(out.as_mut_ptr() as *mut f32, v);
            out.assume_init().0
        }
    }
}

unsafe fn m128_neg_mul_sub(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_sub_ps(c, _mm_mul_ps(a, b))
}

/// Returns a vector whose components are the corresponding components of Angles modulo 2PI.
#[inline]
unsafe fn m128_mod_angles(angles: __m128) -> __m128 {
    // Based on https://github.com/microsoft/DirectXMath `XMVectorModAngles`
    let v = _mm_mul_ps(angles, PS_RECIPROCAL_TWO_PI.m128);
    let v = v.round();
    m128_neg_mul_sub(PS_TWO_PI.m128, v, angles)
}

/// Computes the sine of the angle in each lane of `v`. Values outside
/// the bounds of PI may produce an increasing error as the input angle
/// drifts from `[-PI, PI]`.
#[inline]
unsafe fn m128_sin(v: __m128) -> __m128 {
    // Based on https://github.com/microsoft/DirectXMath `XMVectorSin`

    // 11-degree minimax approximation

    // Force the value within the bounds of pi
    let mut x = m128_mod_angles(v);

    // Map in [-pi/2,pi/2] with sin(y) = sin(x).
    let sign = _mm_and_ps(x, PS_NEGATIVE_ZERO.m128);
    // pi when x >= 0, -pi when x < 0
    let c = _mm_or_ps(PS_PI.m128, sign);
    // |x|
    let absx = _mm_andnot_ps(sign, x);
    let rflx = _mm_sub_ps(c, x);
    let comp = _mm_cmple_ps(absx, PS_HALF_PI.m128);
    let select0 = _mm_and_ps(comp, x);
    let select1 = _mm_andnot_ps(comp, rflx);
    x = _mm_or_ps(select0, select1);

    let x2 = _mm_mul_ps(x, x);

    // Compute polynomial approximation
    const SC1: __m128 = unsafe { PS_SIN_COEFFICIENTS1.m128 };
    let v_constants_b = _mm_shuffle_ps(SC1, SC1, 0b00_00_00_00);

    const SC0: __m128 = unsafe { PS_SIN_COEFFICIENTS0.m128 };
    let mut v_constants = _mm_shuffle_ps(SC0, SC0, 0b11_11_11_11);
    let mut result = v_constants_b.mul_add(x2, v_constants);

    v_constants = _mm_shuffle_ps(SC0, SC0, 0b10_10_10_10);
    result = result.mul_add(x2, v_constants);

    v_constants = _mm_shuffle_ps(SC0, SC0, 0b01_01_01_01);
    result = result.mul_add(x2, v_constants);

    v_constants = _mm_shuffle_ps(SC0, SC0, 0b00_00_00_00);
    result = result.mul_add(x2, v_constants);

    result = result.mul_add(x2, PS_ONE.m128);
    result = _mm_mul_ps(result, x);

    result
}

// Based on http://gruntthepeon.free.fr/ssemath/sse_mathfun.h
// #[cfg(target_feature = "sse2")]
// unsafe fn sin_cos_sse2(x: __m128) -> (__m128, __m128) {
//     let mut sign_bit_sin = x;
//     // take the absolute value
//     let mut x = _mm_and_ps(x, PS_INV_SIGN_MASK.m128);
//     // extract the sign bit (upper one)
//     sign_bit_sin = _mm_and_ps(sign_bit_sin, PS_SIGN_MASK.m128);

//     // scale by 4/Pi
//     let mut y = _mm_mul_ps(x, PS_CEPHES_FOPI.m128);

//     // store the integer part of y in emm2
//     let mut emm2 = _mm_cvttps_epi32(y);

//     // j=(j+1) & (~1) (see the cephes sources)
//     emm2 = _mm_add_epi32(emm2, PI32_1.m128i);
//     emm2 = _mm_and_si128(emm2, PI32_INV_1.m128i);
//     y = _mm_cvtepi32_ps(emm2);

//     let mut emm4 = emm2;

//     /* get the swap sign flag for the sine */
//     let mut emm0 = _mm_and_si128(emm2, PI32_4.m128i);
//     emm0 = _mm_slli_epi32(emm0, 29);
//     let swap_sign_bit_sin = _mm_castsi128_ps(emm0);

//     /* get the polynom selection mask for the sine*/
//     emm2 = _mm_and_si128(emm2, PI32_2.m128i);
//     emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
//     let poly_mask = _mm_castsi128_ps(emm2);

//     /* The magic pass: "Extended precision modular arithmetic"
//     x = ((x - y * DP1) - y * DP2) - y * DP3; */
//     let mut xmm1 = PS_MINUS_CEPHES_DP1.m128;
//     let mut xmm2 = PS_MINUS_CEPHES_DP2.m128;
//     let mut xmm3 = PS_MINUS_CEPHES_DP3.m128;
//     xmm1 = _mm_mul_ps(y, xmm1);
//     xmm2 = _mm_mul_ps(y, xmm2);
//     xmm3 = _mm_mul_ps(y, xmm3);
//     x = _mm_add_ps(x, xmm1);
//     x = _mm_add_ps(x, xmm2);
//     x = _mm_add_ps(x, xmm3);

//     emm4 = _mm_sub_epi32(emm4, PI32_2.m128i);
//     emm4 = _mm_andnot_si128(emm4, PI32_4.m128i);
//     emm4 = _mm_slli_epi32(emm4, 29);
//     let sign_bit_cos = _mm_castsi128_ps(emm4);

//     sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

//     // Evaluate the first polynom  (0 <= x <= Pi/4)
//     let z = _mm_mul_ps(x, x);
//     y = PS_COSCOF_P0.m128;

//     y = _mm_mul_ps(y, z);
//     y = _mm_add_ps(y, PS_COSCOF_P1.m128);
//     y = _mm_mul_ps(y, z);
//     y = _mm_add_ps(y, PS_COSCOF_P2.m128);
//     y = _mm_mul_ps(y, z);
//     y = _mm_mul_ps(y, z);
//     let tmp = _mm_mul_ps(z, PS_0_5.m128);
//     y = _mm_sub_ps(y, tmp);
//     y = _mm_add_ps(y, PS_1_0.m128);

//     // Evaluate the second polynom  (Pi/4 <= x <= 0)
//     let mut y2 = PS_SINCOF_P0.m128;
//     y2 = _mm_mul_ps(y2, z);
//     y2 = _mm_add_ps(y2, PS_SINCOF_P1.m128);
//     y2 = _mm_mul_ps(y2, z);
//     y2 = _mm_add_ps(y2, PS_SINCOF_P2.m128);
//     y2 = _mm_mul_ps(y2, z);
//     y2 = _mm_mul_ps(y2, x);
//     y2 = _mm_add_ps(y2, x);

//     // select the correct result from the two polynoms
//     xmm3 = poly_mask;
//     let ysin2 = _mm_and_ps(xmm3, y2);
//     let ysin1 = _mm_andnot_ps(xmm3, y);
//     y2 = _mm_sub_ps(y2, ysin2);
//     y = _mm_sub_ps(y, ysin1);

//     xmm1 = _mm_add_ps(ysin1, ysin2);
//     xmm2 = _mm_add_ps(y, y2);

//     // update the sign
//     (
//         _mm_xor_ps(xmm1, sign_bit_sin),
//         _mm_xor_ps(xmm2, sign_bit_cos),
//     )
// }

#[test]
fn test_sse2_m128_sin() {
    use core::f32::consts::PI;

    fn test_sse2_m128_sin_angle(a: f32) {
        let v = unsafe { m128_sin(_mm_set_ps1(a)) };
        let v = Vector4::deref(&v);
        let a_sin = a.sin();
        // dbg!((a, a_sin, v));
        assert!(v.abs_diff_eq(Vector::splat(a_sin), 1e-6));
    }

    let mut a = -PI;
    let end = PI;
    let step = PI / 8192.0;

    while a <= end {
        test_sse2_m128_sin_angle(a);
        a += step;
    }
}
