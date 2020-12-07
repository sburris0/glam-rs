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
        unsafe { _mm_add_ps(_mm_mul_ps(self, a), b) }
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
            use crate::f32::funcs::sse2::m128_round;
            m128_round(self)
        }
    }

    #[inline]
    fn floor(self) -> Self {
        unsafe {
            use crate::f32::funcs::sse2::m128_floor;
            m128_floor(self)
        }
    }

    #[inline]
    fn ceil(self) -> Self {
        unsafe {
            use crate::f32::funcs::sse2::m128_ceil;
            m128_ceil(self)
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
                let tmp = crate::f32::funcs::sse2::m128_sin(tmp);

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
