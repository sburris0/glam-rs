use super::Vec3;
use crate::Align16;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::{cmp::Ordering, f32, fmt, mem, ops::*};

pub(crate) const X_AXIS: Align16<(f32, f32, f32, f32)> = Align16((1.0, 0.0, 0.0, 0.0));
pub(crate) const Y_AXIS: Align16<(f32, f32, f32, f32)> = Align16((0.0, 1.0, 0.0, 0.0));
pub(crate) const Z_AXIS: Align16<(f32, f32, f32, f32)> = Align16((0.0, 0.0, 1.0, 0.0));
pub(crate) const W_AXIS: Align16<(f32, f32, f32, f32)> = Align16((0.0, 0.0, 0.0, 1.0));

#[derive(Clone, Copy)]
// if compiling with simd enabled assume alignment needs to match the simd type
#[cfg_attr(not(feature = "scalar-math"), repr(align(16)))]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl fmt::Debug for Vec4 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("Vec4")
            .field(&self.x)
            .field(&self.y)
            .field(&self.z)
            .field(&self.w)
            .finish()
    }
}

#[inline]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

impl Vec4 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn zero() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_set1_ps(0.0).into()
        }
        #[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn one() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_set1_ps(1.0).into()
        }
        #[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    #[inline]
    pub fn unit_x() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_load_ps(&X_AXIS as *const Align16<(f32, f32, f32, f32)> as *const f32).into()
        }
        #[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    pub fn unit_y() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_load_ps(&Y_AXIS as *const Align16<(f32, f32, f32, f32)> as *const f32).into()
        }
    }

    #[inline]
    pub fn unit_z() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_load_ps(&Z_AXIS as *const Align16<(f32, f32, f32, f32)> as *const f32).into()
        }
    }

    #[inline]
    pub fn unit_w() -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_load_ps(&W_AXIS as *const Align16<(f32, f32, f32, f32)> as *const f32).into()
        }
    }

    #[inline]
    pub fn truncate(self) -> Vec3 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *(&self as *const Vec4 as *const Vec3)
        }
    }

    #[inline]
    pub fn splat(v: f32) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_set_ps1(v).into()
        }
    }

    #[inline]
    pub fn sign(self) -> Self {
        let mask = self.cmpge(Self::zero());
        mask.select(Self::splat(1.0), Self::splat(-1.0))
    }

    #[inline]
    pub(crate) fn dup_x(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_shuffle_ps(self.into(), self.into(), 0b00_00_00_00).into()
        }
    }

    #[inline]
    pub(crate) fn dup_y(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_shuffle_ps(self.into(), self.into(), 0b01_01_01_01).into()
        }
    }

    #[inline]
    pub(crate) fn dup_z(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_shuffle_ps(self.into(), self.into(), 0b10_10_10_10).into()
        }
    }

    #[inline]
    pub(crate) fn dup_w(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_shuffle_ps(self.into(), self.into(), 0b11_11_11_11).into()
        }
    }

    #[inline]
    fn dot_as_vec4(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let lhs = self.into();
            let rhs = rhs.into();
            let x2_y2_z2_w2 = _mm_mul_ps(lhs, rhs);
            let z2_w2_0_0 = _mm_shuffle_ps(x2_y2_z2_w2, x2_y2_z2_w2, 0b00_00_11_10);
            let x2z2_y2w2_0_0 = _mm_add_ps(x2_y2_z2_w2, z2_w2_0_0);
            let y2w2_0_0_0 = _mm_shuffle_ps(x2z2_y2w2_0_0, x2z2_y2w2_0_0, 0b00_00_00_01);
            let x2y2z2w2_0_0_0 = _mm_add_ps(x2z2_y2w2_0_0, y2w2_0_0_0);
            x2y2z2w2_0_0_0.into()
        }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        self.dot_as_vec4(rhs).x
    }

    #[inline]
    pub fn length(self) -> f32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let dot = self.dot_as_vec4(self);
            _mm_cvtss_f32(_mm_sqrt_ps(dot.into()))
        }
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length_reciprocal(self) -> f32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            // _mm_rsqrt_ps is lower precision
            let dot = self.dot_as_vec4(self);
            _mm_cvtss_f32(_mm_div_ps(_mm_set_ps1(1.0), _mm_sqrt_ps(dot.into()))).into()
        }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let dot = self.dot_as_vec4(self);
            _mm_div_ps(self.into(), _mm_sqrt_ps(dot.into())).into()
        }
    }

    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_min_ps(self.into(), rhs.into()).into()
        }
    }

    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_max_ps(self.into(), rhs.into()).into()
        }
    }

    #[inline]
    pub fn min_element(self) -> f32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let v = self.into();
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b00_00_11_10));
            let v = _mm_min_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    pub fn max_element(self) -> f32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let v = self.into();
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_11_10));
            let v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0b00_00_00_01));
            _mm_cvtss_f32(v)
        }
    }

    #[inline]
    pub fn cmpeq(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmpeq_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn cmpne(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmpneq_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn cmpge(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmpge_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn cmpgt(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmpgt_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn cmple(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmple_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn cmplt(self, rhs: Self) -> Vec4b {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            Vec4b(_mm_cmplt_ps(self.into(), rhs.into()))
        }
    }

    #[inline]
    pub fn from_slice_unaligned(slice: &[f32]) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            assert!(slice.len() >= 4);
            _mm_loadu_ps(slice.as_ptr()).into()
        }
    }

    #[inline]
    pub fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            assert!(slice.len() >= 4);
            _mm_storeu_ps(slice.as_mut_ptr(), self.into());
        }
    }

    #[inline]
    /// Per component multiplication/addition of the three inputs: b + (self * a)
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            (_mm_add_ps(_mm_mul_ps(self.into(), a.into()), b.into())).into()
        }
    }

    // #[inline]
    // /// Per component negative multiplication/subtraction of the three inputs `-((self * a) - b)`
    // /// This is mathematically equivalent to `b - (self * a)`
    // pub(crate) fn neg_mul_sub(self, a: Self, b: Self) -> Self {
    //     #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
    //     unsafe { _mm_sub_ps(b.into(), _mm_mul_ps(self.into(), a.into())).into() }
    // }

    #[inline]
    pub fn reciprocal(self) -> Self {
        // TODO: Optimize
        Self::one() / self
    }

    #[inline]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self + ((rhs - self) * s)
    }
}

impl AsRef<[f32; 4]> for Vec4 {
    #[inline]
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
}

impl AsMut<[f32; 4]> for Vec4 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl From<Vec4> for __m128 {
    #[inline]
    fn from(t: Vec4) -> Self {
        unsafe { _mm_load_ps(&t as *const Vec4 as *const f32) }
    }
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl From<&Vec4> for __m128 {
    #[inline]
    fn from(t: &Vec4) -> Self {
        unsafe { _mm_load_ps(t as *const Vec4 as *const f32) }
    }
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl From<&mut Vec4> for __m128 {
    #[inline]
    fn from(t: &mut Vec4) -> Self {
        unsafe { _mm_load_ps(t as *const Vec4 as *const f32) }
    }
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl From<__m128> for Vec4 {
    #[inline]
    fn from(t: __m128) -> Self {
        unsafe { *(&t as *const __m128 as *const Self) }
    }
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
impl From<&__m128> for Vec4 {
    #[inline]
    fn from(t: &__m128) -> Self {
        unsafe { *(t as *const __m128 as *const Self) }
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Div<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_div_ps(self.into(), rhs.into()).into()
        }
    }
}

impl DivAssign<Vec4> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_div_ps(self.into(), rhs.into()).into();
        }
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_div_ps(self.into(), _mm_set1_ps(rhs)).into()
        }
    }
}

impl DivAssign<f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_div_ps(self.into(), _mm_set1_ps(rhs)).into()
        }
    }
}

impl Mul<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_mul_ps(self.into(), rhs.into()).into()
        }
    }
}

impl MulAssign<Vec4> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_mul_ps(self.into(), rhs.into()).into();
        }
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_mul_ps(self.into(), _mm_set1_ps(rhs)).into()
        }
    }
}

impl MulAssign<f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_mul_ps(self.into(), _mm_set1_ps(rhs)).into()
        }
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_mul_ps(_mm_set1_ps(self.into()), rhs.into()).into()
        }
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_add_ps(self.into(), rhs.into()).into()
        }
    }
}

impl AddAssign for Vec4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_add_ps(self.into(), rhs.into()).into()
        }
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_sub_ps(self.into(), rhs.into()).into()
        }
    }
}

impl SubAssign for Vec4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            *self = _mm_sub_ps(self.into(), rhs.into()).into()
        }
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_sub_ps(_mm_set1_ps(0.0), self.into()).into()
        }
    }
}

impl Default for Vec4 {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.cmpeq(*rhs).all()
    }
}

impl PartialOrd for Vec4 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    #[inline]
    fn from(t: (f32, f32, f32, f32)) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_loadu_ps(&t as *const (f32, f32, f32, f32) as *const f32).into()
        }
    }
}

impl From<Vec4> for (f32, f32, f32, f32) {
    #[inline]
    fn from(v: Vec4) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let mut out: Align16<(f32, f32, f32, f32)> = mem::uninitialized();
            _mm_store_ps(
                &mut out.0 as *mut (f32, f32, f32, f32) as *mut f32,
                v.into(),
            );
            out.0
        }
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline]
    fn from(a: [f32; 4]) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_loadu_ps(a.as_ptr()).into()
        }
    }
}

impl From<Vec4> for [f32; 4] {
    #[inline]
    fn from(v: Vec4) -> Self {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            let mut out: Align16<[f32; 4]> = mem::uninitialized();
            _mm_store_ps(&mut out.0 as *mut [f32; 4] as *mut f32, v.into());
            out.0
        }
    }
}

#[cfg(feature = "rand")]
impl Distribution<Vec4> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec4 {
        rng.gen::<[f32; 4]>().into()
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "scalar-math"), repr(align(16)))]
#[repr(C)]
#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
pub struct Vec4b(__m128);
#[cfg(not(all(target_feature = "sse2", not(feature = "scalar-math"))))]
pub struct Vec4b(bool, bool, bool, bool);

impl Vec4b {
    #[inline]
    pub fn mask(self) -> u32 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            (_mm_movemask_ps(self.0) as u32)
        }
    }

    #[inline]
    pub fn any(self) -> bool {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_movemask_ps(self.0) != 0
        }
    }

    #[inline]
    pub fn all(self) -> bool {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_movemask_ps(self.0) == 0xf
        }
    }

    #[inline]
    pub fn select(self, if_true: Vec4, if_false: Vec4) -> Vec4 {
        #[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
        unsafe {
            _mm_or_ps(
                _mm_andnot_ps(self.0, if_false.into()),
                _mm_and_ps(if_true.into(), self.0),
            )
            .into()
        }
    }
}
