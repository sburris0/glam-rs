pub use num_traits::{Float, Num};

// use crate::XYZ;
// use crate::XYZW;
use core::{
    marker::Sized,
    ops::{Add, Div, Mul, Sub},
};

pub trait MaskConst: Sized {
    const MASK: [Self; 2];
}

pub trait NumConstEx: Sized {
    const ZERO: Self;
    const ONE: Self;
}

pub trait FloatConstEx: Sized {
    const NEG_ONE: Self;
    const TWO: Self;
    const HALF: Self;
}

pub trait NumEx:
    Num
    + NumConstEx
    + Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
{
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

pub trait FloatEx: Float + FloatConstEx + NumEx {
    // TODO: Move to Signed
    // fn abs(self) -> Self;
    // fn ceil(self) -> Self;
    // fn floor(self) -> Self;
    // fn is_finite(self) -> bool;
    // fn is_nan(self) -> bool;
    // fn recip(self) -> Self;
    // fn round(self) -> Self;
    // fn signum(self) -> Self;
    // fn sqrt(self) -> Self;
    fn acos_approx(self) -> Self;
    // fn sin(self) -> Self;
    // fn sin_cos(self) -> (Self, Self);
    fn from_f32(f: f32) -> Self;
    fn from_f64(f: f64) -> Self;
}

impl MaskConst for u32 {
    const MASK: [u32; 2] = [0, 0xff_ff_ff_ff];
}

impl MaskConst for u64 {
    const MASK: [u64; 2] = [0, 0xff_ff_ff_ff_ff_ff_ff_ff];
}

impl NumConstEx for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl FloatConstEx for f32 {
    const NEG_ONE: Self = -1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
}

impl NumEx for f32 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl FloatEx for f32 {
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
    // #[inline(always)]
    // fn abs(self) -> Self {
    //     Self::abs(self)
    // }
    // #[inline(always)]
    // fn ceil(self) -> Self {
    //     Self::ceil(self)
    // }
    // #[inline(always)]
    // fn floor(self) -> Self {
    //     Self::floor(self)
    // }
    // #[inline(always)]
    // fn is_finite(self) -> bool {
    //     Self::is_finite(self)
    // }
    // #[inline(always)]
    // fn is_nan(self) -> bool {
    //     Self::is_nan(self)
    // }
    // #[inline(always)]
    // fn recip(self) -> Self {
    //     Self::recip(self)
    // }
    // #[inline(always)]
    // fn round(self) -> Self {
    //     Self::round(self)
    // }
    // #[inline(always)]
    // fn sin(self) -> Self {
    //     Self::sin(self)
    // }
    // #[inline(always)]
    // fn sin_cos(self) -> (Self, Self) {
    //     Self::sin_cos(self)
    // }
    // #[inline(always)]
    // fn signum(self) -> Self {
    //     Self::signum(self)
    // }
    // #[inline(always)]
    // fn sqrt(self) -> Self {
    //     Self::sqrt(self)
    // }
    #[inline(always)]
    fn acos_approx(self) -> Self {
        // Based on https://github.com/microsoft/DirectXMath `XMScalarAcos`
        // Clamp input to [-1,1].
        let nonnegative = self >= 0.0;
        let x = self.abs();
        let mut omx = 1.0 - x;
        if omx < 0.0 {
            omx = 0.0;
        }
        let root = omx.sqrt();

        // 7-degree minimax approximation
        #[allow(clippy::approx_constant)]
        let mut result = ((((((-0.001_262_491_1 * x + 0.006_670_09) * x - 0.017_088_126) * x
            + 0.030_891_88)
            * x
            - 0.050_174_303)
            * x
            + 0.088_978_99)
            * x
            - 0.214_598_8)
            * x
            + 1.570_796_3;
        result *= root;

        // acos(x) = pi - acos(-x) when x < 0
        if nonnegative {
            result
        } else {
            core::f32::consts::PI - result
        }
    }
}

impl NumConstEx for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl FloatConstEx for f64 {
    const NEG_ONE: Self = -1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
}

impl NumEx for f64 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

impl FloatEx for f64 {
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as Self
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    // #[inline(always)]
    // fn abs(self) -> Self {
    //     Self::abs(self)
    // }
    // #[inline(always)]
    // fn ceil(self) -> Self {
    //     Self::ceil(self)
    // }
    // #[inline(always)]
    // fn floor(self) -> Self {
    //     Self::floor(self)
    // }
    // #[inline(always)]
    // fn is_finite(self) -> bool {
    //     Self::is_finite(self)
    // }
    // #[inline(always)]
    // fn is_nan(self) -> bool {
    //     Self::is_nan(self)
    // }
    // #[inline(always)]
    // fn recip(self) -> Self {
    //     Self::recip(self)
    // }
    // #[inline(always)]
    // fn round(self) -> Self {
    //     Self::round(self)
    // }
    // #[inline(always)]
    // fn sin(self) -> Self {
    //     Self::sin(self)
    // }
    // #[inline(always)]
    // fn sin_cos(self) -> (Self, Self) {
    //     Self::sin_cos(self)
    // }
    // #[inline(always)]
    // fn signum(self) -> Self {
    //     Self::signum(self)
    // }
    // #[inline(always)]
    // fn sqrt(self) -> Self {
    //     Self::sqrt(self)
    // }
    #[inline(always)]
    fn acos_approx(self) -> Self {
        // TODO: clamp range
        Self::acos(self)
    }
}

/*
impl<T: Num> Add for XYZW<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        XYZW {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl<T: Num> Div for XYZW<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        XYZW {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
            w: self.w / other.w,
        }
    }
}

impl<T: Num> Div<T> for XYZW<T> {
    type Output = Self;
    fn div(self, other: T) -> Self::Output {
        XYZW {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

impl<T: Num> Mul for XYZW<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        XYZW {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

impl<T: Num> Mul<T> for XYZW<T> {
    type Output = Self;
    fn mul(self, other: T) -> Self::Output {
        XYZW {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

impl<T: Num> Sub for XYZW<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        XYZW {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}
*/

#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        assert_approx_eq!($a, $b, core::f32::EPSILON);
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        assert!(
            (a - b).abs() <= eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (a - b).abs()
        );
    }};
}

#[cfg(test)]
macro_rules! assert_relative_eq {
    ($a:expr, $b:expr) => {{
        assert_relative_eq!($a, $b, core::f32::EPSILON);
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        let diff = (a - b).abs();
        let largest = a.abs().max(b.abs());
        assert!(
            diff <= largest * eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            largest * eps,
            diff
        );
    }};
}

#[test]
fn test_scalar_acos() {
    fn test_scalar_acos_angle(a: f32) {
        // 1e-6 is the lowest epsilon that will pass
        assert_relative_eq!(a.acos_approx(), a.acos(), 1e-6);
        // assert_approx_eq!(scalar_acos(a), a.acos(), 1e-6);
    }

    // test 1024 floats between -1.0 and 1.0 inclusive
    const MAX_TESTS: u32 = 1024 / 2;
    const SIGN: u32 = 0x80_00_00_00;
    const PTVE_ONE: u32 = 0x3f_80_00_00; // 1.0_f32.to_bits();
    const NGVE_ONE: u32 = SIGN | PTVE_ONE;
    const STEP_SIZE: usize = (PTVE_ONE / MAX_TESTS) as usize;
    for f in (SIGN..=NGVE_ONE)
        .step_by(STEP_SIZE)
        .map(|i| f32::from_bits(i))
    {
        test_scalar_acos_angle(f);
    }
    for f in (0..=PTVE_ONE).step_by(STEP_SIZE).map(|i| f32::from_bits(i)) {
        test_scalar_acos_angle(f);
    }

    // input is clamped to -1.0..1.0
    assert_approx_eq!(2.0.acos_approx(), 0.0);
    assert_approx_eq!((-2.0).acos_approx(), core::f32::consts::PI);
}
