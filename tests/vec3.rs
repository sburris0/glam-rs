#[macro_use]
mod support;

macro_rules! impl_vec3_tests {
    ($new:ident, $vec3:ident, $mask:ident, $t:ident) => {
        #[test]
        fn test_new() {
            let v = $new(1.0, 2.0, 3.0);

            assert_eq!(v.x, 1.0);
            assert_eq!(v.y, 2.0);
            assert_eq!(v.z, 3.0);

            let t = (1.0, 2.0, 3.0);
            let v = $vec3::from(t);
            assert_eq!(t, v.into());

            let a = [1.0, 2.0, 3.0];
            let v = $vec3::from(a);
            let a1: [$t; 3] = v.into();
            assert_eq!(a, a1);

            let v = $vec3::new(t.0, t.1, t.2);
            assert_eq!(t, v.into());

            assert_eq!($vec3::new(1.0, 0.0, 0.0), $vec3::unit_x());
            assert_eq!($vec3::new(0.0, 1.0, 0.0), $vec3::unit_y());
            assert_eq!($vec3::new(0.0, 0.0, 1.0), $vec3::unit_z());
        }

        #[test]
        fn test_fmt() {
            let a = $vec3::new(1.0, 2.0, 3.0);
            assert_eq!(
                format!("{:?}", a),
                format!("{}(1.0, 2.0, 3.0)", stringify!($vec3))
            );
            // assert_eq!(format!("{:#?}", a), "$vec3(\n    1.0,\n    2.0,\n    3.0\n)");
            assert_eq!(format!("{}", a), "[1, 2, 3]");
        }

        #[test]
        fn test_zero() {
            let v = $vec3::zero();
            assert_eq!((0.0, 0.0, 0.0), v.into());
            assert_eq!(v, $vec3::default());
        }

        #[test]
        fn test_splat() {
            let v = $vec3::splat(1.0);
            assert_eq!((1.0, 1.0, 1.0), v.into());
        }

        #[test]
        fn test_accessors() {
            let mut a = $vec3::zero();
            a.x = 1.0;
            a.y = 2.0;
            a.z = 3.0;
            assert_eq!(1.0, a.x);
            assert_eq!(2.0, a.y);
            assert_eq!(3.0, a.z);
            assert_eq!((1.0, 2.0, 3.0), a.into());

            let mut a = $vec3::zero();
            a[0] = 1.0;
            a[1] = 2.0;
            a[2] = 3.0;
            assert_eq!(1.0, a[0]);
            assert_eq!(2.0, a[1]);
            assert_eq!(3.0, a[2]);
            assert_eq!((1.0, 2.0, 3.0), a.into());
        }

        #[test]
        fn test_funcs() {
            let x = $new(1.0, 0.0, 0.0);
            let y = $new(0.0, 1.0, 0.0);
            let z = $new(0.0, 0.0, 1.0);
            assert_eq!(1.0, x.dot(x));
            assert_eq!(0.0, x.dot(y));
            assert_eq!(-1.0, z.dot(-z));
            assert_eq!(y, z.cross(x));
            assert_eq!(z, x.cross(y));
            assert_eq!(4.0, (2.0 * x).length_squared());
            assert_eq!(9.0, (-3.0 * y).length_squared());
            assert_eq!(16.0, (4.0 * z).length_squared());
            assert_eq!(2.0, (-2.0 * x).length());
            assert_eq!(3.0, (3.0 * y).length());
            assert_eq!(4.0, (-4.0 * z).length());
            assert_eq!(2.0, x.distance_squared(y));
            assert_eq!(13.0, (2.0 * x).distance_squared(-3.0 * z));
            assert_eq!((2.0 as $t).sqrt(), x.distance(y));
            assert_eq!(5.0, (3.0 * x).distance(-4.0 * y));
            assert_eq!(13.0, (-5.0 * z).distance(12.0 * y));
            assert_eq!(x, (2.0 * x).normalize());
            assert_eq!(
                1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0,
                $new(1.0, 2.0, 3.0).dot($new(4.0, 5.0, 6.0))
            );
            assert_eq!(
                2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0,
                $new(2.0, 3.0, 4.0).length_squared()
            );
            assert_eq!(
                (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0).sqrt(),
                $new(2.0, 3.0, 4.0).length()
            );
            assert_eq!(
                1.0 / (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0).sqrt(),
                $new(2.0, 3.0, 4.0).length_recip()
            );
            assert!($new(2.0, 3.0, 4.0).normalize().is_normalized());
            assert_approx_eq!(
                $new(2.0, 3.0, 4.0) / (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0).sqrt(),
                $new(2.0, 3.0, 4.0).normalize()
            );
            assert_eq!($new(0.5, 0.25, 0.125), $new(2.0, 4.0, 8.0).recip());
        }

        #[test]
        fn test_ops() {
            let a = $new(1.0, 2.0, 3.0);
            assert_eq!((2.0, 4.0, 6.0), (a + a).into());
            assert_eq!((0.0, 0.0, 0.0), (a - a).into());
            assert_eq!((1.0, 4.0, 9.0), (a * a).into());
            assert_eq!((2.0, 4.0, 6.0), (a * 2.0).into());
            assert_eq!((2.0, 4.0, 6.0), (2.0 * a).into());
            assert_eq!((1.0, 1.0, 1.0), (a / a).into());
            assert_eq!((0.5, 1.0, 1.5), (a / 2.0).into());
            assert_eq!((2.0, 1.0, 2.0 / 3.0), (2.0 / a).into());
            assert_eq!((-1.0, -2.0, -3.0), (-a).into());
        }

        #[test]
        fn test_assign_ops() {
            let a = $new(1.0, 2.0, 3.0);
            let mut b = a;
            b += a;
            assert_eq!((2.0, 4.0, 6.0), b.into());
            b -= a;
            assert_eq!((1.0, 2.0, 3.0), b.into());
            b *= a;
            assert_eq!((1.0, 4.0, 9.0), b.into());
            b /= a;
            assert_eq!((1.0, 2.0, 3.0), b.into());
            b *= 2.0;
            assert_eq!((2.0, 4.0, 6.0), b.into());
            b /= 2.0;
            assert_eq!((1.0, 2.0, 3.0), b.into());
        }

        #[test]
        fn test_min_max() {
            let a = $new(-1.0, 2.0, -3.0);
            let b = $new(1.0, -2.0, 3.0);
            assert_eq!((-1.0, -2.0, -3.0), a.min(b).into());
            assert_eq!((-1.0, -2.0, -3.0), b.min(a).into());
            assert_eq!((1.0, 2.0, 3.0), a.max(b).into());
            assert_eq!((1.0, 2.0, 3.0), b.max(a).into());
        }

        #[test]
        fn test_hmin_hmax() {
            let a = $new(-1.0, 2.0, -3.0);
            assert_eq!(-3.0, a.min_element());
            assert_eq!(2.0, a.max_element());
        }

        #[test]
        fn test_eq() {
            let a = $new(1.0, 1.0, 1.0);
            let b = $new(1.0, 2.0, 3.0);
            assert!(a.cmpeq(a).all());
            assert!(b.cmpeq(b).all());
            assert!(a.cmpne(b).any());
            assert!(b.cmpne(a).any());
            assert!(b.cmpeq(a).any());
        }

        #[test]
        fn test_cmp() {
            assert!(!$mask::default().any());
            assert!(!$mask::default().all());
            assert_eq!($mask::default().bitmask(), 0x0);
            let a = $new(-1.0, -1.0, -1.0);
            let b = $new(1.0, 1.0, 1.0);
            let c = $new(-1.0, -1.0, 1.0);
            let d = $new(1.0, -1.0, -1.0);
            assert_eq!(a.cmplt(a).bitmask(), 0x0);
            assert_eq!(a.cmplt(b).bitmask(), 0x7);
            assert_eq!(a.cmplt(c).bitmask(), 0x4);
            assert_eq!(c.cmple(a).bitmask(), 0x3);
            assert_eq!(a.cmplt(d).bitmask(), 0x1);
            assert!(a.cmplt(b).all());
            assert!(a.cmplt(c).any());
            assert!(a.cmple(b).all());
            assert!(a.cmple(a).all());
            assert!(b.cmpgt(a).all());
            assert!(b.cmpge(a).all());
            assert!(b.cmpge(b).all());
            assert!(!(a.cmpge(c).all()));
            assert!(c.cmple(c).all());
            assert!(c.cmpge(c).all());
            assert!(a == a);
            assert!(a < b);
            assert!(b > a);
        }

        #[test]
        fn test_extend_truncate() {
            let a = $new(1.0, 2.0, 3.0);
            let b = a.extend(4.0);
            assert_eq!((1.0, 2.0, 3.0, 4.0), b.into());
            let c = $vec3::from(b.truncate());
            assert_eq!(a, c);
        }

        #[test]
        fn test_mask() {
            let mut a = $vec3::zero();
            a.x = 1.0;
            a.y = 1.0;
            a.z = 1.0;
            assert!(!a.cmpeq($vec3::zero()).any());
            assert!(a.cmpeq($vec3::splat(1.0)).all());
        }

        #[test]
        fn test_mask_as_ref() {
            assert_eq!($mask::new(false, false, false).as_ref(), &[0, 0, 0]);
            assert_eq!($mask::new(true, false, false).as_ref(), &[!0, 0, 0]);
            assert_eq!($mask::new(false, true, true).as_ref(), &[0, !0, !0]);
            assert_eq!($mask::new(false, true, false).as_ref(), &[0, !0, 0]);
            assert_eq!($mask::new(true, false, true).as_ref(), &[!0, 0, !0]);
            assert_eq!($mask::new(true, true, true).as_ref(), &[!0, !0, !0]);
        }

        #[test]
        fn test_mask_from() {
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(false, false, false)),
                [0, 0, 0]
            );
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(true, false, false)),
                [!0, 0, 0]
            );
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(false, true, true)),
                [0, !0, !0]
            );
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(false, true, false)),
                [0, !0, 0]
            );
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(true, false, true)),
                [!0, 0, !0]
            );
            assert_eq!(
                Into::<[u32; 3]>::into($mask::new(true, true, true)),
                [!0, !0, !0]
            );
        }

        #[test]
        fn test_mask_bitmask() {
            assert_eq!($mask::new(false, false, false).bitmask(), 0b000);
            assert_eq!($mask::new(true, false, false).bitmask(), 0b001);
            assert_eq!($mask::new(false, true, true).bitmask(), 0b110);
            assert_eq!($mask::new(false, true, false).bitmask(), 0b010);
            assert_eq!($mask::new(true, false, true).bitmask(), 0b101);
            assert_eq!($mask::new(true, true, true).bitmask(), 0b111);
        }

        #[test]
        fn test_mask_any() {
            assert_eq!($mask::new(false, false, false).any(), false);
            assert_eq!($mask::new(true, false, false).any(), true);
            assert_eq!($mask::new(false, true, false).any(), true);
            assert_eq!($mask::new(false, false, true).any(), true);
        }

        #[test]
        fn test_mask_all() {
            assert_eq!($mask::new(true, true, true).all(), true);
            assert_eq!($mask::new(false, true, true).all(), false);
            assert_eq!($mask::new(true, false, true).all(), false);
            assert_eq!($mask::new(true, true, false).all(), false);
        }

        #[test]
        fn test_mask_select() {
            let a = $vec3::new(1.0, 2.0, 3.0);
            let b = $vec3::new(4.0, 5.0, 6.0);
            assert_eq!(
                $vec3::select($mask::new(true, true, true), a, b),
                $vec3::new(1.0, 2.0, 3.0),
            );
            assert_eq!(
                $vec3::select($mask::new(true, false, true), a, b),
                $vec3::new(1.0, 5.0, 3.0),
            );
            assert_eq!(
                $vec3::select($mask::new(false, true, false), a, b),
                $vec3::new(4.0, 2.0, 6.0),
            );
            assert_eq!(
                $vec3::select($mask::new(false, false, false), a, b),
                $vec3::new(4.0, 5.0, 6.0),
            );
        }

        #[test]
        fn test_mask_and() {
            assert_eq!(
                ($mask::new(false, false, false) & $mask::new(false, false, false)).bitmask(),
                0b000,
            );
            assert_eq!(
                ($mask::new(true, true, true) & $mask::new(true, true, true)).bitmask(),
                0b111,
            );
            assert_eq!(
                ($mask::new(true, false, true) & $mask::new(false, true, false)).bitmask(),
                0b000,
            );
            assert_eq!(
                ($mask::new(true, false, true) & $mask::new(true, true, true)).bitmask(),
                0b101,
            );

            let mut mask = $mask::new(true, true, false);
            mask &= $mask::new(true, false, false);
            assert_eq!(mask.bitmask(), 0b001);
        }

        #[test]
        fn test_mask_or() {
            assert_eq!(
                ($mask::new(false, false, false) | $mask::new(false, false, false)).bitmask(),
                0b000,
            );
            assert_eq!(
                ($mask::new(true, true, true) | $mask::new(true, true, true)).bitmask(),
                0b111,
            );
            assert_eq!(
                ($mask::new(true, false, true) | $mask::new(false, true, false)).bitmask(),
                0b111,
            );
            assert_eq!(
                ($mask::new(true, false, true) | $mask::new(true, false, true)).bitmask(),
                0b101,
            );

            let mut mask = $mask::new(true, true, false);
            mask |= $mask::new(true, false, false);
            assert_eq!(mask.bitmask(), 0b011);
        }

        #[test]
        fn test_mask_not() {
            assert_eq!((!$mask::new(false, false, false)).bitmask(), 0b111);
            assert_eq!((!$mask::new(true, true, true)).bitmask(), 0b000);
            assert_eq!((!$mask::new(true, false, true)).bitmask(), 0b010);
            assert_eq!((!$mask::new(false, true, false)).bitmask(), 0b101);
        }

        #[test]
        fn test_mask_fmt() {
            let a = $mask::new(true, false, false);

            // debug fmt
            assert_eq!(
                format!("{:?}", a),
                format!("{}(0xffffffff, 0x0, 0x0)", stringify!($mask))
            );

            // display fmt
            assert_eq!(format!("{}", a), "[true, false, false]");
        }

        #[test]
        fn test_mask_eq() {
            let a = $mask::new(true, false, true);
            let b = $mask::new(true, false, true);
            let c = $mask::new(false, true, true);

            assert_eq!(a, b);
            assert_eq!(b, a);
            assert_ne!(a, c);
            assert_ne!(b, c);

            assert!(a > c);
            assert!(c < a);
        }

        #[test]
        fn test_mask_hash() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hash;
            use std::hash::Hasher;

            let a = $mask::new(true, false, true);
            let b = $mask::new(true, false, true);
            let c = $mask::new(false, true, true);

            let mut hasher = DefaultHasher::new();
            a.hash(&mut hasher);
            let a_hashed = hasher.finish();

            let mut hasher = DefaultHasher::new();
            b.hash(&mut hasher);
            let b_hashed = hasher.finish();

            let mut hasher = DefaultHasher::new();
            c.hash(&mut hasher);
            let c_hashed = hasher.finish();

            assert_eq!(a, b);
            assert_eq!(a_hashed, b_hashed);
            assert_ne!(a, c);
            assert_ne!(a_hashed, c_hashed);
        }

        #[test]
        fn test_signum() {
            assert_eq!($vec3::zero().signum(), $vec3::one());
            assert_eq!(-$vec3::zero().signum(), -$vec3::one());
            assert_eq!($vec3::one().signum(), $vec3::one());
            assert_eq!((-$vec3::one()).signum(), -$vec3::one());
            assert_eq!($vec3::splat($t::INFINITY).signum(), $vec3::one());
            assert_eq!($vec3::splat($t::NEG_INFINITY).signum(), -$vec3::one());
            assert!($vec3::splat($t::NAN).signum().is_nan_mask().all());
        }

        #[test]
        fn test_abs() {
            assert_eq!($vec3::zero().abs(), $vec3::zero());
            assert_eq!($vec3::one().abs(), $vec3::one());
            assert_eq!((-$vec3::one()).abs(), $vec3::one());
        }

        #[test]
        fn test_round() {
            assert_eq!($vec3::new(1.35, 0.0, 0.0).round().x, 1.0);
            assert_eq!($vec3::new(0.0, 1.5, 0.0).round().y, 2.0);
            assert_eq!($vec3::new(0.0, 0.0, -15.5).round().z, -16.0);
            assert_eq!($vec3::new(0.0, 0.0, 0.0).round().z, 0.0);
            assert_eq!($vec3::new(0.0, 21.1, 0.0).round().y, 21.0);
            assert_eq!($vec3::new(0.0, 11.123, 0.0).round().y, 11.0);
            assert_eq!($vec3::new(0.0, 11.499, 0.0).round().y, 11.0);
            assert_eq!(
                $vec3::new($t::NEG_INFINITY, $t::INFINITY, 0.0).round(),
                $vec3::new($t::NEG_INFINITY, $t::INFINITY, 0.0)
            );
            assert!($vec3::new($t::NAN, 0.0, 0.0).round().x.is_nan());
        }

        #[test]
        fn test_floor() {
            assert_eq!(
                $vec3::new(1.35, 1.5, -1.5).floor(),
                $vec3::new(1.0, 1.0, -2.0)
            );
            assert_eq!(
                $vec3::new($t::INFINITY, $t::NEG_INFINITY, 0.0).floor(),
                $vec3::new($t::INFINITY, $t::NEG_INFINITY, 0.0)
            );
            assert!($vec3::new($t::NAN, 0.0, 0.0).floor().x.is_nan());
            assert_eq!(
                $vec3::new(-2000000.123, 10000000.123, 1000.9).floor(),
                $vec3::new(-2000001.0, 10000000.0, 1000.0)
            );
        }

        #[test]
        fn test_ceil() {
            assert_eq!(
                $vec3::new(1.35, 1.5, -1.5).ceil(),
                $vec3::new(2.0, 2.0, -1.0)
            );
            assert_eq!(
                $vec3::new($t::INFINITY, $t::NEG_INFINITY, 0.0).ceil(),
                $vec3::new($t::INFINITY, $t::NEG_INFINITY, 0.0)
            );
            assert!($vec3::new($t::NAN, 0.0, 0.0).ceil().x.is_nan());
            assert_eq!(
                $vec3::new(-2000000.123, 1000000.123, 1000.9).ceil(),
                $vec3::new(-2000000.0, 1000001.0, 1001.0)
            );
        }

        #[test]
        fn test_lerp() {
            let v0 = $vec3::new(-1.0, -1.0, -1.0);
            let v1 = $vec3::new(1.0, 1.0, 1.0);
            assert_approx_eq!(v0, v0.lerp(v1, 0.0));
            assert_approx_eq!(v1, v0.lerp(v1, 1.0));
            assert_approx_eq!($vec3::zero(), v0.lerp(v1, 0.5));
        }

        #[test]
        fn test_to_from_slice() {
            let v = $vec3::new(1.0, 2.0, 3.0);
            let mut a = [0.0, 0.0, 0.0];
            v.write_to_slice_unaligned(&mut a);
            assert_eq!(v, $vec3::from_slice_unaligned(&a));
        }

        #[test]
        fn test_angle_between() {
            let angle = $vec3::new(1.0, 0.0, 1.0).angle_between($vec3::new(1.0, 1.0, 0.0));
            assert_approx_eq!(core::$t::consts::FRAC_PI_3, angle, 1e-6);

            let angle = $vec3::new(10.0, 0.0, 10.0).angle_between($vec3::new(5.0, 5.0, 0.0));
            assert_approx_eq!(core::$t::consts::FRAC_PI_3, angle, 1e-6);

            let angle = $vec3::new(-1.0, 0.0, -1.0).angle_between($vec3::new(1.0, -1.0, 0.0));
            assert_approx_eq!(2.0 * core::$t::consts::FRAC_PI_3, angle, 1e-6);
        }

        #[cfg(feature = "std")]
        #[test]
        fn test_sum() {
            let one = $vec3::one();
            assert_eq!(vec![one, one].iter().sum::<$vec3>(), one + one);
        }

        #[cfg(feature = "std")]
        #[test]
        fn test_product() {
            let two = $vec3::new(2.0, 2.0, 2.0);
            assert_eq!(vec![two, two].iter().product::<$vec3>(), two * two);
        }

        #[test]
        fn test_is_finite() {
            use core::$t::INFINITY;
            use core::$t::NAN;
            use core::$t::NEG_INFINITY;
            assert!($vec3::new(0.0, 0.0, 0.0).is_finite());
            assert!($vec3::new(-1e-10, 1.0, 1e10).is_finite());
            assert!(!$vec3::new(INFINITY, 0.0, 0.0).is_finite());
            assert!(!$vec3::new(0.0, NAN, 0.0).is_finite());
            assert!(!$vec3::new(0.0, 0.0, NEG_INFINITY).is_finite());
            assert!(!$vec3::splat(NAN).is_finite());
        }

        #[test]
        fn test_powf() {
            assert_eq!(
                $vec3::new(2.0, 4.0, 8.0).powf(2.0),
                $vec3::new(4.0, 16.0, 64.0)
            );
        }

        #[test]
        fn test_exp() {
            assert_eq!(
                $vec3::new(1.0, 2.0, 3.0).exp(),
                $vec3::new((1.0 as $t).exp(), (2.0 as $t).exp(), (3.0 as $t).exp())
            );
        }
    };
}

mod vec3 {
    use glam::{vec3, Vec3, Vec3Mask};

    #[test]
    fn test_align() {
        use std::mem;
        assert_eq!(12, mem::size_of::<Vec3>());
        assert_eq!(4, mem::align_of::<Vec3>());
        assert_eq!(12, mem::size_of::<Vec3Mask>());
        assert_eq!(4, mem::align_of::<Vec3Mask>());
    }

    impl_vec3_tests!(vec3, Vec3, Vec3Mask, f32);
}

mod vec3a {
    use glam::{vec3a, Vec3A, Vec3AMask, Vec4};

    #[test]
    fn test_vec3a_align() {
        use std::mem;
        assert_eq!(16, mem::size_of::<Vec3A>());
        assert_eq!(16, mem::align_of::<Vec3A>());
        if cfg!(all(target_feature = "sse2", not(feature = "scalar-math"))) {
            assert_eq!(16, mem::size_of::<Vec3AMask>());
            assert_eq!(16, mem::align_of::<Vec3AMask>());
        } else {
            assert_eq!(12, mem::size_of::<Vec3AMask>());
            assert_eq!(4, mem::align_of::<Vec3AMask>());
        }
    }

    #[test]
    fn test_mask_align16() {
        // make sure the unused 'w' value doesn't break Vec3Ab behaviour
        let a = Vec4::zero();
        let mut b = Vec3A::from(a);
        b.x = 1.0;
        b.y = 1.0;
        b.z = 1.0;
        assert!(!b.cmpeq(Vec3A::zero()).any());
        assert!(b.cmpeq(Vec3A::splat(1.0)).all());
    }

    #[cfg(vec3a_sse2)]
    #[test]
    fn test_vec3a_m128() {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        #[repr(C, align(16))]
        struct F32x3_A16([f32; 3]);

        let v0 = Vec3A::new(1.0, 2.0, 3.0);
        let m0: __m128 = v0.into();
        let mut a0 = F32x3_A16([0.0, 0.0, 0.0]);
        unsafe {
            _mm_store_ps(a0.0.as_mut_ptr(), m0);
        }
        assert_eq!([1.0, 2.0, 3.0], a0.0);
        let v1 = Vec3A::from(m0);
        assert_eq!(v0, v1);

        #[repr(C, align(16))]
        struct U32x3_A16([u32; 3]);

        let v0 = Vec3AMask::new(true, false, true);
        let m0: __m128 = v0.into();
        let mut a0 = U32x3_A16([1, 2, 3]);
        unsafe {
            _mm_store_ps(a0.0.as_mut_ptr() as *mut f32, m0);
        }
        assert_eq!([0xffffffff, 0, 0xffffffff], a0.0);
    }

    impl_vec3_tests!(vec3a, Vec3A, Vec3AMask, f32);
}

mod dvec3 {
    use glam::{dvec3, DVec3, DVec3Mask};

    #[test]
    fn test_align() {
        use std::mem;
        assert_eq!(24, mem::size_of::<DVec3>());
        assert_eq!(8, mem::align_of::<DVec3>());
        assert_eq!(12, mem::size_of::<DVec3Mask>());
        assert_eq!(4, mem::align_of::<DVec3Mask>());
    }

    impl_vec3_tests!(dvec3, DVec3, DVec3Mask, f64);
}
