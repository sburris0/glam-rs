#[macro_use]
mod support;

macro_rules! impl_vec4_tests {
    ($new:ident, $vec4:ident, $mask:ident, $t:ident) => {
        #[test]
        fn test_vec4_new() {
            let v = $new(1.0, 2.0, 3.0, 4.0);

            assert_eq!(v.x, 1.0);
            assert_eq!(v.y, 2.0);
            assert_eq!(v.z, 3.0);
            assert_eq!(v.w, 4.0);

            let t = (1.0, 2.0, 3.0, 4.0);
            let v = $vec4::from(t);
            assert_eq!(t, v.into());

            let a = [1.0, 2.0, 3.0, 4.0];
            let v = $vec4::from(a);
            let a1: [$t; 4] = v.into();
            assert_eq!(a, a1);

            let v = $vec4::new(t.0, t.1, t.2, t.3);
            assert_eq!(t, v.into());

            assert_eq!($vec4::new(1.0, 0.0, 0.0, 0.0), $vec4::unit_x());
            assert_eq!($vec4::new(0.0, 1.0, 0.0, 0.0), $vec4::unit_y());
            assert_eq!($vec4::new(0.0, 0.0, 1.0, 0.0), $vec4::unit_z());
            assert_eq!($vec4::new(0.0, 0.0, 0.0, 1.0), $vec4::unit_w());
        }

        #[test]
        fn test_vec4_fmt() {
            let a = $vec4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(
                format!("{:?}", a),
                format!("{}(1.0, 2.0, 3.0, 4.0)", stringify!($vec4))
            );
            // assert_eq!(
            //     format!("{:#?}", a),
            //     "$vec4(\n    1.0,\n    2.0,\n    3.0,\n    4.0\n)"
            // );
            assert_eq!(format!("{}", a), "[1, 2, 3, 4]");
        }

        #[test]
        fn test_vec4_zero() {
            let v = $vec4::zero();
            assert_eq!((0.0, 0.0, 0.0, 0.0), v.into());
            assert_eq!(v, $vec4::default());
        }

        #[test]
        fn test_vec4_splat() {
            let v = $vec4::splat(1.0);
            assert_eq!((1.0, 1.0, 1.0, 1.0), v.into());
        }

        #[test]
        fn test_vec4_accessors() {
            let mut a = $vec4::zero();
            a.x = 1.0;
            a.y = 2.0;
            a.z = 3.0;
            a.w = 4.0;
            assert_eq!(1.0, a.x);
            assert_eq!(2.0, a.y);
            assert_eq!(3.0, a.z);
            assert_eq!(4.0, a.w);
            assert_eq!((1.0, 2.0, 3.0, 4.0), a.into());

            let mut a = $vec4::zero();
            a[0] = 1.0;
            a[1] = 2.0;
            a[2] = 3.0;
            a[3] = 4.0;
            assert_eq!(1.0, a[0]);
            assert_eq!(2.0, a[1]);
            assert_eq!(3.0, a[2]);
            assert_eq!(4.0, a[3]);
            assert_eq!((1.0, 2.0, 3.0, 4.0), a.into());
        }

        #[test]
        fn test_vec4_funcs() {
            let x = $new(1.0, 0.0, 0.0, 0.0);
            let y = $new(0.0, 1.0, 0.0, 0.0);
            let z = $new(0.0, 0.0, 1.0, 0.0);
            let w = $new(0.0, 0.0, 0.0, 1.0);
            assert_eq!(1.0, x.dot(x));
            assert_eq!(0.0, x.dot(y));
            assert_eq!(-1.0, z.dot(-z));
            assert_eq!(4.0, (2.0 * x).length_squared());
            assert_eq!(9.0, (-3.0 * y).length_squared());
            assert_eq!(16.0, (4.0 * z).length_squared());
            assert_eq!(64.0, (8.0 * w).length_squared());
            assert_eq!(2.0, (-2.0 * x).length());
            assert_eq!(3.0, (3.0 * y).length());
            assert_eq!(4.0, (-4.0 * z).length());
            assert_eq!(5.0, (-5.0 * w).length());
            assert_eq!(2.0, x.distance_squared(y));
            assert_eq!(13.0, (2.0 * x).distance_squared(-3.0 * z));
            assert_eq!((2.0 as $t).sqrt(), w.distance(y));
            assert_eq!(5.0, (3.0 * x).distance(-4.0 * y));
            assert_eq!(13.0, (-5.0 * w).distance(12.0 * y));
            assert_eq!(x, (2.0 * x).normalize());
            assert_eq!(
                1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0,
                $new(1.0, 2.0, 3.0, 4.0).dot($new(5.0, 6.0, 7.0, 8.0))
            );
            assert_eq!(
                2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0,
                $new(2.0, 3.0, 4.0, 5.0).length_squared()
            );
            assert_eq!(
                (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0).sqrt(),
                $new(2.0, 3.0, 4.0, 5.0).length()
            );
            assert_eq!(
                1.0 / (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0).sqrt(),
                $new(2.0, 3.0, 4.0, 5.0).length_recip()
            );
            assert!($new(2.0, 3.0, 4.0, 5.0).normalize().is_normalized());
            assert_approx_eq!(
                $new(2.0, 3.0, 4.0, 5.0)
                    / (2.0 as $t * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0).sqrt(),
                $new(2.0, 3.0, 4.0, 5.0).normalize()
            );
            assert_eq!(
                $new(0.5, 0.25, 0.125, 0.0625),
                $new(2.0, 4.0, 8.0, 16.0).recip()
            );
        }

        #[test]
        fn test_vec4_ops() {
            let a = $new(1.0, 2.0, 3.0, 4.0);
            assert_eq!((2.0, 4.0, 6.0, 8.0), (a + a).into());
            assert_eq!((0.0, 0.0, 0.0, 0.0), (a - a).into());
            assert_eq!((1.0, 4.0, 9.0, 16.0), (a * a).into());
            assert_eq!((2.0, 4.0, 6.0, 8.0), (a * 2.0).into());
            assert_eq!((2.0, 4.0, 6.0, 8.0), (2.0 * a).into());
            assert_eq!((1.0, 1.0, 1.0, 1.0), (a / a).into());
            assert_eq!((0.5, 1.0, 1.5, 2.0), (a / 2.0).into());
            assert_eq!((1.0, 0.5, 1.0 / 3.0, 0.25), (1.0 / a).into());
            assert_eq!((-1.0, -2.0, -3.0, -4.0), (-a).into());
        }

        #[test]
        fn test_vec4_assign_ops() {
            let a = $new(1.0, 2.0, 3.0, 4.0);
            let mut b = a;
            b += a;
            assert_eq!((2.0, 4.0, 6.0, 8.0), b.into());
            b -= a;
            assert_eq!((1.0, 2.0, 3.0, 4.0), b.into());
            b *= a;
            assert_eq!((1.0, 4.0, 9.0, 16.0), b.into());
            b /= a;
            assert_eq!((1.0, 2.0, 3.0, 4.0), b.into());
            b *= 2.0;
            assert_eq!((2.0, 4.0, 6.0, 8.0), b.into());
            b /= 2.0;
            assert_eq!((1.0, 2.0, 3.0, 4.0), b.into());
        }

        #[test]
        fn test_vec4_min_max() {
            let a = $new(-1.0, 2.0, -3.0, 4.0);
            let b = $new(1.0, -2.0, 3.0, -4.0);
            assert_eq!((-1.0, -2.0, -3.0, -4.0), a.min(b).into());
            assert_eq!((-1.0, -2.0, -3.0, -4.0), b.min(a).into());
            assert_eq!((1.0, 2.0, 3.0, 4.0), a.max(b).into());
            assert_eq!((1.0, 2.0, 3.0, 4.0), b.max(a).into());
        }

        #[test]
        fn test_vec4_hmin_hmax() {
            let a = $new(-1.0, 4.0, -3.0, 2.0);
            assert_eq!(-3.0, a.min_element());
            assert_eq!(4.0, a.max_element());
            assert_eq!(3.0, $new(1.0, 2.0, 3.0, 4.0).truncate().max_element());
            assert_eq!(-3.0, $new(-1.0, -2.0, -3.0, -4.0).truncate().min_element());
        }

        #[test]
        fn test_vec4_eq() {
            let a = $new(1.0, 1.0, 1.0, 1.0);
            let b = $new(1.0, 2.0, 3.0, 4.0);
            assert!(a.cmpeq(a).all());
            assert!(b.cmpeq(b).all());
            assert!(a.cmpne(b).any());
            assert!(b.cmpne(a).any());
            assert!(b.cmpeq(a).any());
        }

        #[test]
        fn test_vec4_cmp() {
            assert!(!$mask::default().any());
            assert!(!$mask::default().all());
            assert_eq!($mask::default().bitmask(), 0x0);
            let a = $new(-1.0, -1.0, -1.0, -1.0);
            let b = $new(1.0, 1.0, 1.0, 1.0);
            let c = $new(-1.0, -1.0, 1.0, 1.0);
            let d = $new(1.0, -1.0, -1.0, 1.0);
            assert_eq!(a.cmplt(a).bitmask(), 0x0);
            assert_eq!(a.cmplt(b).bitmask(), 0xf);
            assert_eq!(a.cmplt(c).bitmask(), 0xc);
            assert_eq!(c.cmple(a).bitmask(), 0x3);
            assert_eq!(a.cmplt(d).bitmask(), 0x9);
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
            assert!(a.cmpeq(a).all());
            assert!(!a.cmpeq(b).all());
            assert!(a.cmpeq(c).any());
            assert!(!a.cmpne(a).all());
            assert!(a.cmpne(b).all());
            assert!(a.cmpne(c).any());
            assert!(a == a);
            assert!(a < b);
            assert!(b > a);
        }

        #[test]
        fn test_vec4_slice() {
            let a = [1.0, 2.0, 3.0, 4.0];
            let b = $vec4::from_slice_unaligned(&a);
            let c: [$t; 4] = b.into();
            assert_eq!(a, c);
            let mut d = [0.0, 0.0, 0.0, 0.0];
            b.write_to_slice_unaligned(&mut d[..]);
            assert_eq!(a, d);
        }

        #[test]
        fn test_vec4_signum() {
            assert_eq!($vec4::zero().signum(), $vec4::one());
            assert_eq!(-$vec4::zero().signum(), -$vec4::one());
            assert_eq!($vec4::one().signum(), $vec4::one());
            assert_eq!((-$vec4::one()).signum(), -$vec4::one());
            assert_eq!($vec4::splat($t::INFINITY).signum(), $vec4::one());
            assert_eq!($vec4::splat($t::NEG_INFINITY).signum(), -$vec4::one());
            assert!($vec4::splat($t::NAN).signum().is_nan_mask().all());
        }

        #[test]
        fn test_vec4_abs() {
            assert_eq!($vec4::zero().abs(), $vec4::zero());
            assert_eq!($vec4::one().abs(), $vec4::one());
            assert_eq!((-$vec4::one()).abs(), $vec4::one());
        }

        #[test]
        fn test_vec4mask_as_ref() {
            assert_eq!(
                $mask::new(false, false, false, false).as_ref(),
                &[0, 0, 0, 0]
            );
            assert_eq!(
                $mask::new(false, false, true, true).as_ref(),
                &[0, 0, !0, !0]
            );
            assert_eq!(
                $mask::new(true, true, false, false).as_ref(),
                &[!0, !0, 0, 0]
            );
            assert_eq!(
                $mask::new(false, true, false, true).as_ref(),
                &[0, !0, 0, !0]
            );
            assert_eq!(
                $mask::new(true, false, true, false).as_ref(),
                &[!0, 0, !0, 0]
            );
            assert_eq!(
                $mask::new(true, true, true, true).as_ref(),
                &[!0, !0, !0, !0]
            );
        }

        #[test]
        fn test_vec4mask_from() {
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(false, false, false, false)),
                [0, 0, 0, 0]
            );
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(false, false, true, true)),
                [0, 0, !0, !0]
            );
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(true, true, false, false)),
                [!0, !0, 0, 0]
            );
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(false, true, false, true)),
                [0, !0, 0, !0]
            );
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(true, false, true, false)),
                [!0, 0, !0, 0]
            );
            assert_eq!(
                Into::<[u32; 4]>::into($mask::new(true, true, true, true)),
                [!0, !0, !0, !0]
            );
        }

        #[test]
        fn test_vec4mask_bitmask() {
            assert_eq!($mask::new(false, false, false, false).bitmask(), 0b0000);
            assert_eq!($mask::new(false, false, true, true).bitmask(), 0b1100);
            assert_eq!($mask::new(true, true, false, false).bitmask(), 0b0011);
            assert_eq!($mask::new(false, true, false, true).bitmask(), 0b1010);
            assert_eq!($mask::new(true, false, true, false).bitmask(), 0b0101);
            assert_eq!($mask::new(true, true, true, true).bitmask(), 0b1111);
        }

        #[test]
        fn test_vec4mask_any() {
            assert_eq!($mask::new(false, false, false, false).any(), false);
            assert_eq!($mask::new(true, false, false, false).any(), true);
            assert_eq!($mask::new(false, true, false, false).any(), true);
            assert_eq!($mask::new(false, false, true, false).any(), true);
            assert_eq!($mask::new(false, false, false, true).any(), true);
        }

        #[test]
        fn test_vec4mask_all() {
            assert_eq!($mask::new(true, true, true, true).all(), true);
            assert_eq!($mask::new(false, true, true, true).all(), false);
            assert_eq!($mask::new(true, false, true, true).all(), false);
            assert_eq!($mask::new(true, true, false, true).all(), false);
            assert_eq!($mask::new(true, true, true, false).all(), false);
        }

        #[test]
        fn test_vec4mask_select() {
            let a = $vec4::new(1.0, 2.0, 3.0, 4.0);
            let b = $vec4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(
                $vec4::select($mask::new(true, true, true, true), a, b),
                $vec4::new(1.0, 2.0, 3.0, 4.0),
            );
            assert_eq!(
                $vec4::select($mask::new(true, false, true, false), a, b),
                $vec4::new(1.0, 6.0, 3.0, 8.0),
            );
            assert_eq!(
                $vec4::select($mask::new(false, true, false, true), a, b),
                $vec4::new(5.0, 2.0, 7.0, 4.0),
            );
            assert_eq!(
                $vec4::select($mask::new(false, false, false, false), a, b),
                $vec4::new(5.0, 6.0, 7.0, 8.0),
            );
        }

        #[test]
        fn test_vec4mask_and() {
            assert_eq!(
                ($mask::new(false, false, false, false) & $mask::new(false, false, false, false))
                    .bitmask(),
                0b0000,
            );
            assert_eq!(
                ($mask::new(true, true, true, true) & $mask::new(true, true, true, true)).bitmask(),
                0b1111,
            );
            assert_eq!(
                ($mask::new(true, false, true, false) & $mask::new(false, true, false, true))
                    .bitmask(),
                0b0000,
            );
            assert_eq!(
                ($mask::new(true, false, true, true) & $mask::new(true, true, true, false))
                    .bitmask(),
                0b0101,
            );

            let mut mask = $mask::new(true, true, false, false);
            mask &= $mask::new(true, false, true, false);
            assert_eq!(mask.bitmask(), 0b0001);
        }

        #[test]
        fn test_vec4mask_or() {
            assert_eq!(
                ($mask::new(false, false, false, false) | $mask::new(false, false, false, false))
                    .bitmask(),
                0b0000,
            );
            assert_eq!(
                ($mask::new(true, true, true, true) | $mask::new(true, true, true, true)).bitmask(),
                0b1111,
            );
            assert_eq!(
                ($mask::new(true, false, true, false) | $mask::new(false, true, false, true))
                    .bitmask(),
                0b1111,
            );
            assert_eq!(
                ($mask::new(true, false, true, false) | $mask::new(true, false, true, false))
                    .bitmask(),
                0b0101,
            );

            let mut mask = $mask::new(true, true, false, false);
            mask |= $mask::new(true, false, true, false);
            assert_eq!(mask.bitmask(), 0b0111);
        }

        #[test]
        fn test_vec4mask_not() {
            assert_eq!((!$mask::new(false, false, false, false)).bitmask(), 0b1111);
            assert_eq!((!$mask::new(true, true, true, true)).bitmask(), 0b0000);
            assert_eq!((!$mask::new(true, false, true, false)).bitmask(), 0b1010);
            assert_eq!((!$mask::new(false, true, false, true)).bitmask(), 0b0101);
        }

        #[test]
        fn test_vec4mask_fmt() {
            let a = $mask::new(true, false, true, false);

            assert_eq!(format!("{}", a), "[true, false, true, false]");
            assert_eq!(
                format!("{:?}", a),
                format!("{}(0xffffffff, 0x0, 0xffffffff, 0x0)", stringify!($mask))
            );
        }

        #[test]
        fn test_vec4mask_eq() {
            let a = $mask::new(true, false, true, false);
            let b = $mask::new(true, false, true, false);
            let c = $mask::new(false, true, true, false);

            assert_eq!(a, b);
            assert_eq!(b, a);
            assert_ne!(a, c);
            assert_ne!(b, c);

            assert!(a > c);
            assert!(c < a);
        }

        #[test]
        fn test_vec4mask_hash() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hash;
            use std::hash::Hasher;

            let a = $mask::new(true, false, true, false);
            let b = $mask::new(true, false, true, false);
            let c = $mask::new(false, true, true, false);

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
        fn test_vec4_round() {
            assert_eq!($vec4::new(1.35, 0.0, 0.0, 0.0).round().x, 1.0);
            assert_eq!($vec4::new(0.0, 1.5, 0.0, 0.0).round().y, 2.0);
            assert_eq!($vec4::new(0.0, 0.0, -15.5, 0.0).round().z, -16.0);
            assert_eq!($vec4::new(0.0, 0.0, 0.0, 0.0).round().z, 0.0);
            assert_eq!($vec4::new(0.0, 21.1, 0.0, 0.0).round().y, 21.0);
            assert_eq!($vec4::new(0.0, 0.0, 0.0, 11.123).round().w, 11.0);
            assert_eq!($vec4::new(0.0, 0.0, 11.501, 0.0).round().z, 12.0);
            assert_eq!(
                $vec4::new($t::NEG_INFINITY, $t::INFINITY, 1.0, -1.0).round(),
                $vec4::new($t::NEG_INFINITY, $t::INFINITY, 1.0, -1.0)
            );
            assert!($vec4::new($t::NAN, 0.0, 0.0, 1.0).round().x.is_nan());
        }

        #[test]
        fn test_vec4_floor() {
            assert_eq!(
                $vec4::new(1.35, 1.5, -1.5, 1.999).floor(),
                $vec4::new(1.0, 1.0, -2.0, 1.0)
            );
            assert_eq!(
                $vec4::new($t::INFINITY, $t::NEG_INFINITY, 0.0, 0.0).floor(),
                $vec4::new($t::INFINITY, $t::NEG_INFINITY, 0.0, 0.0)
            );
            assert!($vec4::new(0.0, $t::NAN, 0.0, 0.0).floor().y.is_nan());
            assert_eq!(
                $vec4::new(-0.0, -2000000.123, 10000000.123, 1000.9).floor(),
                $vec4::new(-0.0, -2000001.0, 10000000.0, 1000.0)
            );
        }

        #[test]
        fn test_vec4_ceil() {
            assert_eq!(
                $vec4::new(1.35, 1.5, -1.5, 1234.1234).ceil(),
                $vec4::new(2.0, 2.0, -1.0, 1235.0)
            );
            assert_eq!(
                $vec4::new($t::INFINITY, $t::NEG_INFINITY, 0.0, 0.0).ceil(),
                $vec4::new($t::INFINITY, $t::NEG_INFINITY, 0.0, 0.0)
            );
            assert!($vec4::new(0.0, 0.0, $t::NAN, 0.0).ceil().z.is_nan());
            assert_eq!(
                $vec4::new(-1234.1234, -2000000.123, 1000000.123, 1000.9).ceil(),
                $vec4::new(-1234.0, -2000000.0, 1000001.0, 1001.0)
            );
        }

        #[test]
        fn test_vec4_lerp() {
            let v0 = $vec4::new(-1.0, -1.0, -1.0, -1.0);
            let v1 = $vec4::new(1.0, 1.0, 1.0, 1.0);
            assert_approx_eq!(v0, v0.lerp(v1, 0.0));
            assert_approx_eq!(v1, v0.lerp(v1, 1.0));
            assert_approx_eq!($vec4::zero(), v0.lerp(v1, 0.5));
        }

        #[test]
        fn test_vec4_to_from_slice() {
            let v = $vec4::new(1.0, 2.0, 3.0, 4.0);
            let mut a = [0.0, 0.0, 0.0, 0.0];
            v.write_to_slice_unaligned(&mut a);
            assert_eq!(v, $vec4::from_slice_unaligned(&a));
        }

        #[cfg(feature = "serde")]
        #[test]
        fn test_vec4_serde() {
            let a = $vec4::new(1.0, 2.0, 3.0, 4.0);
            let serialized = serde_json::to_string(&a).unwrap();
            assert_eq!(serialized, "[1.0,2.0,3.0,4.0]");
            let deserialized = serde_json::from_str(&serialized).unwrap();
            assert_eq!(a, deserialized);
            let deserialized = serde_json::from_str::<$vec4>("[]");
            assert!(deserialized.is_err());
            let deserialized = serde_json::from_str::<$vec4>("[1.0]");
            assert!(deserialized.is_err());
            let deserialized = serde_json::from_str::<$vec4>("[1.0,2.0]");
            assert!(deserialized.is_err());
            let deserialized = serde_json::from_str::<$vec4>("[1.0,2.0,3.0]");
            assert!(deserialized.is_err());
            let deserialized = serde_json::from_str::<$vec4>("[1.0,2.0,3.0,4.0,5.0]");
            assert!(deserialized.is_err());
        }

        #[cfg(feature = "rand")]
        #[test]
        fn test_vec4_rand() {
            use rand::{Rng, SeedableRng};
            use rand_xoshiro::Xoshiro256Plus;
            let mut rng1 = Xoshiro256Plus::seed_from_u64(0);
            let a: ($t, $t, $t, $t) = rng1.gen();
            let mut rng2 = Xoshiro256Plus::seed_from_u64(0);
            let b: $vec4 = rng2.gen();
            assert_eq!(a, b.into());
        }

        #[cfg(feature = "std")]
        #[test]
        fn test_sum() {
            let one = $vec4::one();
            assert_eq!(vec![one, one].iter().sum::<$vec4>(), one + one);
        }

        #[cfg(feature = "std")]
        #[test]
        fn test_product() {
            let two = $vec4::new(2.0, 2.0, 2.0, 2.0);
            assert_eq!(vec![two, two].iter().product::<$vec4>(), two * two);
        }

        #[test]
        fn test_vec4_is_finite() {
            use std::$t::INFINITY;
            use std::$t::NAN;
            use std::$t::NEG_INFINITY;
            assert!($vec4::new(0.0, 0.0, 0.0, 0.0).is_finite());
            assert!($vec4::new(-1e-10, 1.0, 1e10, 42.0).is_finite());
            assert!(!$vec4::new(INFINITY, 0.0, 0.0, 0.0).is_finite());
            assert!(!$vec4::new(0.0, NAN, 0.0, 0.0).is_finite());
            assert!(!$vec4::new(0.0, 0.0, NEG_INFINITY, 0.0).is_finite());
            assert!(!$vec4::new(0.0, 0.0, 0.0, NAN).is_finite());
        }

        #[test]
        fn test_powf() {
            assert_eq!(
                $vec4::new(2.0, 4.0, 8.0, 16.0).powf(2.0),
                $vec4::new(4.0, 16.0, 64.0, 256.0)
            );
        }

        #[test]
        fn test_exp() {
            assert_eq!(
                $vec4::new(1.0, 2.0, 3.0, 4.0).exp(),
                $vec4::new(
                    (1.0 as $t).exp(),
                    (2.0 as $t).exp(),
                    (3.0 as $t).exp(),
                    (4.0 as $t).exp()
                )
            );
        }
    };
}

mod vec4 {
    use glam::{vec4, Vec4, Vec4Mask};

    #[test]
    fn test_vec4_align() {
        use std::mem;
        assert_eq!(16, mem::size_of::<Vec4>());
        assert_eq!(16, mem::size_of::<Vec4Mask>());
        if cfg!(feature = "scalar-math") {
            assert_eq!(4, mem::align_of::<Vec4>());
            assert_eq!(4, mem::align_of::<Vec4Mask>());
        } else {
            assert_eq!(16, mem::align_of::<Vec4>());
            assert_eq!(16, mem::align_of::<Vec4Mask>());
        }
    }

    #[cfg(vec4_sse2)]
    #[test]
    fn test_vec4_m128() {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        #[repr(C, align(16))]
        struct F32x4_A16([f32; 4]);

        let v0 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let m0: __m128 = v0.into();
        let mut a0 = F32x4_A16([0.0, 0.0, 0.0, 0.0]);
        unsafe {
            _mm_store_ps(a0.0.as_mut_ptr(), m0);
        }
        assert_eq!([1.0, 2.0, 3.0, 4.0], a0.0);
        let v1 = Vec4::from(m0);
        assert_eq!(v0, v1);

        #[repr(C, align(16))]
        struct U32x4_A16([u32; 4]);

        let v0 = Vec4Mask::new(true, false, true, false);
        let m0: __m128 = v0.into();
        let mut a0 = U32x4_A16([1, 2, 3, 4]);
        unsafe {
            _mm_store_ps(a0.0.as_mut_ptr() as *mut f32, m0);
        }
        assert_eq!([0xffffffff, 0, 0xffffffff, 0], a0.0);
    }

    impl_vec4_tests!(vec4, Vec4, Vec4Mask, f32);
}

mod dvec4 {
    use glam::{dvec4, DVec4, DVec4Mask};

    #[test]
    fn test_vec4_align() {
        use std::mem;
        assert_eq!(32, mem::size_of::<DVec4>());
        assert_eq!(16, mem::size_of::<DVec4Mask>());
        assert_eq!(8, mem::align_of::<DVec4>());
        assert_eq!(4, mem::align_of::<DVec4Mask>());
    }

    impl_vec4_tests!(dvec4, DVec4, DVec4Mask, f64);
}
