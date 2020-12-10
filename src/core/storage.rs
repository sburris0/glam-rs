#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(C)]
pub struct XY<T> {
    pub x: T,
    pub y: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(C)]
pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(C)]
pub struct XYZW<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(C)]
pub struct XYAxes<T> {
    pub x_axis: XY<T>,
    pub y_axis: XY<T>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(align(16))]
pub(crate) struct Align16<T>(pub T);

impl<T> Align16<T> {
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const T {
        &self.0
    }

    #[allow(dead_code)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.0
    }
}

#[test]
fn test_align16() {
    use core::{mem, ptr};
    let mut a = Align16::<f32>(1.0);
    assert_eq!(mem::align_of_val(&a), 16);
    unsafe {
        assert_eq!(ptr::read(a.as_ptr()), 1.0);
        ptr::write(a.as_mut_ptr(), -1.0);
    }
    assert_eq!(a.0, -1.0);
}
