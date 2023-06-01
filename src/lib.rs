//! A simple global allocator which hooks into `libc`.
//! Useful when linking `no_std` + `alloc` code into existing embedded C code.
//!
//! Uses `posix_memalign` for allocations, `realloc` for reallocations, and
//! `free` for deallocations.
//!
//! ## Example
//!
//! ```
//! use libc_alloc::LibcAlloc;
//!
//! #[global_allocator]
//! static ALLOCATOR: LibcAlloc = LibcAlloc;
//! ```

#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::ptr;

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[cfg(any(
    target_arch = "x86",
    target_arch = "arm",
    target_arch = "m68k",
    target_arch = "mips",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "sparc",
    target_arch = "asmjs",
    target_arch = "wasm32",
    target_arch = "hexagon",
    all(target_arch = "riscv32", not(target_os = "espidf")),
    all(target_arch = "xtensa", not(target_os = "espidf")),
))]
pub const MIN_ALIGN: usize = 8;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "loongarch64",
    target_arch = "mips64",
    target_arch = "s390x",
    target_arch = "sparc64",
    target_arch = "riscv64",
    target_arch = "wasm64",
))]
pub const MIN_ALIGN: usize = 16;
// The allocator on the esp-idf platform guarantees 4 byte alignment.
#[cfg(any(
    all(target_arch = "riscv32", target_os = "espidf"),
    all(target_arch = "xtensa", target_os = "espidf"),
))]
pub const MIN_ALIGN: usize = 4;

#[cfg(any(target_family = "unix", target_family = "none"))]
mod libc;

#[cfg(target_family = "windows")]
mod win_crt;

/// Global Allocator which hooks into libc to allocate / free memory.
pub struct LibcAlloc;

#[cfg(any(target_family = "unix", target_family = "none"))]
unsafe impl GlobalAlloc for LibcAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ptr = ptr::null_mut();
        let ret = libc::posix_memalign(
            &mut ptr,
            layout.align().max(core::mem::size_of::<usize>()),
            layout.size(),
        );
        if ret == 0 {
            ptr as *mut u8
        } else {
            ptr::null_mut()
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Unfortunately, calloc doesn't make any alignment guarantees, so the memory
        // has to be manually zeroed-out.
        let ptr = self.alloc(layout);
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        libc::free(ptr as *mut c_void);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // check layout, and if it requires stricter alignment, fallback to alloc + copy + free.
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            libc::realloc(ptr as *mut c_void, new_size) as *mut u8
        } else {
            realloc_fallback(self, ptr, layout, new_size)
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "none"))]
pub unsafe fn realloc_fallback(
    alloc: &LibcAlloc,
    ptr: *mut u8,
    old_layout: Layout,
    new_size: usize,
) -> *mut u8 {
    let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

    let new_ptr = alloc.alloc(new_layout);
    if !new_ptr.is_null() {
        let size = core::cmp::min(old_layout.size(), new_size);
        ptr::copy_nonoverlapping(ptr, new_ptr, size);
        alloc.dealloc(ptr, old_layout);
    }
    new_ptr
}

#[cfg(target_family = "windows")]
unsafe impl GlobalAlloc for LibcAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        win_crt::_aligned_malloc(layout.size(), layout.align()) as *mut u8
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        win_crt::_aligned_free(ptr as *mut c_void)
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Unfortunately, _aligned_calloc does not exist, so the memory
        // has to be manually zeroed-out.
        let ptr = self.alloc(layout);
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        win_crt::_aligned_realloc(ptr as *mut c_void, new_size, layout.align()) as *mut u8
    }
}
