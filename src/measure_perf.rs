use std::{ffi::c_void, mem::size_of, time::Instant};

use crate::{
    api::{
        build_program, create_buffer, create_kernels_in_program, create_program_with_source,
        enqueue_nd_range_kernel, release_mem_object, set_kernel_arg, wait_for_event, MemFlags,
        OCLErrorKind,
    },
    CLDevice, Error,
};

const SIZE: usize = 100_000;

pub fn measure_perf(device: &CLDevice) -> Result<std::time::Duration, Error> {
    let src = "
        __kernel void add(__global const int* lhs, __global const int* rhs, __global int* out) {
            int idx = get_global_id(0);

            out[idx] = lhs[idx] + rhs[idx];
        }
    ";

    let program = unsafe { create_program_with_source(&device.ctx, src)? };
    unsafe { build_program(&program, &[device.device], Some("-cl-std=CL1.2"))? }; //-cl-single-precision-constant

    let kernel = unsafe {
        create_kernels_in_program(&program)?
            .into_iter()
            .next()
            .ok_or(OCLErrorKind::InvalidKernel)?
    };

    let lhs = unsafe {
        create_buffer::<i32>(
            &device.ctx,
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            SIZE,
            Some(&vec![1i32; SIZE]),
        )?
    };
    let rhs = unsafe {
        create_buffer::<i32>(
            &device.ctx,
            MemFlags::MemReadWrite | MemFlags::MemCopyHostPtr,
            SIZE,
            Some(&vec![2i32; SIZE]),
        )?
    };
    let out =
        unsafe { create_buffer::<i32>(&device.ctx, MemFlags::MemReadWrite as u64, SIZE, None)? };

    unsafe { set_kernel_arg(&kernel, 0, lhs, size_of::<*const c_void>(), false)? };
    unsafe { set_kernel_arg(&kernel, 1, rhs, size_of::<*const c_void>(), false)? };
    unsafe { set_kernel_arg(&kernel, 2, out, size_of::<*const c_void>(), false)? };

    let start = Instant::now();

    for _ in 0..10 {
        // waits till completion
        unsafe {
            wait_for_event(enqueue_nd_range_kernel(
                &device.queue,
                &kernel,
                1,
                &[SIZE, 0, 0],
                None,
                None,
                None,
            )?)?
        };
    }

    let dur = start.elapsed();

    unsafe { release_mem_object(lhs)? };
    unsafe { release_mem_object(rhs)? };
    unsafe { release_mem_object(out)? };

    Ok(dur)
}
