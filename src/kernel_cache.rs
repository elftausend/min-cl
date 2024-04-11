use crate::{CLDevice, Error};

use super::api::{
    build_program, create_kernels_in_program, create_program_with_source, Kernel, OCLErrorKind,
};
use std::collections::HashMap;

#[derive(Debug, Default)]
/// This stores the previously compiled OpenCL kernels.
pub struct KernelCache {
    /// Uses the kernel source code to retrieve the corresponding `Kernel`.
    pub kernel_cache: HashMap<String, Kernel>,
}

impl KernelCache {
    /// Returns a cached kernel. If the kernel source code does not exist, a new kernel is created and cached.
    pub fn kernel(&mut self, device: &CLDevice, src: &str) -> Result<&Kernel, Error> {
        if self.kernel_cache.contains_key(src) {
            return Ok(self.kernel_cache.get(src).unwrap());
        }
        /*if let Some(kernel) = self.kernel_cache.get(src) {
            return Ok(kernel);
        }*/

        let program = unsafe { create_program_with_source(&device.ctx, src)? };
        unsafe { build_program(&program, &[device.device], Some("-cl-std=CL1.2"))?; }//-cl-single-precision-constant

        let kernel = unsafe { create_kernels_in_program(&program)?
            .into_iter()
            .next()
            .ok_or(OCLErrorKind::InvalidKernel)? };

        self.kernel_cache.insert(src.to_string(), kernel);
        Ok(self.kernel_cache.get(src).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Error, CLDevice};

    use super::KernelCache;
    use std::collections::HashMap;

    #[test]
    fn test_kernel_cache() -> Result<(), Error> {
        let device = CLDevice::new(0)?;

        let mut kernel_cache = KernelCache {
            kernel_cache: HashMap::new(),
        };

        /*let mut kernel_fn = || {

        };*/

        let kernel = kernel_cache
            .kernel(
                &device,
                "
                __kernel void foo(__global float* test) {}
            ",
            )?
            .0;

        let same_kernel = kernel_cache
            .kernel(
                &device,
                "
                __kernel void foo(__global float* test) {}
            ",
            )?
            .0;

        assert_eq!(kernel, same_kernel);

        let another_kernel = kernel_cache
            .kernel(
                &device,
                "
                __kernel void bar(__global float* test, __global float* out) {}
            ",
            )?
            .0;

        assert_ne!(kernel, another_kernel);

        Ok(())
    }
}
