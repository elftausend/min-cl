use std::{cell::RefCell, ffi::c_void, fmt::Debug, time::Duration};

use crate::{
    api::{
        create_command_queue, create_context, enqueue_full_copy_buffer, enqueue_map_buffer,
        enqueue_nd_range_kernel, enqueue_read_buffer, enqueue_write_buffer, get_device_ids,
        get_platforms, wait_for_events, CLIntDevice, CommandQueue, Context, DeviceType, Event,
        Kernel, OCLErrorKind, Platform,
    },
    init_devices,
    kernel_cache::KernelCache,
    measure_perf::measure_perf,
    Error, DEVICES,
};

pub fn all_devices() -> Result<Vec<Vec<CLIntDevice>>, Error> {
    Ok(get_platforms()?
        .into_iter()
        .map(all_devices_of_platform)
        .collect())
}

pub fn all_devices_of_platform(platform: Platform) -> Vec<CLIntDevice> {
    [
        DeviceType::GPU as u64 | DeviceType::ACCELERATOR as u64,
        DeviceType::CPU as u64,
    ]
    .into_iter()
    .flat_map(|device_type| get_device_ids(platform, &device_type))
    .flatten()
    .collect()
}

/// Internal representation of an OpenCL Device.
pub struct CLDevice {
    pub device: CLIntDevice,
    pub ctx: Context,
    pub queue: CommandQueue,
    pub unified_mem: bool,
    pub event_wait_list: RefCell<Vec<Event>>,
    pub kernel_cache: RefCell<KernelCache>,
}

impl Debug for CLDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CLDevice")
            .field("name", &self.device.get_name().unwrap())
            .field("unified_mem", &self.unified_mem)
            .field("event_wait_list", &self.event_wait_list.borrow())
            .finish()
    }
}

unsafe impl Sync for CLDevice {}
unsafe impl Send for CLDevice {}

impl TryFrom<CLIntDevice> for CLDevice {
    type Error = Error;

    fn try_from(device: CLIntDevice) -> Result<Self, Self::Error> {
        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;
        let unified_mem = device.unified_mem()?;

        Ok(CLDevice {
            device,
            ctx,
            queue,
            unified_mem,
            event_wait_list: RefCell::new(Vec::with_capacity(100)),
            kernel_cache: Default::default(),
        })
    }
}

pub fn measured_devices() -> Result<Vec<(Duration, usize, usize, CLDevice)>, Error> {
    Ok(all_devices()?
        .into_iter()
        .enumerate()
        .map(|(platform_idx, devices)| {
            devices
                .into_iter()
                .map(TryInto::try_into)
                .enumerate()
                .filter_map(move |(device_idx, device)| {
                    Some((
                        measure_perf(device.as_ref().ok()?).ok()?,
                        platform_idx,
                        device_idx,
                        device.ok()?,
                    ))
                })
        })
        .flatten()
        .collect::<Vec<_>>())
}

pub fn extract_indices_from_device_idx(device_idx: usize) -> Result<(usize, usize), Error> {
    let rwlock_guard = DEVICES.read().map_err(|_| OCLErrorKind::InvalidDevice)?;
    let devices = rwlock_guard.as_ref().unwrap();

    let (_, platform_idx, device_idx, _) = &devices
        .get(device_idx)
        .ok_or(OCLErrorKind::InvalidDeviceIdx)?;
    Ok((*platform_idx, *device_idx))
}

impl CLDevice {
    pub fn from_indices(platform_idx: usize, device_idx: usize) -> Result<CLDevice, Error> {
        let platform = get_platforms()?[platform_idx];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;

        devices[device_idx].try_into()
    }
    pub fn new(device_idx: usize) -> Result<CLDevice, Error> {
        init_devices();

        let (platform_idx, device_idx) = extract_indices_from_device_idx(device_idx)?;
        CLDevice::from_indices(platform_idx, device_idx)
    }

    pub fn fastest() -> Result<CLDevice, Error> {
        Ok(measured_devices()?
            .into_iter()
            .min_by_key(|(dur, _, _, _)| *dur)
            .ok_or(OCLErrorKind::InvalidDevice)
            .map(|(_, _, _, device)| device)?)
    }

    pub fn enqueue_nd_range_kernel(
        &self,
        kernel: &Kernel,
        wd: usize,
        gws: &[usize; 3],
        lws: Option<&[usize; 3]>,
        offset: Option<[usize; 3]>,
    ) -> Result<(), Error> {
        let event = unsafe {
            enqueue_nd_range_kernel(
                &self.queue,
                kernel,
                wd,
                gws,
                lws,
                offset,
                Some(&self.event_wait_list.borrow()),
            )
        }?;

        {
            let mut event_wait_list = self.event_wait_list.borrow_mut();
            event_wait_list.clear();
            event_wait_list.push(event);
        }

        // self.wait_for_events().unwrap();

        Ok(())
    }

    pub unsafe fn enqueue_read_buffer<T>(
        &self,
        src_ptr: *mut c_void,
        dst_slice: &mut [T],
        block: bool,
    ) -> Result<Event, Error> {
        unsafe {
            enqueue_read_buffer(
                self.queue(),
                src_ptr,
                dst_slice,
                block,
                Some(&self.event_wait_list.borrow()),
            )
        }
    }

    pub unsafe fn enqueue_write_buffer<T>(
        &self,
        dst_ptr: *mut c_void,
        src_slice: &[T],
        block: bool,
    ) -> Result<Event, Error> {
        unsafe {
            enqueue_write_buffer(
                self.queue(),
                dst_ptr,
                &src_slice,
                block,
                Some(&self.event_wait_list.borrow()),
            )
        }
    }

    pub unsafe fn enqueue_full_copy_buffer<T>(
        &self,
        src_mem: *mut c_void,
        dst_mem: *mut c_void,
        size: usize,
    ) -> Result<Event, Error> {
        unsafe {
            enqueue_full_copy_buffer::<T>(
                self.queue(),
                src_mem,
                dst_mem,
                size,
                Some(&self.event_wait_list.borrow()),
            )
        }
    }

    pub unsafe fn unified_ptr<T>(&self, ptr: *mut c_void, len: usize) -> Result<*mut T, Error> {
        unsafe {
            enqueue_map_buffer::<T>(
                self.queue(),
                ptr,
                true,
                2 | 1,
                0,
                len,
                Some(&self.event_wait_list.borrow()),
            )
            .map(|ptr| ptr as *mut T)
        }
    }

    #[inline]
    pub fn wait_for_events(&self) -> Result<(), Error> {
        unsafe {
            wait_for_events(&self.event_wait_list.borrow())?;
        }
        self.event_wait_list.borrow_mut().clear();
        Ok(())
    }

    /// Context of the OpenCL device.
    #[inline]
    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    /// Command queue of the OpenCL device.
    #[inline]
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// CLIntDevice of the OpenCL device.
    #[inline]
    pub fn device(&self) -> CLIntDevice {
        self.device
    }

    /// Returns the global memory size in GB.
    pub fn global_mem_size_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_global_mem()? as f64 * 10f64.powi(-9))
    }

    /// Returns the maximum memory allocation size in GB.
    pub fn max_mem_alloc_in_gb(&self) -> Result<f64, Error> {
        Ok(self.device().get_max_mem_alloc()? as f64 * 10f64.powi(-9))
    }

    /// Returns the name of the OpenCL device.
    pub fn name(&self) -> Result<String, Error> {
        self.device().get_name()
    }

    /// Returns the OpenCL version of the device.
    pub fn version(&self) -> Result<String, Error> {
        self.device().get_version()
    }

    /// Checks whether the device supports unified memory.
    #[inline]
    pub fn unified_mem(&self) -> bool {
        self.unified_mem
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        api::{create_buffer, MemFlags},
        CLDevice,
    };

    #[test]
    fn test_get_fastest() {
        let device = CLDevice::fastest().unwrap();
        unsafe {
            create_buffer::<f32>(&device.ctx, MemFlags::MemReadWrite as u64, 10000, None).unwrap()
        };
        println!("device name: {}", device.device.get_name().unwrap());
        unsafe {
            create_buffer::<f32>(&device.ctx, MemFlags::MemReadWrite as u64, 9423 * 123, None)
                .unwrap()
        };

        println!(
            "{}",
            device.device.get_global_mem().unwrap() as f32 * 10f32.powf(-9.)
        )
    }
}
