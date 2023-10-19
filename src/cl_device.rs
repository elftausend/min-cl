use std::{cell::RefCell, fmt::Debug, time::Duration};

use crate::{
    api::{
        clEnqueueNDRangeKernel, cl_event, create_command_queue, create_context, get_device_ids,
        get_platforms, wait_for_events, CLIntDevice, CommandQueue, Context, DeviceType, Event,
        Kernel, OCLErrorKind, Platform,
    },
    init_devices,
    measure_perf::measure_perf,
    Error, DEVICES, kernel_cache::KernelCache,
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
    pub kernel_cache: RefCell<KernelCache>
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
            event_wait_list: Default::default(),
            kernel_cache: Default::default()
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
        let mut event = [std::ptr::null_mut(); 1];
        let lws = match lws {
            Some(lws) => lws.as_ptr(),
            None => std::ptr::null(),
        };
        let offset = match offset {
            Some(offset) => offset.as_ptr(),
            None => std::ptr::null(),
        };

        let value = unsafe {
            clEnqueueNDRangeKernel(
                self.queue.0,
                kernel.0,
                wd as u32,
                offset,
                gws.as_ptr(),
                lws,
                0,
                std::ptr::null(),
                event.as_mut_ptr() as *mut cl_event,
            )
        };

        if value != 0 {
            return Err(Error::from(OCLErrorKind::from_value(value)));
        }

        self.event_wait_list.borrow_mut().push(Event(event[0]));

        Ok(())
    }

    #[inline]
    pub fn wait_for_events(&self) -> Result<(), Error> {
        wait_for_events(&self.event_wait_list.borrow())?;
        self.event_wait_list.borrow_mut().clear();
        Ok(())
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
        create_buffer::<f32>(&device.ctx, MemFlags::MemReadWrite as u64, 10000, None).unwrap();
        println!("device name: {}", device.device.get_name().unwrap());
        create_buffer::<f32>(&device.ctx, MemFlags::MemReadWrite as u64, 9423 * 123, None).unwrap();

        println!(
            "{}",
            device.device.get_global_mem().unwrap() as f32 * 10f32.powf(-9.)
        )
    }
}
