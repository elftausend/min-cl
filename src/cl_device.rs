use crate::{
    api::{
        create_command_queue, create_context, get_device_ids, get_platforms, CLIntDevice,
        CommandQueue, Context, DeviceType, OCLErrorKind,
    },
    Error,
};

/// Internal representation of an OpenCL Device.
#[derive(Debug)]
pub struct CLDevice {
    pub device: CLIntDevice,
    pub ctx: Context,
    pub queue: CommandQueue,
    pub unified_mem: bool,
}

unsafe impl Sync for CLDevice {}

impl CLDevice {
    pub fn new(device_idx: usize) -> Result<CLDevice, Error> {
        let platform = get_platforms()?[0];
        let devices = get_device_ids(platform, &(DeviceType::GPU as u64))?;

        if device_idx >= devices.len() {
            return Err(OCLErrorKind::InvalidDeviceIdx.into());
        }
        let device = devices[0];

        let ctx = create_context(&[device])?;
        let queue = create_command_queue(&ctx, device)?;
        let unified_mem = device.unified_mem()?;

        Ok(CLDevice {
            device,
            ctx,
            queue,
            unified_mem,
        })
    }
}
