use std::{cell::RefCell};

use crate::{
    api::{
        create_command_queue, create_context, get_device_ids, get_platforms, CLIntDevice,
        CommandQueue, Context, DeviceType, Event, Kernel, OCLErrorKind, clEnqueueNDRangeKernel, cl_event, wait_for_events,
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
    pub event_wait_list: RefCell<Vec<Event>>,
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
            event_wait_list: Default::default(),
        })
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
