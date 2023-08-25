pub mod api;
mod cl_device;
mod measure_perf;
pub use cl_device::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

use crate::measured_devices;
use std::{sync::RwLock, time::Duration};

pub static DEVICES: RwLock<Option<Vec<(Duration, usize, usize, CLDevice)>>> = RwLock::new(None);

pub fn init_devices() {
    if DEVICES.read().unwrap().is_none() {
        *DEVICES.write().unwrap() =
            Some(measured_devices().expect("Could not gather OpenCL devices"));
    }
}

#[cfg(test)]
mod tests {
    use crate::{init_devices, DEVICES};

    #[test]
    fn test_devices() {
        init_devices();
        println!("{:?}", DEVICES.read())
    }
}
