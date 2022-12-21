pub mod api;
mod cl_device;
pub use cl_device::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;