#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OCLErrorKind {
    GetPlatformIDs,
    GetDeviceIDs,
    InvalidDeviceIdx,
    GetDeviceInfo,
    CreateContext,
    CreateCommandQueue,
    WaitForEvents,
    EnqueueReadBuffer,
    EnqueueWriteBuffer,
    EnqueueCopyBuffer,
    EnqueueNDRangeKernel,
    CreateBuffer,
    DeviceNotFound,
    DeviceNotAvailable,
    CompilerNotAvailable,
    MemObjectAllocationFailure,
    OutOfResources,
    OutOfHostMemory,
    ProfilingInfoNotAvailable,
    MemCopyOverlap,
    ImageFormatMismatch,
    ImageFormatNotSupported,
    BuildProgramFailures,
    MapFailure,
    MisalignedSubBufferOffset,
    ExecStatusErrorForEventsInWaitList,
    CompileProgramFailure,
    LinkerNotAvailable,
    LinkProgramFailure,
    DevicePartitionFailed,
    KernelArgInfoNotAvailable,
    InvalidValue,
    InvalidDeviceType,
    InvalidPlatform,
    InvalidDevice,
    InvalidContext,
    InvalidQueueProperties,
    InvalidCommandQueue,
    InvalidHostPtr,
    InvalidMemObject,
    InvalidImageFormatDescriptor,
    InvalidImageSize,
    InvalidSampler,
    InvalidBinary,
    InvalidBuildOptions,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefintion,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidKernelArgs,
    InvalidWorkDimension,
    InvalidWorkGroupSize,
    InvalidWorkItemSize,
    InvalidGlobalOffset,
    InvalidEventWaitList,
    InvalidEvent,
    InvalidOperation,
    InvalidGlObject,
    InvalidBufferSize,
    InvalidMIPLevel,
    InvalidGlobalWorkSize,
    InvalidProperty,
    InvalidImageDescriptor,
    InvalidCompilerOptions,
    InvalidLinkerOptions,
    InvalidDevicePartitionCount,
    InvalidPipeSize,
    InvalidDeviceQueue,
    PlatformNotFoundKHR,
    Unknown,
}

impl OCLErrorKind {
    pub fn from_value(value: i32) -> OCLErrorKind {
        match value {
            -1 => OCLErrorKind::DeviceNotFound,
            -2 => OCLErrorKind::DeviceNotAvailable,
            -3 => OCLErrorKind::CompilerNotAvailable,
            -4 => OCLErrorKind::MemObjectAllocationFailure,
            -5 => OCLErrorKind::OutOfResources,
            -6 => OCLErrorKind::OutOfHostMemory,
            -7 => OCLErrorKind::ProfilingInfoNotAvailable,
            -8 => OCLErrorKind::MemCopyOverlap,
            -9 => OCLErrorKind::ImageFormatMismatch,
            -10 => OCLErrorKind::ImageFormatNotSupported,
            -11 => OCLErrorKind::BuildProgramFailures,
            -12 => OCLErrorKind::MapFailure,
            -13 => OCLErrorKind::MisalignedSubBufferOffset,
            -14 => OCLErrorKind::ExecStatusErrorForEventsInWaitList,
            -15 => OCLErrorKind::CompileProgramFailure,
            -16 => OCLErrorKind::LinkerNotAvailable,
            -17 => OCLErrorKind::LinkProgramFailure,
            -18 => OCLErrorKind::DevicePartitionFailed,
            -19 => OCLErrorKind::KernelArgInfoNotAvailable,
            -30 => OCLErrorKind::InvalidValue,
            -31 => OCLErrorKind::InvalidDeviceType,
            -32 => OCLErrorKind::InvalidPlatform,
            -33 => OCLErrorKind::InvalidDevice,
            -34 => OCLErrorKind::InvalidContext,
            -35 => OCLErrorKind::InvalidQueueProperties,
            -36 => OCLErrorKind::InvalidCommandQueue,
            -37 => OCLErrorKind::InvalidHostPtr,
            -38 => OCLErrorKind::InvalidMemObject,
            -39 => OCLErrorKind::InvalidImageFormatDescriptor,
            -40 => OCLErrorKind::InvalidImageSize,
            -41 => OCLErrorKind::InvalidSampler,
            -42 => OCLErrorKind::InvalidBinary,
            -43 => OCLErrorKind::InvalidBuildOptions,
            -44 => OCLErrorKind::InvalidProgram,
            -45 => OCLErrorKind::InvalidProgramExecutable,
            -46 => OCLErrorKind::InvalidKernelName,
            -47 => OCLErrorKind::InvalidKernelDefintion,
            -48 => OCLErrorKind::InvalidKernel,
            -49 => OCLErrorKind::InvalidArgIndex,
            -50 => OCLErrorKind::InvalidArgValue,
            -51 => OCLErrorKind::InvalidArgSize,
            -52 => OCLErrorKind::InvalidKernelArgs,
            -53 => OCLErrorKind::InvalidWorkDimension,
            -54 => OCLErrorKind::InvalidWorkGroupSize,
            -55 => OCLErrorKind::InvalidWorkItemSize,
            -56 => OCLErrorKind::InvalidGlobalOffset,
            -57 => OCLErrorKind::InvalidEventWaitList,
            -58 => OCLErrorKind::InvalidEvent,
            -59 => OCLErrorKind::InvalidOperation,
            -60 => OCLErrorKind::InvalidGlObject,
            -61 => OCLErrorKind::InvalidBufferSize,
            -62 => OCLErrorKind::InvalidMIPLevel,
            -63 => OCLErrorKind::InvalidGlobalWorkSize,
            -64 => OCLErrorKind::InvalidProperty,
            -65 => OCLErrorKind::InvalidImageDescriptor,
            -66 => OCLErrorKind::InvalidCompilerOptions,
            -67 => OCLErrorKind::InvalidLinkerOptions,
            -68 => OCLErrorKind::InvalidDevicePartitionCount,
            -69 => OCLErrorKind::InvalidPipeSize,
            -70 => OCLErrorKind::InvalidDeviceQueue,
            -1001 => OCLErrorKind::PlatformNotFoundKHR,
            _ => OCLErrorKind::Unknown,
        }
    }
}

impl OCLErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            OCLErrorKind::Unknown => "Unkwown OpenCL Error",
            OCLErrorKind::GetPlatformIDs => "",
            OCLErrorKind::GetDeviceIDs => "",
            OCLErrorKind::InvalidDeviceIdx => {
                "Invalid device idx, specific OpenCL device not found"
            }
            OCLErrorKind::GetDeviceInfo => "",
            OCLErrorKind::CreateContext => "",
            OCLErrorKind::CreateCommandQueue => "",
            OCLErrorKind::WaitForEvents => "",
            OCLErrorKind::EnqueueReadBuffer => "",
            OCLErrorKind::EnqueueWriteBuffer => "",
            OCLErrorKind::EnqueueCopyBuffer => "",
            OCLErrorKind::EnqueueNDRangeKernel => "",
            OCLErrorKind::CreateBuffer => "",
            OCLErrorKind::DeviceNotFound => "(-1 DeviceNotFound) No OpenCL device found",
            OCLErrorKind::DeviceNotAvailable => {
                "(-2 DeviceNotAvailAble) OpenCL device is currently not available"
            }
            OCLErrorKind::CompilerNotAvailable => {
                "(-3 CompilerNotAvailable) Compiler is not available"
            }
            OCLErrorKind::MemObjectAllocationFailure => {
                "(-4 MemObjectAllocationFailure) Memory for buffer object could not be allocated"
            }
            OCLErrorKind::OutOfResources => {
                "(-5 OutOfResources) Allocation of resources failed on the OpenCL device"
            }
            OCLErrorKind::OutOfHostMemory => {
                "(-6 OutOfHostMemory) Allocation of resources failed on the host device"
            }
            OCLErrorKind::ProfilingInfoNotAvailable => "(-7 ProfilingInfoNotAvailable)",
            OCLErrorKind::MemCopyOverlap => "(-8 MemCopyOverlap)",
            OCLErrorKind::ImageFormatMismatch => "(-9 ImageFormatMismatch)",
            OCLErrorKind::ImageFormatNotSupported => "(-10 ImageFormatNotSupported)",
            OCLErrorKind::BuildProgramFailures => "(-11 BuildProgramFailures)",
            OCLErrorKind::MapFailure => "(-12 MapFailure)",
            OCLErrorKind::MisalignedSubBufferOffset => "(-13 MisalignedSubBufferOffset)",
            OCLErrorKind::ExecStatusErrorForEventsInWaitList => {
                "(-14 ExecStatusErrorForEventsInWaitList)"
            }
            OCLErrorKind::CompileProgramFailure => "(-15 CompileProgramFailure)",
            OCLErrorKind::LinkerNotAvailable => "(-16 LinkerNotAvailable)",
            OCLErrorKind::LinkProgramFailure => "(-17 LinkProgramFailure)",
            OCLErrorKind::DevicePartitionFailed => "(-18 DevicePartitionFailed)",
            OCLErrorKind::KernelArgInfoNotAvailable => "(-19 KernelArgInfoNotAvailable)",
            OCLErrorKind::InvalidValue => "(-30 InvalidValue)",
            OCLErrorKind::InvalidDeviceType => "(-31 InvalidDeviceType)",
            OCLErrorKind::InvalidPlatform => "(-32 InvalidPlatform)",
            OCLErrorKind::InvalidDevice => "(-33 InvalidDevice)",
            OCLErrorKind::InvalidContext => "(-34 InvalidContext)",
            OCLErrorKind::InvalidQueueProperties => "(-35 InvalidQueueProperties)",
            OCLErrorKind::InvalidCommandQueue => "(-36 InvalidCommandQueue)",
            OCLErrorKind::InvalidHostPtr => "(-37 InvalidHostPtr)",
            OCLErrorKind::InvalidMemObject => "(-38 InvalidMemObject)",
            OCLErrorKind::InvalidImageFormatDescriptor => "(-39 InvalidImageFormatDescriptor)",
            OCLErrorKind::InvalidImageSize => "(-40 InvalidImageSize)",
            OCLErrorKind::InvalidSampler => "(-41 InvalidSampler)",
            OCLErrorKind::InvalidBinary => "(-42 InvalidBinary)",
            OCLErrorKind::InvalidBuildOptions => "(-43 InvalidBuildOptions)",
            OCLErrorKind::InvalidProgram => "(-44 InvalidProgram)",
            OCLErrorKind::InvalidProgramExecutable => "(-45 InvalidProgramExecutable)",
            OCLErrorKind::InvalidKernelName => "(-46 InvalidKernelName)",
            OCLErrorKind::InvalidKernelDefintion => "(-47 InvalidKernelDefinition)",
            OCLErrorKind::InvalidKernel => "(-48 InvalidKernel)",
            OCLErrorKind::InvalidArgIndex => "(-49 InvalidArgIndex)",
            OCLErrorKind::InvalidArgValue => "(-50 InvalidArgValue)",
            OCLErrorKind::InvalidArgSize => "(-51 InvalidArgSize)",
            OCLErrorKind::InvalidKernelArgs => "(-52 InvalidKernelArgs) Invalid kernel args",
            OCLErrorKind::InvalidWorkDimension => "(-53 InvalidWorkDimension)",
            OCLErrorKind::InvalidWorkGroupSize => "(-54 InvalidWorkGroupSize)",
            OCLErrorKind::InvalidWorkItemSize => "(-55 InvalidWorkItemSize)",
            OCLErrorKind::InvalidGlobalOffset => "(-56 InvalidGlobalOffset)",
            OCLErrorKind::InvalidEventWaitList => "(-57 InvalidEventWaitList)",
            OCLErrorKind::InvalidEvent => "(-58 InvalidEvent)",
            OCLErrorKind::InvalidOperation => "(-59 InvalidOperation)",
            OCLErrorKind::InvalidGlObject => "(-60 InvalidGlObject)",
            OCLErrorKind::InvalidBufferSize => "(-61 InvalidBufferSize)",
            OCLErrorKind::InvalidMIPLevel => "(-62 InvalidMIPLevel)",
            OCLErrorKind::InvalidGlobalWorkSize => "(-63 InvalidGlobalWorkSize)",
            OCLErrorKind::InvalidProperty => "(-64 InvalidProperty)",
            OCLErrorKind::InvalidImageDescriptor => "(-65 InvalidImageDescriptor)",
            OCLErrorKind::InvalidCompilerOptions => "(-66 InvalidCompilerOptions)",
            OCLErrorKind::InvalidLinkerOptions => "(-67 InvalidLinkerOptions)",
            OCLErrorKind::InvalidDevicePartitionCount => "(-68 InvalidDevicePartitionCount)",
            OCLErrorKind::InvalidPipeSize => "(-69 InvalidPipeSize)",
            OCLErrorKind::InvalidDeviceQueue => "(-70 InvalidDeviceQueue)",
            OCLErrorKind::PlatformNotFoundKHR => "(-1001 PlatformNotFoundKHR)",
        }
    }
}

impl core::fmt::Debug for OCLErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl core::fmt::Display for OCLErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OCLErrorKind {}
