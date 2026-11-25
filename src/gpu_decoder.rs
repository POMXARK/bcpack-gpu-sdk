use bcpack::bc1::{QuadTable, Window};
use bcpack::bc7 as bcp_bc7;
use bcpack::bcn as bcp_bcn;
use bcpack::format::Mode as BcpMode;
use bcpack::tables::{BASE_TABLE, FREQ_TABLE, RF_TABLE, RN_TABLE, RS_TABLE};
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop};
use std::ptr::copy_nonoverlapping;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory6, DXGI_ADAPTER_FLAG_SOFTWARE,
    DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

pub const DECODE_CS_DXIL: &[u8] = include_bytes!("../shaders/bcpack_decode.dxil");
pub const BC7_RANS_CS_DXIL: &[u8] = include_bytes!("../shaders/bcpack_bc7_rans.dxil");
pub const BCN_RANS_CS_DXIL: &[u8] = include_bytes!("../shaders/bcpack_bcn_rans.dxil");
pub const BCN_REBUILD_CS_DXIL: &[u8] = include_bytes!("../shaders/bcpack_bcn_rebuild.dxil");
pub const BC7_REBUILD_CS_DXIL: &[u8] = include_bytes!("../shaders/bcpack_bc7_rebuild.dxil");
pub const BC16_PACK_CS_DXIL: &[u8] = include_bytes!("../shaders/bc16_pack.dxil");
pub const GFX10_DETILE_CS_DXIL: &[u8] = include_bytes!("../shaders/gfx10_detile.dxil");

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelDataGpu {
    pub bf: [u32; 10],    // low16 = base, high16 = freq
    pub rfrsn: [u32; 10], // bits 0..15 = rf, bits 16..20 = rs, bit 21 = rn
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ChunkDescGpu {
    pub stream_offset: u32,
    pub stream_len: u32,
    pub blocks: u32,
    pub out_block_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChunkStatusGpu {
    pub code: u32,
    pub consumed: u32,
    pub cycles: u32,
    pub blocks_written: u32,
}

pub struct BatchPayload {
    pub stream_bytes: Vec<u8>,
    pub chunks: Vec<ChunkDescGpu>,
    pub real_chunk_count: usize,
    pub total_blocks: usize,
}

pub fn pack_batch<'a, I>(items: I) -> BatchPayload
where
    I: IntoIterator<Item = (&'a [u8], usize)>,
{
    let mut stream_bytes = Vec::new();
    let mut chunks = Vec::new();
    let mut total_blocks = 0usize;

    for (stream, blocks) in items {
        while stream_bytes.len() % 4 != 0 {
            stream_bytes.push(0);
        }
        let offset = stream_bytes.len() as u32;
        stream_bytes.extend_from_slice(stream);
        chunks.push(ChunkDescGpu {
            stream_offset: offset,
            stream_len: stream.len() as u32,
            blocks: blocks as u32,
            out_block_offset: total_blocks as u32,
        });
        total_blocks += blocks;
    }
    let real_chunk_count = chunks.len();
    if chunks.len() % 2 == 1 {
        chunks.push(ChunkDescGpu {
            stream_offset: 0,
            stream_len: 0,
            blocks: 0,
            out_block_offset: 0,
        });
    }
    while stream_bytes.len() % 4 != 0 {
        stream_bytes.push(0);
    }
    // 256-byte safety tail so coalesced 64-byte stream_cache loads never read past buffer end
    stream_bytes.resize(stream_bytes.len() + 256, 0);

    BatchPayload {
        stream_bytes,
        chunks,
        real_chunk_count,
        total_blocks,
    }
}

pub fn build_packed_models() -> Vec<ModelDataGpu> {
    let mut models = Vec::with_capacity(144);
    for eab in 0..9 {
        for q in 0..16 {
            let mut md = ModelDataGpu {
                bf: [0; 10],
                rfrsn: [0; 10],
            };
            for k in 0..10 {
                let base = BASE_TABLE[eab][q][k] as u32;
                let freq = FREQ_TABLE[eab][q][k] as u32;
                let rf = RF_TABLE[eab][q][k] as u32;
                let rs = RS_TABLE[eab][q][k] as u32;
                let rn = RN_TABLE[eab][q][k] as u32;
                debug_assert!(rs <= 31 && rn <= 1);
                md.bf[k] = base | (freq << 16);
                md.rfrsn[k] = rf | (rs << 16) | (rn << 21);
            }
            models.push(md);
        }
    }
    models
}

pub fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

pub fn heap_props(ty: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    }
}

pub fn transition_barrier(
    res: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(res.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

pub fn uav_barrier(res: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(Some(res.clone())),
            }),
        },
    }
}

pub fn uav_barrier_all() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(None),
            }),
        },
    }
}

pub unsafe fn drop_barriers(barriers: &mut [D3D12_RESOURCE_BARRIER]) {
    for b in barriers.iter_mut() {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
            let mut tr = ManuallyDrop::take(&mut b.Anonymous.Transition);
            let _ = ManuallyDrop::take(&mut tr.pResource);
        } else if b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV {
            let mut uav = ManuallyDrop::take(&mut b.Anonymous.UAV);
            let _ = ManuallyDrop::take(&mut uav.pResource);
        }
    }
}

pub unsafe fn cmd_barriers(cmd: &ID3D12GraphicsCommandList, mut barriers: Vec<D3D12_RESOURCE_BARRIER>) {
    cmd.ResourceBarrier(&barriers);
    drop_barriers(&mut barriers);
}

pub fn create_committed_buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    initial_state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut res: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(
                &heap_props(heap),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, flags),
                initial_state,
                None,
                &mut res,
            )
            .expect("CreateCommittedResource failed");
    }
    res.unwrap()
}

pub unsafe fn dump_d3d12_info_queue(device: &ID3D12Device) {
    let reason = device.GetDeviceRemovedReason();
    eprintln!("[d3d12] GetDeviceRemovedReason = {reason:?}");
    if let Ok(iq) = device.cast::<ID3D12InfoQueue>() {
        let count = iq.GetNumStoredMessagesAllowedByRetrievalFilter();
        for i in 0..count {
            let mut len = 0usize;
            let _ = iq.GetMessage(i, None, &mut len);
            if len > 0 {
                let mut buf = vec![0u8; len];
                let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                if iq.GetMessage(i, Some(msg_ptr), &mut len).is_ok() {
                    let msg = &*msg_ptr;
                    if !msg.pDescription.is_null() && msg.DescriptionByteLength > 0 {
                        let s = std::slice::from_raw_parts(msg.pDescription, msg.DescriptionByteLength - 1);
                        eprintln!("[d3d12 {:?}] {}", msg.Severity, String::from_utf8_lossy(s));
                    }
                }
            }
        }
        iq.ClearStoredMessages();
    }
}

pub unsafe fn upload_slice<T: Copy>(res: &ID3D12Resource, data: &[T]) {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let empty_range = D3D12_RANGE { Begin: 0, End: 0 };
    if let Err(e) = res.Map(0, Some(&empty_range), Some(&mut ptr)) {
        let mut dev: Option<ID3D12Device> = None;
        let _ = res.GetDevice(&mut dev);
        if let Some(d) = dev {
            dump_d3d12_info_queue(&d);
        }
        panic!("Map failed: {e:?}");
    }
    copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, std::mem::size_of_val(data));
    res.Unmap(0, None);
}

unsafe fn create_root_sig_from_params(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
) -> ID3D12RootSignature {
    let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        },
    };
    let mut blob = None;
    let mut err_blob = None;
    if let Err(e) = D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err_blob)) {
        if let Some(eb) = err_blob {
            let msg = std::slice::from_raw_parts(eb.GetBufferPointer() as *const u8, eb.GetBufferSize());
            panic!("RootSig serialize error: {}", String::from_utf8_lossy(msg));
        }
        panic!("RootSig serialize failed: {e:?}");
    }
    let blob = blob.unwrap();
    let blob_slice = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    device
        .CreateRootSignature(0, blob_slice)
        .expect("CreateRootSignature")
}

unsafe fn create_compute_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    dxil: &[u8],
    label: &str,
) -> ID3D12PipelineState {
    let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr() as *const c_void,
            BytecodeLength: dxil.len(),
        },
        NodeMask: 0,
        CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
        Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
    };
    let pso = device
        .CreateComputePipelineState(&pso_desc)
        .unwrap_or_else(|e| panic!("CreateComputePipelineState ({label}) failed: {e:?}"));
    let _ = ManuallyDrop::take(&mut pso_desc.pRootSignature);
    pso
}

pub struct GpuContext {
    pub adapter_name: String,
    pub dedicated_vram_mb: u64,
    pub os_vram_budget_mb: u64,
    pub vrs_tier2: bool,
    pub vrs_tile_size: u32,
    pub vrs_additional_rates: bool,
    pub factory: IDXGIFactory6,
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub timestamp_freq: u64,
    pub fence: ID3D12Fence,
    pub fence_value: u64,
    pub fence_event: HANDLE,

    // Dedicated Async Compute engine (D3D12_COMMAND_LIST_TYPE_COMPUTE)
    pub compute_queue: ID3D12CommandQueue,
    pub compute_timestamp_freq: u64,
    pub compute_fence: ID3D12Fence,
    pub compute_fence_value: u64,
    pub compute_fence_event: HANDLE,
}

impl GpuContext {
    pub fn new(enable_debug: bool) -> Self {
        unsafe {
            if enable_debug {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok() {
                    if let Some(d) = debug {
                        d.EnableDebugLayer();
                    }
                }
            }

            let factory: IDXGIFactory6 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("CreateDXGIFactory2");

            let mut chosen: Option<(IDXGIAdapter1, String, u64)> = None;
            let mut idx = 0u32;
            while let Ok(adapter) =
                factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            {
                let desc = adapter.GetDesc1().expect("GetDesc1");
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0 {
                    let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
                    let name = String::from_utf16_lossy(&desc.Description[..end]);
                    let vram_mb = (desc.DedicatedVideoMemory as u64) / (1024 * 1024);
                    chosen = Some((adapter, name, vram_mb));
                    break;
                }
                idx += 1;
            }

            let (adapter, adapter_name, dedicated_vram_mb) = chosen.expect("No hardware DXGI adapter found");
            let os_vram_budget_mb = if let Ok(a3) = adapter.cast::<IDXGIAdapter3>() {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                if a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).is_ok() {
                    info.Budget / (1024 * 1024)
                } else {
                    dedicated_vram_mb
                }
            } else {
                dedicated_vram_mb
            };
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                .expect("D3D12CreateDevice");
            let device = device.unwrap();

            let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    &mut opts1 as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
                )
                .expect("CheckFeatureSupport OPTIONS1");
            assert!(opts1.WaveOps.as_bool(), "GPU does not support HLSL WaveOps");

            let mut opts6 = D3D12_FEATURE_DATA_D3D12_OPTIONS6::default();
            let opts6_ok = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS6,
                    &mut opts6 as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>() as u32,
                )
                .is_ok();
            let vrs_tier2 = opts6_ok && opts6.VariableShadingRateTier == D3D12_VARIABLE_SHADING_RATE_TIER_2;
            let vrs_tile_size = if vrs_tier2 { opts6.ShadingRateImageTileSize.max(8) } else { 16 };
            let vrs_additional_rates = vrs_tier2 && opts6.AdditionalShadingRatesSupported.as_bool();

            let direct_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&direct_desc).expect("CreateCommandQueue DIRECT");
            let timestamp_freq = queue.GetTimestampFrequency().expect("GetTimestampFrequency DIRECT");
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence DIRECT");
            let fence_event = CreateEventW(None, false, false, None).expect("CreateEventW DIRECT");

            let compute_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let compute_queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&compute_desc).expect("CreateCommandQueue COMPUTE");
            let compute_timestamp_freq = compute_queue
                .GetTimestampFrequency()
                .expect("GetTimestampFrequency COMPUTE");
            let compute_fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence COMPUTE");
            let compute_fence_event =
                CreateEventW(None, false, false, None).expect("CreateEventW COMPUTE");

            Self {
                adapter_name,
                dedicated_vram_mb,
                os_vram_budget_mb,
                vrs_tier2,
                vrs_tile_size,
                vrs_additional_rates,
                factory,
                device,
                queue,
                timestamp_freq,
                fence,
                fence_value: 0,
                fence_event,
                compute_queue,
                compute_timestamp_freq,
                compute_fence,
                compute_fence_value: 0,
                compute_fence_event,
            }
        }
    }

    /// Creates a `GpuContext` sharing an already-existing caller `ID3D12Device` (and optional `COMPUTE` queue).
    pub fn from_external_device(
        device: ID3D12Device,
        external_compute_queue: Option<ID3D12CommandQueue>,
    ) -> Self {
        unsafe {
            let factory: IDXGIFactory6 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("CreateDXGIFactory2");

            let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    &mut opts1 as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>() as u32,
                )
                .expect("CheckFeatureSupport OPTIONS1");
            assert!(opts1.WaveOps.as_bool(), "GPU does not support HLSL WaveOps");

            let direct_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&direct_desc).expect("CreateCommandQueue DIRECT (shared)");
            let timestamp_freq = queue.GetTimestampFrequency().expect("GetTimestampFrequency DIRECT");
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence DIRECT (shared)");
            let fence_event = CreateEventW(None, false, false, None).expect("CreateEventW DIRECT");

            let compute_queue: ID3D12CommandQueue = external_compute_queue.unwrap_or_else(|| {
                let compute_desc = D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                    Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                    NodeMask: 0,
                };
                device
                    .CreateCommandQueue(&compute_desc)
                    .expect("CreateCommandQueue COMPUTE (shared)")
            });
            let compute_timestamp_freq = compute_queue
                .GetTimestampFrequency()
                .expect("GetTimestampFrequency COMPUTE");
            let compute_fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("CreateFence COMPUTE (shared)");
            let compute_fence_event =
                CreateEventW(None, false, false, None).expect("CreateEventW COMPUTE");

            let mut opts6 = D3D12_FEATURE_DATA_D3D12_OPTIONS6::default();
            let opts6_ok = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS6,
                    &mut opts6 as *mut _ as *mut c_void,
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>() as u32,
                )
                .is_ok();
            let vrs_tier2 = opts6_ok && opts6.VariableShadingRateTier == D3D12_VARIABLE_SHADING_RATE_TIER_2;
            let vrs_tile_size = if vrs_tier2 { opts6.ShadingRateImageTileSize.max(8) } else { 16 };
            let vrs_additional_rates = vrs_tier2 && opts6.AdditionalShadingRatesSupported.as_bool();

            Self {
                adapter_name: "External ID3D12Device".to_string(),
                dedicated_vram_mb: 0,
                os_vram_budget_mb: 0,
                vrs_tier2,
                vrs_tile_size,
                vrs_additional_rates,
                factory,
                device,
                queue,
                timestamp_freq,
                fence,
                fence_value: 0,
                fence_event,
                compute_queue,
                compute_timestamp_freq,
                compute_fence,
                compute_fence_value: 0,
                compute_fence_event,
            }
        }
    }

    pub fn signal_direct_async(&mut self) -> u64 {
        unsafe {
            self.fence_value += 1;
            self.queue
                .Signal(&self.fence, self.fence_value)
                .expect("Signal DIRECT async");
            self.fence_value
        }
    }

    pub fn wait_direct_cpu(&self, val: u64) {
        unsafe {
            if val == 0 || self.fence.GetCompletedValue() >= val {
                return;
            }
            self.fence
                .SetEventOnCompletion(val, self.fence_event)
                .expect("SetEventOnCompletion DIRECT");
            WaitForSingleObject(self.fence_event, INFINITE);
        }
    }

    pub fn signal_and_wait(&mut self) {
        let v = self.signal_direct_async();
        self.wait_direct_cpu(v);
    }

    pub fn signal_compute(&mut self) -> u64 {
        unsafe {
            self.compute_fence_value += 1;
            self.compute_queue
                .Signal(&self.compute_fence, self.compute_fence_value)
                .expect("Signal COMPUTE");
            self.compute_fence_value
        }
    }

    pub fn wait_compute_cpu(&self, val: u64, timeout_ms: u32) -> bool {
        unsafe {
            if self.compute_fence.GetCompletedValue() >= val {
                return true;
            }
            self.compute_fence
                .SetEventOnCompletion(val, self.compute_fence_event)
                .expect("SetEventOnCompletion COMPUTE");
            WaitForSingleObject(self.compute_fence_event, timeout_ms) == windows::Win32::Foundation::WAIT_OBJECT_0
        }
    }

    pub fn signal_and_wait_compute(&mut self) {
        let val = self.signal_compute();
        let _ = self.wait_compute_cpu(val, INFINITE);
    }

    pub fn direct_wait_for_compute(&self, compute_fence_val: u64) {
        unsafe {
            self.queue
                .Wait(&self.compute_fence, compute_fence_val)
                .expect("Direct queue Wait on Compute fence");
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.fence_event);
            let _ = CloseHandle(self.compute_fence_event);
        }
    }
}

pub struct Bc1SlotResources {
    pub streams_upload: ID3D12Resource,
    pub streams_default: ID3D12Resource,
    pub chunks_upload: ID3D12Resource,
    pub chunks_default: ID3D12Resource,
    pub out_blocks_default: ID3D12Resource,
    pub status_default: ID3D12Resource,
    pub ts_query_heap: ID3D12QueryHeap,
    pub ts_readback: ID3D12Resource,
    pub cmd_alloc: ID3D12CommandAllocator,
    pub cmd_list: ID3D12GraphicsCommandList,
    pub last_compute_fence: u64,
}

impl Bc1SlotResources {
    fn new(
        device: &ID3D12Device,
        max_stream_bytes: usize,
        padded_max_chunks: usize,
        max_blocks: usize,
    ) -> Self {
        unsafe {
            let streams_upload = create_committed_buffer(device, D3D12_HEAP_TYPE_UPLOAD, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let streams_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let chunks_bytes = (padded_max_chunks * size_of::<ChunkDescGpu>()) as u64;
            let chunks_upload = create_committed_buffer(device, D3D12_HEAP_TYPE_UPLOAD, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let chunks_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let out_bytes = (max_blocks * 8) as u64;
            let out_blocks_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let status_bytes = (padded_max_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let status_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, status_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let cmd_alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .expect("CreateCommandAllocator COMPUTE slot");
            let cmd_list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None)
                .expect("CreateCommandList COMPUTE slot");
            cmd_list.Close().expect("Close initial slot cmd_list");

            Self {
                streams_upload,
                streams_default,
                chunks_upload,
                chunks_default,
                out_blocks_default,
                status_default,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
                last_compute_fence: 0,
            }
        }
    }
}

pub struct BcpackGpuDecoder {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    models_buf: ID3D12Resource,
    window_buf: ID3D12Resource,
    quads_buf: ID3D12Resource,

    cap_stream_bytes: usize,
    cap_chunks: usize,
    cap_blocks: usize,

    pub slots: [Bc1SlotResources; 2],
    out_readback: ID3D12Resource,
    status_readback: ID3D12Resource,
}

impl BcpackGpuDecoder {
    pub fn new(
        ctx: &mut GpuContext,
        window: &Window,
        quads: &QuadTable,
        max_chunks: usize,
        max_blocks: usize,
        max_stream_bytes: usize,
    ) -> Self {
        unsafe {
            let mut params = [D3D12_ROOT_PARAMETER1::default(); 7];
            for i in 0..5 {
                params[i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }
            for i in 0..2 {
                params[5 + i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }

            let root_sig = create_root_sig_from_params(&ctx.device, &params);
            let pso = create_compute_pso(&ctx.device, &root_sig, DECODE_CS_DXIL, "bcpack_decode.dxil");

            let models = build_packed_models();
            let mut win_packed = [0u32; 256];
            for (s, f) in window.slots.iter().enumerate() {
                let c0 = ((f[0] as u32) << 11) | ((f[1] as u32) << 5) | (f[2] as u32);
                let c1 = ((f[3] as u32) << 11) | ((f[4] as u32) << 5) | (f[5] as u32);
                win_packed[s] = c0 | (c1 << 16);
            }

            let mut quad_masks = [0u32; 1024];
            for q in 0..4 {
                quad_masks[q * 256..(q + 1) * 256].copy_from_slice(&quads.masks[q]);
            }

            let models_size = (models.len() * size_of::<ModelDataGpu>()) as u64;
            let win_size = (win_packed.len() * size_of::<u32>()) as u64;
            let quad_size = (quad_masks.len() * size_of::<u32>()) as u64;

            let models_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let win_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, win_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let quad_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, quad_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&models_up, &models);
            upload_slice(&win_up, &win_packed);
            upload_slice(&quad_up, &quad_masks);

            let models_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
            let window_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, win_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
            let quads_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, quad_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

            let padded_max_chunks = max_chunks + 2;
            let slots = [
                Bc1SlotResources::new(&ctx.device, max_stream_bytes, padded_max_chunks, max_blocks),
                Bc1SlotResources::new(&ctx.device, max_stream_bytes, padded_max_chunks, max_blocks),
            ];

            slots[0].cmd_alloc.Reset().unwrap();
            slots[0].cmd_list.Reset(&slots[0].cmd_alloc, None).unwrap();
            slots[0].cmd_list.CopyBufferRegion(&models_buf, 0, &models_up, 0, models_size);
            slots[0].cmd_list.CopyBufferRegion(&window_buf, 0, &win_up, 0, win_size);
            slots[0].cmd_list.CopyBufferRegion(&quads_buf, 0, &quad_up, 0, quad_size);
            let mut barriers = [
                transition_barrier(&models_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&window_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&quads_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ];
            slots[0].cmd_list.ResourceBarrier(&barriers);
            drop_barriers(&mut barriers);
            slots[0].cmd_list.Close().expect("Close");
            let lists = [Some(slots[0].cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();

            let out_bytes = (max_blocks * 8) as u64;
            let status_bytes = (padded_max_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let out_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let status_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, status_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            Self {
                root_sig,
                pso,
                models_buf,
                window_buf,
                quads_buf,
                cap_stream_bytes: max_stream_bytes,
                cap_chunks: padded_max_chunks,
                cap_blocks: max_blocks,
                slots,
                out_readback,
                status_readback,
            }
        }
    }

    pub fn out_buffer(&self, slot: usize) -> &ID3D12Resource {
        &self.slots[slot & 1].out_blocks_default
    }

    pub unsafe fn upload_batch_slot(&self, slot: usize, batch: &BatchPayload) {
        assert!(batch.stream_bytes.len() <= self.cap_stream_bytes, "stream buffer capacity exceeded");
        assert!(batch.chunks.len() <= self.cap_chunks, "chunks capacity exceeded");
        assert!(batch.total_blocks <= self.cap_blocks, "blocks capacity exceeded");
        let s = &self.slots[slot & 1];
        upload_slice(&s.streams_upload, &batch.stream_bytes);
        upload_slice(&s.chunks_upload, &batch.chunks);
    }

    pub unsafe fn upload_batch(&self, batch: &BatchPayload) {
        self.upload_batch_slot(0, batch);
    }

    pub unsafe fn record_slot_upload_and_dispatch(
        &self,
        slot: usize,
        cmd: &ID3D12GraphicsCommandList,
        batch: &BatchPayload,
        ts_heap: Option<&ID3D12QueryHeap>,
        ts_start_idx: u32,
        ts_end_idx: u32,
    ) {
        let s = &self.slots[slot & 1];
        let s_len = batch.stream_bytes.len() as u64;
        let c_len = (batch.chunks.len() * size_of::<ChunkDescGpu>()) as u64;

        let mut b_pre = [
            transition_barrier(&s.streams_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
            transition_barrier(&s.chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
            transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ];
        cmd.ResourceBarrier(&b_pre);
        drop_barriers(&mut b_pre);
        cmd.CopyBufferRegion(&s.streams_default, 0, &s.streams_upload, 0, s_len);
        cmd.CopyBufferRegion(&s.chunks_default, 0, &s.chunks_upload, 0, c_len);
        let mut b_post = [
            transition_barrier(&s.streams_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition_barrier(&s.chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        ];
        cmd.ResourceBarrier(&b_post);
        drop_barriers(&mut b_post);

        cmd.SetComputeRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);
        cmd.SetComputeRootShaderResourceView(0, self.models_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, self.window_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.quads_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(3, s.streams_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(4, s.chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(5, s.out_blocks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(6, s.status_default.GetGPUVirtualAddress());

        if let Some(qh) = ts_heap {
            cmd.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, ts_start_idx);
        }
        cmd.Dispatch((batch.chunks.len() / 2) as u32, 1, 1);
        if let Some(qh) = ts_heap {
            cmd.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, ts_end_idx);
        }
        let mut b_end = [
            transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
        ];
        cmd.ResourceBarrier(&b_end);
        drop_barriers(&mut b_end);
    }

    pub unsafe fn record_upload_and_dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        batch: &BatchPayload,
        ts_heap: Option<&ID3D12QueryHeap>,
        ts_start_idx: u32,
        ts_end_idx: u32,
    ) {
        self.record_slot_upload_and_dispatch(0, cmd, batch, ts_heap, ts_start_idx, ts_end_idx);
    }

    /// Submits BC1 decode onto `ctx.compute_queue` for Ping-Pong slot (0 or 1); returns Compute fence value.
    pub fn submit_slot_async_compute(&mut self, ctx: &mut GpuContext, slot: usize, batch: &BatchPayload) -> u64 {
        let idx = slot & 1;
        unsafe {
            let prev = self.slots[idx].last_compute_fence;
            if prev > 0 {
                let _ = ctx.wait_compute_cpu(prev, INFINITE);
            }
            self.upload_batch_slot(idx, batch);
            let s = &mut self.slots[idx];
            s.cmd_alloc.Reset().expect("slot cmd_alloc Reset");
            s.cmd_list.Reset(&s.cmd_alloc, None).expect("slot cmd_list Reset");
            self.record_slot_upload_and_dispatch(idx, &self.slots[idx].cmd_list, batch, Some(&self.slots[idx].ts_query_heap), 0, 1);
            let s = &mut self.slots[idx];
            s.cmd_list.ResolveQueryData(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &s.ts_readback, 0);
            s.cmd_list.Close().expect("Close");

            let lists = [Some(s.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            let val = ctx.signal_compute();
            s.last_compute_fence = val;
            val
        }
    }

    pub fn submit_async_compute(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> u64 {
        self.submit_slot_async_compute(ctx, 0, batch)
    }

    pub fn read_slot_kernel_ms(&self, ctx: &GpuContext, slot: usize) -> f64 {
        unsafe {
            let s = &self.slots[slot & 1];
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_ts = D3D12_RANGE { Begin: 0, End: 16 };
            s.ts_readback.Map(0, Some(&r_ts), Some(&mut ptr)).expect("Map ts");
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let ticks = ts[1].saturating_sub(ts[0]);
            s.ts_readback.Unmap(0, None);
            (ticks as f64 / ctx.compute_timestamp_freq as f64) * 1000.0
        }
    }

    pub fn read_last_kernel_ms(&self, ctx: &GpuContext) -> f64 {
        self.read_slot_kernel_ms(ctx, 0)
    }

    pub fn decode_batch_and_readback(
        &mut self,
        ctx: &mut GpuContext,
        batch: &BatchPayload,
    ) -> (Vec<u8>, Vec<ChunkStatusGpu>, f64, f64) {
        unsafe {
            let t0 = std::time::Instant::now();
            if self.slots[0].last_compute_fence > 0 {
                let _ = ctx.wait_compute_cpu(self.slots[0].last_compute_fence, INFINITE);
            }
            self.upload_batch_slot(0, batch);

            {
                let s = &self.slots[0];
                s.cmd_alloc.Reset().expect("cmd_alloc Reset");
                s.cmd_list.Reset(&s.cmd_alloc, None).expect("cmd_list Reset");
            }

            self.record_slot_upload_and_dispatch(0, &self.slots[0].cmd_list, batch, Some(&self.slots[0].ts_query_heap), 0, 1);

            let out_bytes = (batch.total_blocks * 8) as u64;
            let status_bytes = (batch.chunks.len() * size_of::<ChunkStatusGpu>()) as u64;
            let s = &mut self.slots[0];
            let mut b_to_copy = [
                transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ];
            s.cmd_list.ResourceBarrier(&b_to_copy);
            drop_barriers(&mut b_to_copy);
            s.cmd_list.CopyBufferRegion(&self.out_readback, 0, &s.out_blocks_default, 0, out_bytes);
            s.cmd_list.CopyBufferRegion(&self.status_readback, 0, &s.status_default, 0, status_bytes);
            s.cmd_list.ResolveQueryData(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &s.ts_readback, 0);

            let mut b_restore = [
                transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            ];
            s.cmd_list.ResourceBarrier(&b_restore);
            drop_barriers(&mut b_restore);
            s.cmd_list.Close().expect("Close");

            let lists = [Some(s.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();
            s.last_compute_fence = ctx.compute_fence_value;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_out = D3D12_RANGE { Begin: 0, End: out_bytes as usize };
            self.out_readback.Map(0, Some(&r_out), Some(&mut ptr)).expect("Map out");
            let decoded = std::slice::from_raw_parts(ptr as *const u8, out_bytes as usize).to_vec();
            self.out_readback.Unmap(0, None);

            let r_st = D3D12_RANGE { Begin: 0, End: status_bytes as usize };
            self.status_readback.Map(0, Some(&r_st), Some(&mut ptr)).expect("Map status");
            let statuses = std::slice::from_raw_parts(ptr as *const ChunkStatusGpu, batch.real_chunk_count).to_vec();
            self.status_readback.Unmap(0, None);

            let kernel_ms = self.read_slot_kernel_ms(ctx, 0);
            (decoded, statuses, kernel_ms, total_ms)
        }
    }

    pub fn bench_gpu_kernel_only(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let fence_val = self.submit_slot_async_compute(ctx, 0, batch);
        let _ = ctx.wait_compute_cpu(fence_val, INFINITE);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kernel_ms = self.read_slot_kernel_ms(ctx, 0);
        (kernel_ms, wall_ms)
    }
}

pub struct Bc7RansGpuDecoder {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    models_buf: ID3D12Resource,

    cap_stream_bytes: usize,
    cap_chunks: usize,
    cap_sym_rows: usize,

    streams_upload: ID3D12Resource,
    streams_default: ID3D12Resource,
    chunks_upload: ID3D12Resource,
    chunks_default: ID3D12Resource,
    pub out_syms_default: ID3D12Resource,
    status_default: ID3D12Resource,

    out_readback: ID3D12Resource,
    status_readback: ID3D12Resource,
    ts_query_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,

    pub cmd_alloc: ID3D12CommandAllocator,
    pub cmd_list: ID3D12GraphicsCommandList,
}

impl Bc7RansGpuDecoder {
    pub fn new(
        ctx: &mut GpuContext,
        max_chunks: usize,
        max_sym_rows: usize,
        max_stream_bytes: usize,
    ) -> Self {
        unsafe {
            // t0 = g_Models, t1 = g_Streams, t2 = g_Chunks, u0 = g_OutSyms, u1 = g_Status
            let mut params = [D3D12_ROOT_PARAMETER1::default(); 5];
            for i in 0..3 {
                params[i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }
            for i in 0..2 {
                params[3 + i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }
            let root_sig = create_root_sig_from_params(&ctx.device, &params);
            let pso = create_compute_pso(&ctx.device, &root_sig, BC7_RANS_CS_DXIL, "bcpack_bc7_rans.dxil");

            let models = build_packed_models();
            let models_size = (models.len() * size_of::<ModelDataGpu>()) as u64;
            let models_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&models_up, &models);
            let models_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

            let cmd_alloc: ID3D12CommandAllocator = ctx
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .expect("CreateCommandAllocator BC7 COMPUTE");
            let cmd_list: ID3D12GraphicsCommandList = ctx
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None)
                .expect("CreateCommandList BC7 COMPUTE");
            cmd_list.CopyBufferRegion(&models_buf, 0, &models_up, 0, models_size);
            cmd_barriers(&cmd_list, vec![transition_barrier(&models_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
            cmd_list.Close().unwrap();
            let lists = [Some(cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();

            let streams_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let streams_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let padded_max_chunks = max_chunks + 2;
            let chunks_bytes = (padded_max_chunks * size_of::<ChunkDescGpu>()) as u64;
            let chunks_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let chunks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            // 16 x uint16 symbols = 32 bytes per symbol-row
            let out_bytes = (max_sym_rows * 32) as u64;
            let out_syms_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let out_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let status_bytes = (padded_max_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let status_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, status_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let status_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, status_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            ctx.device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap BC7");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            Self {
                root_sig,
                pso,
                models_buf,
                cap_stream_bytes: max_stream_bytes,
                cap_chunks: padded_max_chunks,
                cap_sym_rows: max_sym_rows,
                streams_upload,
                streams_default,
                chunks_upload,
                chunks_default,
                out_syms_default,
                status_default,
                out_readback,
                status_readback,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
            }
        }
    }

    pub unsafe fn upload_batch(&self, batch: &BatchPayload) {
        assert!(batch.stream_bytes.len() <= self.cap_stream_bytes, "BC7 stream capacity exceeded");
        assert!(batch.chunks.len() <= self.cap_chunks, "BC7 chunks capacity exceeded");
        assert!(batch.total_blocks <= self.cap_sym_rows, "BC7 symbol-row capacity exceeded");
        upload_slice(&self.streams_upload, &batch.stream_bytes);
        upload_slice(&self.chunks_upload, &batch.chunks);
    }

    unsafe fn record_dispatch(&self, cmd: &ID3D12GraphicsCommandList, batch: &BatchPayload) {
        let s_len = batch.stream_bytes.len() as u64;
        let c_len = (batch.chunks.len() * size_of::<ChunkDescGpu>()) as u64;

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.streams_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        cmd.CopyBufferRegion(&self.streams_default, 0, &self.streams_upload, 0, s_len);
        cmd.CopyBufferRegion(&self.chunks_default, 0, &self.chunks_upload, 0, c_len);
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.streams_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        cmd.SetComputeRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);
        cmd.SetComputeRootShaderResourceView(0, self.models_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, self.streams_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, self.out_syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(4, self.status_default.GetGPUVirtualAddress());

        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        cmd.Dispatch((batch.chunks.len() / 2) as u32, 1, 1);
        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }

    pub fn submit_async_compute(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> u64 {
        unsafe {
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn submit_async_to_buffer(
        &mut self,
        ctx: &mut GpuContext,
        batch: &BatchPayload,
        dst: &ID3D12Resource,
        dst_offset: u64,
    ) -> u64 {
        unsafe {
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);

            let out_bytes = (batch.total_blocks * 32) as u64;
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            self.cmd_list.CopyBufferRegion(dst, dst_offset, &self.out_syms_default, 0, out_bytes);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn read_last_kernel_ms(&self, ctx: &GpuContext) -> f64 {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_ts = D3D12_RANGE { Begin: 0, End: 16 };
            self.ts_readback.Map(0, Some(&r_ts), Some(&mut ptr)).unwrap();
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let ticks = ts[1].saturating_sub(ts[0]);
            self.ts_readback.Unmap(0, None);
            (ticks as f64 / ctx.compute_timestamp_freq as f64) * 1000.0
        }
    }

    pub fn decode_batch_and_readback(
        &mut self,
        ctx: &mut GpuContext,
        batch: &BatchPayload,
    ) -> (Vec<u16>, Vec<ChunkStatusGpu>, f64, f64) {
        unsafe {
            let t0 = std::time::Instant::now();
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);

            let out_bytes = (batch.total_blocks * 32) as u64;
            let status_bytes = (batch.chunks.len() * size_of::<ChunkStatusGpu>()) as u64;
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                ],
            );
            self.cmd_list.CopyBufferRegion(&self.out_readback, 0, &self.out_syms_default, 0, out_bytes);
            self.cmd_list.CopyBufferRegion(&self.status_readback, 0, &self.status_default, 0, status_bytes);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_syms_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_out = D3D12_RANGE { Begin: 0, End: out_bytes as usize };
            self.out_readback.Map(0, Some(&r_out), Some(&mut ptr)).unwrap();
            let decoded = std::slice::from_raw_parts(ptr as *const u16, batch.total_blocks * 16).to_vec();
            self.out_readback.Unmap(0, None);

            let r_st = D3D12_RANGE { Begin: 0, End: status_bytes as usize };
            self.status_readback.Map(0, Some(&r_st), Some(&mut ptr)).unwrap();
            let statuses = std::slice::from_raw_parts(ptr as *const ChunkStatusGpu, batch.real_chunk_count).to_vec();
            self.status_readback.Unmap(0, None);

            (decoded, statuses, self.read_last_kernel_ms(ctx), total_ms)
        }
    }

    pub fn bench_gpu_kernel_only(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let fence_val = self.submit_async_compute(ctx, batch);
        let _ = ctx.wait_compute_cpu(fence_val, INFINITE);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        (self.read_last_kernel_ms(ctx), wall_ms)
    }
}

pub struct Bc16SlotResources {
    pub streams_upload: ID3D12Resource,
    pub streams_default: ID3D12Resource,
    pub chunks_upload: ID3D12Resource,
    pub chunks_default: ID3D12Resource,
    pub syms_default: ID3D12Resource,
    pub out_blocks_default: ID3D12Resource,
    pub status_default: ID3D12Resource,
    pub ts_query_heap: ID3D12QueryHeap,
    pub ts_readback: ID3D12Resource,
    pub cmd_alloc: ID3D12CommandAllocator,
    pub cmd_list: ID3D12GraphicsCommandList,
    pub last_compute_fence: u64,
}

impl Bc16SlotResources {
    fn new(
        device: &ID3D12Device,
        max_stream_bytes: usize,
        padded_max_chunks: usize,
        max_rows: usize,
    ) -> Self {
        unsafe {
            let streams_upload = create_committed_buffer(device, D3D12_HEAP_TYPE_UPLOAD, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let streams_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let chunks_bytes = (padded_max_chunks * size_of::<ChunkDescGpu>()) as u64;
            let chunks_upload = create_committed_buffer(device, D3D12_HEAP_TYPE_UPLOAD, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let chunks_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, chunks_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let syms_bytes = (max_rows * 32) as u64;
            let syms_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, syms_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let out_bytes = (max_rows * 8) as u64;
            let out_blocks_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let status_bytes = (padded_max_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let status_default = create_committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, status_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap BC16");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let cmd_alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .expect("CreateCommandAllocator BC16 COMPUTE");
            let cmd_list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None)
                .expect("CreateCommandList BC16 COMPUTE");
            cmd_list.Close().expect("Close initial BC16 slot cmd_list");

            Self {
                streams_upload,
                streams_default,
                chunks_upload,
                chunks_default,
                syms_default,
                out_blocks_default,
                status_default,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
                last_compute_fence: 0,
            }
        }
    }
}

/// Ping-ponged Async Compute GPU decompressor for 16-byte block textures (BC7 & BC5):
/// runs 16-lane rANS (`bcpack_bc7_rans.dxil`) + fused block packer (`bc16_pack.dxil`).
pub struct Bc16GpuStreamer {
    rans_root_sig: ID3D12RootSignature,
    rans_pso: ID3D12PipelineState,
    pack_root_sig: ID3D12RootSignature,
    pack_pso: ID3D12PipelineState,
    models_buf: ID3D12Resource,

    cap_stream_bytes: usize,
    cap_chunks: usize,
    cap_rows: usize,

    pub slots: [Bc16SlotResources; 2],
    out_readback: ID3D12Resource,
    status_readback: ID3D12Resource,
}

impl Bc16GpuStreamer {
    pub fn new(
        ctx: &mut GpuContext,
        max_chunks: usize,
        max_rows: usize,
        max_stream_bytes: usize,
    ) -> Self {
        unsafe {
            let mut rans_params = [D3D12_ROOT_PARAMETER1::default(); 5];
            for i in 0..3 {
                rans_params[i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }
            for i in 0..2 {
                rans_params[3 + i] = D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: i as u32,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                };
            }
            let rans_root_sig = create_root_sig_from_params(&ctx.device, &rans_params);
            let rans_pso = create_compute_pso(&ctx.device, &rans_root_sig, BC7_RANS_CS_DXIL, "bcpack_bc7_rans.dxil");

            let pack_params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let pack_root_sig = create_root_sig_from_params(&ctx.device, &pack_params);
            let pack_pso = create_compute_pso(&ctx.device, &pack_root_sig, BC16_PACK_CS_DXIL, "bc16_pack.dxil");

            let models = build_packed_models();
            let models_size = (models.len() * size_of::<ModelDataGpu>()) as u64;
            let models_up = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload_slice(&models_up, &models);
            let models_buf = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, models_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);

            let padded_max_chunks = max_chunks + 2;
            let slots = [
                Bc16SlotResources::new(&ctx.device, max_stream_bytes, padded_max_chunks, max_rows),
                Bc16SlotResources::new(&ctx.device, max_stream_bytes, padded_max_chunks, max_rows),
            ];

            slots[0].cmd_alloc.Reset().unwrap();
            slots[0].cmd_list.Reset(&slots[0].cmd_alloc, None).unwrap();
            slots[0].cmd_list.CopyBufferRegion(&models_buf, 0, &models_up, 0, models_size);
            cmd_barriers(&slots[0].cmd_list, vec![transition_barrier(&models_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
            slots[0].cmd_list.Close().unwrap();
            let lists = [Some(slots[0].cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();

            let out_bytes = (max_rows * 8) as u64;
            let status_bytes = (padded_max_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let out_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            let status_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, status_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            Self {
                rans_root_sig,
                rans_pso,
                pack_root_sig,
                pack_pso,
                models_buf,
                cap_stream_bytes: max_stream_bytes,
                cap_chunks: padded_max_chunks,
                cap_rows: max_rows,
                slots,
                out_readback,
                status_readback,
            }
        }
    }

    pub fn out_buffer(&self, slot: usize) -> &ID3D12Resource {
        &self.slots[slot & 1].out_blocks_default
    }

    pub unsafe fn upload_batch_slot(&self, slot: usize, batch: &BatchPayload) {
        assert!(batch.stream_bytes.len() <= self.cap_stream_bytes, "BC16 stream capacity exceeded");
        assert!(batch.chunks.len() <= self.cap_chunks, "BC16 chunks capacity exceeded");
        assert!(batch.total_blocks <= self.cap_rows, "BC16 row capacity exceeded");
        let s = &self.slots[slot & 1];
        upload_slice(&s.streams_upload, &batch.stream_bytes);
        upload_slice(&s.chunks_upload, &batch.chunks);
    }

    unsafe fn record_slot_dispatch(&self, slot: usize, cmd: &ID3D12GraphicsCommandList, batch: &BatchPayload) {
        let s = &self.slots[slot & 1];
        let s_len = batch.stream_bytes.len() as u64;
        let c_len = (batch.chunks.len() * size_of::<ChunkDescGpu>()) as u64;

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&s.streams_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&s.chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&s.syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        cmd.CopyBufferRegion(&s.streams_default, 0, &s.streams_upload, 0, s_len);
        cmd.CopyBufferRegion(&s.chunks_default, 0, &s.chunks_upload, 0, c_len);
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&s.streams_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&s.chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        cmd.EndQuery(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

        cmd.SetComputeRootSignature(&self.rans_root_sig);
        cmd.SetPipelineState(&self.rans_pso);
        cmd.SetComputeRootShaderResourceView(0, self.models_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, s.streams_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, s.chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, s.syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(4, s.status_default.GetGPUVirtualAddress());
        cmd.Dispatch((batch.chunks.len() / 2) as u32, 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&s.syms_default),
                transition_barrier(&s.syms_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        let total_rows = batch.total_blocks as u32;
        let pack_cb = [total_rows, 0u32, 0u32, 0u32];
        cmd.SetComputeRootSignature(&self.pack_root_sig);
        cmd.SetPipelineState(&self.pack_pso);
        cmd.SetComputeRoot32BitConstants(0, 4, pack_cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, s.syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, s.out_blocks_default.GetGPUVirtualAddress());
        cmd.Dispatch(total_rows.div_ceil(64).max(1), 1, 1);

        cmd.EndQuery(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&s.syms_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }

    pub fn submit_slot_async_compute(&mut self, ctx: &mut GpuContext, slot: usize, batch: &BatchPayload) -> u64 {
        let idx = slot & 1;
        unsafe {
            let prev = self.slots[idx].last_compute_fence;
            if prev > 0 {
                let _ = ctx.wait_compute_cpu(prev, INFINITE);
            }
            self.upload_batch_slot(idx, batch);
            let s = &mut self.slots[idx];
            s.cmd_alloc.Reset().expect("bc16 slot cmd_alloc Reset");
            s.cmd_list.Reset(&s.cmd_alloc, None).expect("bc16 slot cmd_list Reset");
            self.record_slot_dispatch(idx, &self.slots[idx].cmd_list, batch);
            let s = &mut self.slots[idx];
            s.cmd_list.ResolveQueryData(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &s.ts_readback, 0);
            s.cmd_list.Close().expect("Close bc16 slot");

            let lists = [Some(s.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            let val = ctx.signal_compute();
            s.last_compute_fence = val;
            val
        }
    }

    pub fn read_slot_kernel_ms(&self, ctx: &GpuContext, slot: usize) -> f64 {
        unsafe {
            let s = &self.slots[slot & 1];
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_ts = D3D12_RANGE { Begin: 0, End: 16 };
            s.ts_readback.Map(0, Some(&r_ts), Some(&mut ptr)).expect("Map bc16 ts");
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let ticks = ts[1].saturating_sub(ts[0]);
            s.ts_readback.Unmap(0, None);
            (ticks as f64 / ctx.compute_timestamp_freq as f64) * 1000.0
        }
    }

    pub fn decode_batch_and_readback(
        &mut self,
        ctx: &mut GpuContext,
        batch: &BatchPayload,
    ) -> (Vec<u8>, Vec<ChunkStatusGpu>, f64, f64) {
        unsafe {
            let t0 = std::time::Instant::now();
            if self.slots[0].last_compute_fence > 0 {
                let _ = ctx.wait_compute_cpu(self.slots[0].last_compute_fence, INFINITE);
            }
            self.upload_batch_slot(0, batch);

            {
                let s = &self.slots[0];
                s.cmd_alloc.Reset().unwrap();
                s.cmd_list.Reset(&s.cmd_alloc, None).unwrap();
            }
            self.record_slot_dispatch(0, &self.slots[0].cmd_list, batch);

            let out_bytes = (batch.total_blocks * 8) as u64;
            let status_bytes = (batch.chunks.len() * size_of::<ChunkStatusGpu>()) as u64;
            let s = &mut self.slots[0];
            cmd_barriers(
                &s.cmd_list,
                vec![
                    transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                ],
            );
            s.cmd_list.CopyBufferRegion(&self.out_readback, 0, &s.out_blocks_default, 0, out_bytes);
            s.cmd_list.CopyBufferRegion(&self.status_readback, 0, &s.status_default, 0, status_bytes);
            s.cmd_list.ResolveQueryData(&s.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &s.ts_readback, 0);
            cmd_barriers(
                &s.cmd_list,
                vec![
                    transition_barrier(&s.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&s.status_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            s.cmd_list.Close().unwrap();

            let lists = [Some(s.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();
            s.last_compute_fence = ctx.compute_fence_value;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            let r_out = D3D12_RANGE { Begin: 0, End: out_bytes as usize };
            self.out_readback.Map(0, Some(&r_out), Some(&mut ptr)).unwrap();
            let decoded = std::slice::from_raw_parts(ptr as *const u8, out_bytes as usize).to_vec();
            self.out_readback.Unmap(0, None);

            let r_st = D3D12_RANGE { Begin: 0, End: status_bytes as usize };
            self.status_readback.Map(0, Some(&r_st), Some(&mut ptr)).unwrap();
            let statuses = std::slice::from_raw_parts(ptr as *const ChunkStatusGpu, batch.real_chunk_count).to_vec();
            self.status_readback.Unmap(0, None);

            let kernel_ms = self.read_slot_kernel_ms(ctx, 0);
            (decoded, statuses, kernel_ms, total_ms)
        }
    }
}

pub struct Gfx10DetileGpu {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    cap_bytes: usize,
    src_upload: ID3D12Resource,
    src_default: ID3D12Resource,
    dst_default: ID3D12Resource,
    dst_readback: ID3D12Resource,
    ts_query_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,
    cmd_alloc: ID3D12CommandAllocator,
    cmd_list: ID3D12GraphicsCommandList,
}

impl Gfx10DetileGpu {
    pub fn new(ctx: &GpuContext, max_buffer_bytes: usize) -> Self {
        unsafe {
            let params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 8,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let root_sig = create_root_sig_from_params(&ctx.device, &params);
            let pso = create_compute_pso(&ctx.device, &root_sig, GFX10_DETILE_CS_DXIL, "gfx10_detile.dxil");

            let cap = max_buffer_bytes as u64;
            let src_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, cap, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let src_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, cap, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            let dst_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, cap, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let dst_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, cap, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            ctx.device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap Detile");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let cmd_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let cmd_list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None).unwrap();
            cmd_list.Close().unwrap();

            Self {
                root_sig,
                pso,
                cap_bytes: max_buffer_bytes,
                src_upload,
                src_default,
                dst_default,
                dst_readback,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
            }
        }
    }

    pub unsafe fn record_detile(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        src_gpu_va: u64,
        dst_gpu_va: u64,
        width_blocks: u32,
        height_blocks: u32,
        tile_mode: u32,
        block_bytes: u32,
        src_byte_offset: u32,
        dst_byte_offset: u32,
        total_swizzled_blocks: u32,
    ) {
        let constants = [
            width_blocks,
            height_blocks,
            tile_mode,
            block_bytes,
            src_byte_offset,
            dst_byte_offset,
            total_swizzled_blocks,
            0u32,
        ];
        cmd.SetComputeRootSignature(&self.root_sig);
        cmd.SetPipelineState(&self.pso);
        cmd.SetComputeRoot32BitConstants(0, 8, constants.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, src_gpu_va);
        cmd.SetComputeRootUnorderedAccessView(2, dst_gpu_va);
        cmd.Dispatch((total_swizzled_blocks + 63) / 64, 1, 1);
    }

    pub fn detile_and_readback(
        &self,
        ctx: &mut GpuContext,
        swizzled_bytes: &[u8],
        width_blocks: u32,
        height_blocks: u32,
        tile_mode: u32,
        block_bytes: u32,
    ) -> (Vec<u8>, f64) {
        unsafe {
            let total_swizzled = (swizzled_bytes.len() / block_bytes as usize) as u32;
            let linear_bytes = (width_blocks as usize) * (height_blocks as usize) * (block_bytes as usize);
            assert!(swizzled_bytes.len() <= self.cap_bytes && linear_bytes <= self.cap_bytes);
            upload_slice(&self.src_upload, swizzled_bytes);

            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.src_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                    transition_barrier(&self.dst_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                ],
            );
            self.cmd_list.CopyBufferRegion(&self.src_default, 0, &self.src_upload, 0, swizzled_bytes.len() as u64);
            cmd_barriers(
                &self.cmd_list,
                vec![transition_barrier(&self.src_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)],
            );

            self.cmd_list.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            self.record_detile(
                &self.cmd_list,
                self.src_default.GetGPUVirtualAddress(),
                self.dst_default.GetGPUVirtualAddress(),
                width_blocks,
                height_blocks,
                tile_mode,
                block_bytes,
                0,
                0,
                total_swizzled,
            );
            self.cmd_list.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

            cmd_barriers(
                &self.cmd_list,
                vec![transition_barrier(&self.dst_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)],
            );
            self.cmd_list.CopyBufferRegion(&self.dst_readback, 0, &self.dst_default, 0, linear_bytes as u64);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![transition_barrier(&self.dst_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)],
            );
            self.cmd_list.Close().unwrap();

            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();

            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.dst_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: linear_bytes }), Some(&mut ptr)).unwrap();
            let out = std::slice::from_raw_parts(ptr as *const u8, linear_bytes).to_vec();
            self.dst_readback.Unmap(0, None);

            self.ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let kms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.compute_timestamp_freq as f64) * 1000.0;
            self.ts_readback.Unmap(0, None);

            (out, kms)
        }
    }

    pub fn bench_detile_batch_kernel_ms(
        &self,
        ctx: &mut GpuContext,
        swizzled_bytes_len: usize,
        width_blocks: u32,
        height_blocks: u32,
        tile_mode: u32,
        block_bytes: u32,
        repeat_textures: u32,
    ) -> f64 {
        unsafe {
            let tex_blocks = swizzled_bytes_len as u32 / block_bytes;
            let total_bytes = (swizzled_bytes_len as usize) * (repeat_textures as usize);
            assert!(total_bytes <= self.cap_bytes);

            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            cmd_barriers(
                &self.cmd_list,
                vec![transition_barrier(&self.dst_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
            );
            self.cmd_list.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            let src_va = self.src_default.GetGPUVirtualAddress();
            let dst_va = self.dst_default.GetGPUVirtualAddress();
            for r in 0..repeat_textures {
                let off = r * (swizzled_bytes_len as u32);
                self.record_detile(&self.cmd_list, src_va, dst_va, width_blocks, height_blocks, tile_mode, block_bytes, off, off, tex_blocks);
            }
            self.cmd_list.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![transition_barrier(&self.dst_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)],
            );
            self.cmd_list.Close().unwrap();

            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();

            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let kms = (ts[1].saturating_sub(ts[0]) as f64 / ctx.compute_timestamp_freq as f64) * 1000.0;
            self.ts_readback.Unmap(0, None);
            kms
        }
    }
}

unsafe fn upload_static_srv<T: Copy>(ctx: &mut GpuContext, data: &[T]) -> ID3D12Resource {
    let bytes = (data.len() * size_of::<T>()).max(256) as u64;
    let up = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_UPLOAD,
        bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    upload_slice(&up, data);
    let dst = create_committed_buffer(
        &ctx.device,
        D3D12_HEAP_TYPE_DEFAULT,
        bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let alloc: ID3D12CommandAllocator = ctx
        .device
        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
        .expect("CreateCommandAllocator static SRV");
    let cmd: ID3D12GraphicsCommandList = ctx
        .device
        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, None)
        .expect("CreateCommandList static SRV");
    cmd.CopyBufferRegion(&dst, 0, &up, 0, bytes);
    cmd_barriers(
        &cmd,
        vec![transition_barrier(
            &dst,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )],
    );
    cmd.Close().unwrap();
    let lists = [Some(cmd.cast::<ID3D12CommandList>().unwrap())];
    ctx.compute_queue.ExecuteCommandLists(&lists);
    ctx.signal_and_wait_compute();
    dst
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BcnChunkDescGpu {
    pub stream_offset: u32,
    pub stream_len: u32,
    pub rows: u32,
    pub out_row_offset: u32,
    pub mode: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BcnRebuildChunkGpu {
    pub in_row_offset: u32,
    pub blocks: u32,
    pub out_byte_offset: u32,
    pub mode: u32,
}

pub struct BcnBatchPayload {
    pub stream_bytes: Vec<u8>,
    pub rans_chunks: Vec<BcnChunkDescGpu>,
    pub rebuild_chunks: Vec<BcnRebuildChunkGpu>,
    pub real_chunk_count: usize,
    pub total_rows: usize,
    pub total_out_bytes: usize,
}

#[inline]
pub fn bcn_mode_id(mode: BcpMode) -> u32 {
    match mode {
        BcpMode::Bc3 => 0,
        BcpMode::Bc4 => 1,
        BcpMode::Bc5 => 2,
        other => panic!("BcnGpuDecoder supports only BC3/BC4/BC5, got {:?}", other),
    }
}

/// Packs `(compressed_stream, mode, num_blocks)` items for `BcnGpuDecoder` (BC3 / BC4 / BC5).
pub fn pack_bcn_batch<'a, I>(items: I) -> BcnBatchPayload
where
    I: IntoIterator<Item = (&'a [u8], BcpMode, usize)>,
{
    let mut stream_bytes = Vec::new();
    let mut rans_chunks = Vec::new();
    let mut rebuild_chunks = Vec::new();
    let mut total_rows = 0usize;
    let mut total_out_bytes = 0usize;

    for (stream, mode, blocks) in items {
        while stream_bytes.len() % 4 != 0 {
            stream_bytes.push(0);
        }
        let m_id = bcn_mode_id(mode);
        let rows = blocks * mode.symbols_per_block();
        let out_bytes = blocks * mode.block_bytes();
        let offset = stream_bytes.len() as u32;
        stream_bytes.extend_from_slice(stream);

        rans_chunks.push(BcnChunkDescGpu {
            stream_offset: offset,
            stream_len: stream.len() as u32,
            rows: rows as u32,
            out_row_offset: total_rows as u32,
            mode: m_id,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
        rebuild_chunks.push(BcnRebuildChunkGpu {
            in_row_offset: total_rows as u32,
            blocks: blocks as u32,
            out_byte_offset: total_out_bytes as u32,
            mode: m_id,
        });
        total_rows += rows;
        total_out_bytes += out_bytes;
    }

    let real_chunk_count = rans_chunks.len();
    if rans_chunks.len() % 2 == 1 {
        rans_chunks.push(BcnChunkDescGpu {
            stream_offset: 0,
            stream_len: 0,
            rows: 0,
            out_row_offset: 0,
            mode: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
    }
    while stream_bytes.len() % 4 != 0 {
        stream_bytes.push(0);
    }
    stream_bytes.resize(stream_bytes.len() + 256, 0);

    BcnBatchPayload {
        stream_bytes,
        rans_chunks,
        rebuild_chunks,
        real_chunk_count,
        total_rows,
        total_out_bytes,
    }
}

fn pack_bcn_tables() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let modes = [BcpMode::Bc3, BcpMode::Bc4, BcpMode::Bc5];
    let mut pairs = vec![0u32; 3 * 64];
    let mut quads = vec![0u32; 256];
    let mut dict = vec![0u32; 3 * 255 * 6];

    for (mi, &m) in modes.iter().enumerate() {
        let t = bcp_bcn::Tables::default_for(m).expect("embedded BCn table");
        for s in 0..64 {
            pairs[mi * 64 + s] = (t.pairs.pairs[s][0] as u32) | ((t.pairs.pairs[s][1] as u32) << 8);
        }
        if m == BcpMode::Bc3 {
            for s in 0..256 {
                let q = t.quads.quads[s];
                quads[s] = (q[0] as u32)
                    | ((q[1] as u32) << 8)
                    | ((q[2] as u32) << 16)
                    | ((q[3] as u32) << 24);
            }
        }
        for (k, (st, _)) in t.dictionary.iter().enumerate() {
            let base = (mi * 255 + k) * 6;
            dict[base] = (st.ep[0] as u32)
                | ((st.ep[1] as u32) << 8)
                | ((st.ep[2] as u32) << 16)
                | ((st.ep[3] as u32) << 24);
            dict[base + 1] = (st.ep[4] as u32)
                | ((st.ep[5] as u32) << 8)
                | ((st.ep[6] as u32) << 16)
                | ((st.ep[7] as u32) << 24);
            for p in 0..32 {
                dict[base + 2 + (p >> 3)] |= ((st.ix[p] & 0xF) as u32) << ((p & 7) * 4);
            }
        }
    }
    (pairs, quads, dict)
}

/// Async Compute GPU Decoder for BC3 (`0x0C`), BC4 (`0x10`), and BC5 (`0x14`).
pub struct BcnGpuDecoder {
    rans_root_sig: ID3D12RootSignature,
    rans_pso: ID3D12PipelineState,
    rebuild_root_sig: ID3D12RootSignature,
    rebuild_pso: ID3D12PipelineState,

    models_buf: ID3D12Resource,
    pairs_buf: ID3D12Resource,
    quads_buf: ID3D12Resource,
    dict_buf: ID3D12Resource,

    cap_stream_bytes: usize,
    cap_chunks: usize,
    cap_sym_rows: usize,
    cap_out_bytes: usize,

    streams_upload: ID3D12Resource,
    streams_default: ID3D12Resource,
    rans_chunks_upload: ID3D12Resource,
    rans_chunks_default: ID3D12Resource,
    rebuild_chunks_upload: ID3D12Resource,
    rebuild_chunks_default: ID3D12Resource,
    syms_default: ID3D12Resource,
    hist_default: ID3D12Resource,
    status_default: ID3D12Resource,
    pub out_blocks_default: ID3D12Resource,

    out_readback: ID3D12Resource,
    status_readback: ID3D12Resource,
    ts_query_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,

    cmd_alloc: ID3D12CommandAllocator,
    cmd_list: ID3D12GraphicsCommandList,
}

fn srv_param(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn uav_param(reg: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn cb_param(reg: u32, num_dwords: u32) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: reg,
                RegisterSpace: 0,
                Num32BitValues: num_dwords,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

impl BcnGpuDecoder {
    pub fn new(
        ctx: &mut GpuContext,
        max_chunks: usize,
        max_sym_rows: usize,
        max_out_bytes: usize,
        max_stream_bytes: usize,
    ) -> Self {
        unsafe {
            let rans_params = [srv_param(0), srv_param(1), srv_param(2), uav_param(0), uav_param(1)];
            let rans_root_sig = create_root_sig_from_params(&ctx.device, &rans_params);
            let rans_pso = create_compute_pso(&ctx.device, &rans_root_sig, BCN_RANS_CS_DXIL, "bcpack_bcn_rans.dxil");

            let rebuild_params = [
                cb_param(0, 4),
                srv_param(0),
                srv_param(1),
                srv_param(2),
                srv_param(3),
                srv_param(4),
                uav_param(0),
                uav_param(1),
            ];
            let rebuild_root_sig = create_root_sig_from_params(&ctx.device, &rebuild_params);
            let rebuild_pso = create_compute_pso(&ctx.device, &rebuild_root_sig, BCN_REBUILD_CS_DXIL, "bcpack_bcn_rebuild.dxil");

            let models = build_packed_models();
            let (pairs, quads, dict) = pack_bcn_tables();
            let models_buf = upload_static_srv(ctx, &models);
            let pairs_buf = upload_static_srv(ctx, &pairs);
            let quads_buf = upload_static_srv(ctx, &quads);
            let dict_buf = upload_static_srv(ctx, &dict);

            let padded_chunks = max_chunks + 2;
            let streams_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let streams_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, max_stream_bytes as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let rc_bytes = (padded_chunks * size_of::<BcnChunkDescGpu>()) as u64;
            let rans_chunks_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, rc_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let rans_chunks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, rc_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let rb_bytes = (padded_chunks * size_of::<BcnRebuildChunkGpu>()) as u64;
            let rebuild_chunks_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, rb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let rebuild_chunks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, rb_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            // 6 uints = 24 bytes per symbol row
            let syms_bytes = (max_sym_rows * 24).max(256) as u64;
            let syms_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, syms_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let hist_bytes = (padded_chunks * 256 * 6 * 4) as u64;
            let hist_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, hist_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let status_bytes = (padded_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let status_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, status_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let status_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, status_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let out_cap = max_out_bytes.max(256);
            let out_blocks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, out_cap as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let out_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_cap as u64, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            ctx.device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap Bcn");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let cmd_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let cmd_list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None).unwrap();
            cmd_list.Close().unwrap();

            Self {
                rans_root_sig,
                rans_pso,
                rebuild_root_sig,
                rebuild_pso,
                models_buf,
                pairs_buf,
                quads_buf,
                dict_buf,
                cap_stream_bytes: max_stream_bytes,
                cap_chunks: padded_chunks,
                cap_sym_rows: max_sym_rows,
                cap_out_bytes: out_cap,
                streams_upload,
                streams_default,
                rans_chunks_upload,
                rans_chunks_default,
                rebuild_chunks_upload,
                rebuild_chunks_default,
                syms_default,
                hist_default,
                status_default,
                out_blocks_default,
                out_readback,
                status_readback,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
            }
        }
    }

    unsafe fn upload_batch(&self, batch: &BcnBatchPayload) {
        assert!(batch.stream_bytes.len() <= self.cap_stream_bytes, "BCn stream capacity exceeded");
        assert!(batch.rans_chunks.len() <= self.cap_chunks, "BCn chunks capacity exceeded");
        assert!(batch.total_rows <= self.cap_sym_rows, "BCn symbol-row capacity exceeded");
        assert!(batch.total_out_bytes <= self.cap_out_bytes, "BCn output capacity exceeded");
        upload_slice(&self.streams_upload, &batch.stream_bytes);
        upload_slice(&self.rans_chunks_upload, &batch.rans_chunks);
        upload_slice(&self.rebuild_chunks_upload, &batch.rebuild_chunks);
    }

    unsafe fn record_dispatch(&self, cmd: &ID3D12GraphicsCommandList, batch: &BcnBatchPayload) {
        let s_len = batch.stream_bytes.len() as u64;
        let rc_len = (batch.rans_chunks.len() * size_of::<BcnChunkDescGpu>()) as u64;
        let rb_len = (batch.rebuild_chunks.len() * size_of::<BcnRebuildChunkGpu>()) as u64;

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.streams_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.rans_chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.rebuild_chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.hist_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        cmd.CopyBufferRegion(&self.streams_default, 0, &self.streams_upload, 0, s_len);
        cmd.CopyBufferRegion(&self.rans_chunks_default, 0, &self.rans_chunks_upload, 0, rc_len);
        cmd.CopyBufferRegion(&self.rebuild_chunks_default, 0, &self.rebuild_chunks_upload, 0, rb_len);
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.streams_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.rans_chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.rebuild_chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

        // Stage 1: 12-lane Wave32 rANS symbol decode
        cmd.SetComputeRootSignature(&self.rans_root_sig);
        cmd.SetPipelineState(&self.rans_pso);
        cmd.SetComputeRootShaderResourceView(0, self.models_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, self.streams_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.rans_chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, self.syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(4, self.status_default.GetGPUVirtualAddress());
        cmd.Dispatch((batch.rans_chunks.len() / 2) as u32, 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.syms_default),
                transition_barrier(&self.syms_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        // Stage 2: BC3 / BC4 / BC5 block rebuild (history + dict + pair/quad rank tables)
        let cb = [batch.real_chunk_count as u32, 0u32, 0u32, 0u32];
        cmd.SetComputeRootSignature(&self.rebuild_root_sig);
        cmd.SetPipelineState(&self.rebuild_pso);
        cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, self.syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.rebuild_chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(3, self.pairs_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(4, self.quads_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(5, self.dict_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(6, self.out_blocks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(7, self.hist_default.GetGPUVirtualAddress());
        cmd.Dispatch(((batch.real_chunk_count + 63) / 64).max(1) as u32, 1, 1);

        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.syms_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.hist_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }

    pub fn submit_async_compute(&mut self, ctx: &mut GpuContext, batch: &BcnBatchPayload) -> u64 {
        unsafe {
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn submit_async_to_buffer(&mut self, ctx: &mut GpuContext, batch: &BcnBatchPayload, dst: &ID3D12Resource, dst_offset: u64) -> u64 {
        unsafe {
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            self.cmd_list.CopyBufferRegion(dst, dst_offset, &self.out_blocks_default, 0, batch.total_out_bytes as u64);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn read_last_kernel_ms(&self, ctx: &GpuContext) -> f64 {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let ticks = ts[1].saturating_sub(ts[0]);
            self.ts_readback.Unmap(0, None);
            (ticks as f64 / ctx.compute_timestamp_freq as f64) * 1000.0
        }
    }

    pub fn decode_batch_and_readback(&mut self, ctx: &mut GpuContext, batch: &BcnBatchPayload) -> (Vec<u8>, Vec<ChunkStatusGpu>, f64, f64) {
        unsafe {
            let t0 = std::time::Instant::now();
            self.upload_batch(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_dispatch(&self.cmd_list, batch);
            let out_bytes = batch.total_out_bytes as u64;
            let status_bytes = (batch.real_chunk_count * size_of::<ChunkStatusGpu>()) as u64;
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                ],
            );
            self.cmd_list.CopyBufferRegion(&self.out_readback, 0, &self.out_blocks_default, 0, out_bytes);
            self.cmd_list.CopyBufferRegion(&self.status_readback, 0, &self.status_default, 0, status_bytes);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&self.status_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.out_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: batch.total_out_bytes }), Some(&mut ptr)).unwrap();
            let decoded = std::slice::from_raw_parts(ptr as *const u8, batch.total_out_bytes).to_vec();
            self.out_readback.Unmap(0, None);

            self.status_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: status_bytes as usize }), Some(&mut ptr)).unwrap();
            let statuses = std::slice::from_raw_parts(ptr as *const ChunkStatusGpu, batch.real_chunk_count).to_vec();
            self.status_readback.Unmap(0, None);

            (decoded, statuses, self.read_last_kernel_ms(ctx), total_ms)
        }
    }

    pub fn bench_gpu_kernel_only(&mut self, ctx: &mut GpuContext, batch: &BcnBatchPayload) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let fence_val = self.submit_async_compute(ctx, batch);
        let _ = ctx.wait_compute_cpu(fence_val, INFINITE);
        (self.read_last_kernel_ms(ctx), t0.elapsed().as_secs_f64() * 1000.0)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bc7RebuildChunkGpu {
    pub in_row_offset: u32,
    pub blocks: u32,
    pub out_block_offset: u32,
    pub _pad0: u32,
}

fn pack_bc7_tables() -> (Vec<u32>, Vec<u32>) {
    let t = bcp_bc7::Tables::default_table();
    let mut controls = vec![0u32; 512];
    for (idx, opt) in t.control.iter().enumerate() {
        if let Some(c) = opt {
            controls[idx] = 0x8000_0000u32
                | (c.mode as u32 & 7)
                | ((c.partition as u32 & 63) << 8)
                | ((c.rotation as u32 & 3) << 16)
                | ((c.index_selection as u32 & 1) << 24);
        }
    }
    let mut dict = vec![0u32; 255 * 11];
    for (k, (st, _)) in t.dictionary.iter().enumerate() {
        let base = k * 11;
        dict[base] = st.subsets as u32;
        for row in 0..3 {
            for col in 0..8 {
                let idx = row * 8 + col;
                dict[base + 1 + (idx >> 2)] |= (st.ep[row][col] as u32) << ((idx & 3) * 8);
            }
        }
        for set in 0..2 {
            for p in 0..16 {
                let idx = set * 16 + p;
                dict[base + 7 + (idx >> 3)] |= ((st.ix[set][p] & 0xF) as u32) << ((idx & 7) * 4);
            }
        }
    }
    (controls, dict)
}

/// Async Compute GPU Decoder for Full Multi-Mode BC7 (Modes 0..7, `0x1C`):
/// chains `bcpack_bc7_rans.dxil` (100% Wave32 16-lane rANS) -> `bcpack_bc7_rebuild.dxil`.
pub struct Bc7FullGpuDecoder {
    pub rans: Bc7RansGpuDecoder,
    rebuild_root_sig: ID3D12RootSignature,
    rebuild_pso: ID3D12PipelineState,

    controls_buf: ID3D12Resource,
    dict_buf: ID3D12Resource,

    cap_chunks: usize,
    cap_blocks: usize,

    chunks_upload: ID3D12Resource,
    chunks_default: ID3D12Resource,
    hist_default: ID3D12Resource,
    pub out_blocks_default: ID3D12Resource,
    out_readback: ID3D12Resource,
    status_readback: ID3D12Resource,
    ts_query_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,

    cmd_alloc: ID3D12CommandAllocator,
    cmd_list: ID3D12GraphicsCommandList,
}

impl Bc7FullGpuDecoder {
    pub fn new(
        ctx: &mut GpuContext,
        max_chunks: usize,
        max_bc7_blocks: usize,
        max_stream_bytes: usize,
    ) -> Self {
        unsafe {
            let max_sym_rows = max_bc7_blocks * 2;
            let rans = Bc7RansGpuDecoder::new(ctx, max_chunks, max_sym_rows, max_stream_bytes);

            let params = [
                cb_param(0, 4),
                srv_param(0),
                srv_param(1),
                srv_param(2),
                srv_param(3),
                uav_param(0),
                uav_param(1),
            ];
            let rebuild_root_sig = create_root_sig_from_params(&ctx.device, &params);
            let rebuild_pso = create_compute_pso(&ctx.device, &rebuild_root_sig, BC7_REBUILD_CS_DXIL, "bcpack_bc7_rebuild.dxil");

            let (controls, dict) = pack_bc7_tables();
            let controls_buf = upload_static_srv(ctx, &controls);
            let dict_buf = upload_static_srv(ctx, &dict);

            let padded_chunks = max_chunks + 2;
            let rc_bytes = (padded_chunks * size_of::<Bc7RebuildChunkGpu>()) as u64;
            let chunks_upload = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_UPLOAD, rc_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            let chunks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, rc_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            let hist_bytes = (padded_chunks * 256 * 11 * 4) as u64;
            let hist_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, hist_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);

            let out_bytes = (max_bc7_blocks * 16).max(256) as u64;
            let out_blocks_default = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            let out_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let status_bytes = (padded_chunks * size_of::<ChunkStatusGpu>()) as u64;
            let status_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, status_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let mut ts_query_heap: Option<ID3D12QueryHeap> = None;
            ctx.device
                .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut ts_query_heap)
                .expect("CreateQueryHeap Bc7Full");
            let ts_query_heap = ts_query_heap.unwrap();
            let ts_readback = create_committed_buffer(&ctx.device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

            let cmd_alloc: ID3D12CommandAllocator = ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
            let cmd_list: ID3D12GraphicsCommandList = ctx.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &cmd_alloc, None).unwrap();
            cmd_list.Close().unwrap();

            Self {
                rans,
                rebuild_root_sig,
                rebuild_pso,
                controls_buf,
                dict_buf,
                cap_chunks: padded_chunks,
                cap_blocks: max_bc7_blocks,
                chunks_upload,
                chunks_default,
                hist_default,
                out_blocks_default,
                out_readback,
                status_readback,
                ts_query_heap,
                ts_readback,
                cmd_alloc,
                cmd_list,
            }
        }
    }

    fn build_rebuild_chunks(batch: &BatchPayload) -> Vec<Bc7RebuildChunkGpu> {
        let mut out_blk = 0u32;
        batch.chunks[..batch.real_chunk_count]
            .iter()
            .map(|c| {
                let blks = c.blocks / 2;
                let rc = Bc7RebuildChunkGpu {
                    in_row_offset: c.out_block_offset,
                    blocks: blks,
                    out_block_offset: out_blk,
                    _pad0: 0,
                };
                out_blk += blks;
                rc
            })
            .collect()
    }

    unsafe fn record_full_dispatch(&self, cmd: &ID3D12GraphicsCommandList, batch: &BatchPayload, rchunks: &[Bc7RebuildChunkGpu]) {
        assert!(batch.real_chunk_count <= self.cap_chunks, "BC7Full chunks capacity exceeded");
        assert!(batch.total_blocks / 2 <= self.cap_blocks, "BC7Full block capacity exceeded");
        self.rans.upload_batch(batch);
        upload_slice(&self.chunks_upload, rchunks);

        let s_len = batch.stream_bytes.len() as u64;
        let c_len = (batch.chunks.len() * size_of::<ChunkDescGpu>()) as u64;
        let rc_len = (rchunks.len() * size_of::<Bc7RebuildChunkGpu>()) as u64;

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.rans.streams_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.rans.chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.chunks_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                transition_barrier(&self.rans.out_syms_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.rans.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.hist_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ],
        );
        cmd.CopyBufferRegion(&self.rans.streams_default, 0, &self.rans.streams_upload, 0, s_len);
        cmd.CopyBufferRegion(&self.rans.chunks_default, 0, &self.rans.chunks_upload, 0, c_len);
        cmd.CopyBufferRegion(&self.chunks_default, 0, &self.chunks_upload, 0, rc_len);
        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.rans.streams_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.rans.chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                transition_barrier(&self.chunks_default, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

        // Stage 1: 100% Wave32 16-lane rANS entropy decode
        cmd.SetComputeRootSignature(&self.rans.root_sig);
        cmd.SetPipelineState(&self.rans.pso);
        cmd.SetComputeRootShaderResourceView(0, self.rans.models_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(1, self.rans.streams_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.rans.chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(3, self.rans.out_syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(4, self.rans.status_default.GetGPUVirtualAddress());
        cmd.Dispatch((batch.chunks.len() / 2) as u32, 1, 1);

        cmd_barriers(
            cmd,
            vec![
                uav_barrier(&self.rans.out_syms_default),
                transition_barrier(&self.rans.out_syms_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            ],
        );

        // Stage 2: Full Multi-Mode BC7 (Modes 0..7) block rebuild
        let cb = [batch.real_chunk_count as u32, 0u32, 0u32, 0u32];
        cmd.SetComputeRootSignature(&self.rebuild_root_sig);
        cmd.SetPipelineState(&self.rebuild_pso);
        cmd.SetComputeRoot32BitConstants(0, 4, cb.as_ptr() as *const c_void, 0);
        cmd.SetComputeRootShaderResourceView(1, self.rans.out_syms_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(2, self.chunks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(3, self.controls_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootShaderResourceView(4, self.dict_buf.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(5, self.out_blocks_default.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(6, self.hist_default.GetGPUVirtualAddress());
        cmd.Dispatch(((batch.real_chunk_count + 63) / 64).max(1) as u32, 1, 1);

        cmd.EndQuery(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

        cmd_barriers(
            cmd,
            vec![
                transition_barrier(&self.rans.out_syms_default, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.rans.status_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.hist_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
            ],
        );
    }

    pub fn submit_async_compute(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> u64 {
        unsafe {
            let rchunks = Self::build_rebuild_chunks(batch);
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_full_dispatch(&self.cmd_list, batch, &rchunks);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn submit_async_to_buffer(&mut self, ctx: &mut GpuContext, batch: &BatchPayload, dst: &ID3D12Resource, dst_offset: u64) -> u64 {
        unsafe {
            let rchunks = Self::build_rebuild_chunks(batch);
            let out_bytes = ((batch.total_blocks / 2) * 16) as u64;
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_full_dispatch(&self.cmd_list, batch, &rchunks);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                ],
            );
            self.cmd_list.CopyBufferRegion(dst, dst_offset, &self.out_blocks_default, 0, out_bytes);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_compute()
        }
    }

    pub fn read_last_kernel_ms(&self, ctx: &GpuContext) -> f64 {
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.ts_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut ptr)).unwrap();
            let ts = std::slice::from_raw_parts(ptr as *const u64, 2);
            let ticks = ts[1].saturating_sub(ts[0]);
            self.ts_readback.Unmap(0, None);
            (ticks as f64 / ctx.compute_timestamp_freq as f64) * 1000.0
        }
    }

    pub fn decode_batch_and_readback(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> (Vec<u8>, Vec<ChunkStatusGpu>, f64, f64) {
        unsafe {
            let t0 = std::time::Instant::now();
            let rchunks = Self::build_rebuild_chunks(batch);
            let out_bytes = ((batch.total_blocks / 2) * 16) as u64;
            let status_bytes = (batch.real_chunk_count * size_of::<ChunkStatusGpu>()) as u64;
            self.cmd_alloc.Reset().unwrap();
            self.cmd_list.Reset(&self.cmd_alloc, None).unwrap();
            self.record_full_dispatch(&self.cmd_list, batch, &rchunks);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    transition_barrier(&self.rans.status_default, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
                ],
            );
            self.cmd_list.CopyBufferRegion(&self.out_readback, 0, &self.out_blocks_default, 0, out_bytes);
            self.cmd_list.CopyBufferRegion(&self.status_readback, 0, &self.rans.status_default, 0, status_bytes);
            self.cmd_list.ResolveQueryData(&self.ts_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ts_readback, 0);
            cmd_barriers(
                &self.cmd_list,
                vec![
                    transition_barrier(&self.out_blocks_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                    transition_barrier(&self.rans.status_default, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
                ],
            );
            self.cmd_list.Close().unwrap();
            let lists = [Some(self.cmd_list.cast::<ID3D12CommandList>().unwrap())];
            ctx.compute_queue.ExecuteCommandLists(&lists);
            ctx.signal_and_wait_compute();
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            self.out_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: out_bytes as usize }), Some(&mut ptr)).unwrap();
            let decoded = std::slice::from_raw_parts(ptr as *const u8, out_bytes as usize).to_vec();
            self.out_readback.Unmap(0, None);

            self.status_readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: status_bytes as usize }), Some(&mut ptr)).unwrap();
            let statuses = std::slice::from_raw_parts(ptr as *const ChunkStatusGpu, batch.real_chunk_count).to_vec();
            self.status_readback.Unmap(0, None);

            (decoded, statuses, self.read_last_kernel_ms(ctx), total_ms)
        }
    }

    pub fn bench_gpu_kernel_only(&mut self, ctx: &mut GpuContext, batch: &BatchPayload) -> (f64, f64) {
        let t0 = std::time::Instant::now();
        let fence_val = self.submit_async_compute(ctx, batch);
        let _ = ctx.wait_compute_cpu(fence_val, INFINITE);
        (self.read_last_kernel_ms(ctx), t0.elapsed().as_secs_f64() * 1000.0)
    }
}

