use crate::{eval::TznueNetwork, Board, ZcnEvaluator};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

const ACCUMULATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> board: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> activations: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let neuron = id.x;
    if neuron >= 256u {
        return;
    }

    var white = biases[neuron];
    var black = biases[neuron];
    for (var square = 0u; square < 64u; square += 1u) {
        let code = board[square];
        if code != 0u {
            let white_feature = (code - 1u) * 64u + square;
            let mirrored_code = select(code + 6u, code - 6u, code > 6u);
            let black_feature = (mirrored_code - 1u) * 64u + (square ^ 56u);
            white += weights[white_feature * 256u + neuron];
            black += weights[black_feature * 256u + neuron];
        }
    }
    activations[neuron] = clamp(white, 0.0, 1.0);
    activations[256u + neuron] = clamp(black, 0.0, 1.0);
}
"#;

const OUTPUT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> board: array<u32>;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    var sum = weights[512u];
    for (var neuron = 0u; neuron < 256u; neuron += 1u) {
        let stm = select(256u + neuron, neuron, board[64u] != 0u);
        let nstm = select(neuron, 256u + neuron, board[64u] != 0u);
        sum += activations[stm] * weights[neuron];
        sum += activations[nstm] * weights[256u + neuron];
    }
    output[0] = round(sum);
}
"#;

pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    board_buffer: wgpu::Buffer,
    feature_weight_buffer: wgpu::Buffer,
    feature_bias_buffer: wgpu::Buffer,
    output_weight_buffer: wgpu::Buffer,
    activation_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    accumulate_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
}

#[derive(Debug)]
pub struct GpuError(String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

impl GpuEvaluator {
    pub fn new(evaluator: &ZcnEvaluator) -> Result<Self, GpuError> {
        let network = evaluator
            .nnue
            .as_ref()
            .ok_or_else(|| GpuError("GPU inference requires a loaded TZNUE001 network".into()))?;
        pollster::block_on(Self::initialize(network))
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn evaluate(&mut self, board: &Board) -> Result<i32, GpuError> {
        let mut board_data = [0u32; 65];
        for (square, piece) in board.squares.iter().enumerate() {
            if let Some(piece) = piece {
                board_data[square] = *piece as u32 + 1;
            }
        }
        board_data[64] = u32::from(board.white_to_move());
        self.queue
            .write_buffer(&self.board_buffer, 0, &u32_bytes(&board_data));

        let accumulate_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zcn-accumulate-bind-group"),
            layout: &self.accumulate_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.board_buffer),
                entry(1, &self.feature_weight_buffer),
                entry(2, &self.feature_bias_buffer),
                entry(3, &self.activation_buffer),
            ],
        });
        let output_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zcn-output-bind-group"),
            layout: &self.output_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.board_buffer),
                entry(1, &self.activation_buffer),
                entry(2, &self.output_weight_buffer),
                entry(3, &self.result_buffer),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zcn-inference-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zcn-accumulate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.set_bind_group(0, &accumulate_bind_group, &[]);
            pass.dispatch_workgroups(4, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zcn-output-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.output_pipeline);
            pass.set_bind_group(0, &output_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.result_buffer, 0, &self.readback_buffer, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| GpuError(format!("GPU readback failed: {error}")))?
            .map_err(|error| GpuError(format!("GPU buffer mapping failed: {error}")))?;
        let mapped = slice.get_mapped_range();
        let score = f32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]) as i32;
        drop(mapped);
        self.readback_buffer.unmap();
        Ok(score)
    }

    async fn initialize(network: &TznueNetwork) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuError("No compatible GPU adapter was found".into()))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("zero-chess-net-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|error| GpuError(format!("Could not create GPU device: {error}")))?;

        let board_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zcn-board"),
            size: 65 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let feature_weight_buffer = storage_buffer(
            &device,
            "zcn-feature-weights",
            &network.ft_weights,
        );
        let feature_bias_buffer = storage_buffer(&device, "zcn-feature-biases", &network.ft_biases);
        let output_weights: Vec<f32> = network
            .out_weights
            .iter()
            .copied()
            .chain(std::iter::once(network.out_bias))
            .collect();
        let output_weight_buffer = storage_buffer(&device, "zcn-output-weights", &output_weights);
        let activation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zcn-activations"),
            size: 512 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zcn-result"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zcn-readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let accumulate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zcn-accumulate-shader"),
            source: wgpu::ShaderSource::Wgsl(ACCUMULATE_SHADER.into()),
        });
        let output_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zcn-output-shader"),
            source: wgpu::ShaderSource::Wgsl(OUTPUT_SHADER.into()),
        });
        let accumulate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("zcn-accumulate-pipeline"),
                layout: None,
                module: &accumulate_shader,
                entry_point: "main",
            });
        let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("zcn-output-pipeline"),
            layout: None,
            module: &output_shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            adapter_name,
            board_buffer,
            feature_weight_buffer,
            feature_bias_buffer,
            output_weight_buffer,
            activation_buffer,
            result_buffer,
            readback_buffer,
            accumulate_pipeline,
            output_pipeline,
        })
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &f32_bytes(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
