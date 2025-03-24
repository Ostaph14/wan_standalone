import os
import torch
import gc
import numpy as np
import math
from tqdm import tqdm
import argparse
from PIL import Image
import glob
from safetensors.torch import load_file

# Add ONNX and TensorRT imports
import onnx
import onnxruntime as ort

try:
	import tensorrt as trt
	from cuda import cudart
	import pycuda.driver as cuda
	import pycuda.autoinit

	TRT_AVAILABLE = True
except ImportError:
	print("TensorRT not available, falling back to ONNX Runtime")
	TRT_AVAILABLE = False

# Import the WanVideo modules
from wanvideo.modules.clip import CLIPModel
from wanvideo.modules.model import WanModel, rope_params
from wanvideo.modules.t5 import T5EncoderModel
from wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                       get_sampling_sigmas, retrieve_timesteps)
from wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wanvideo.wan_video_vae import WanVideoVAE

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


# TensorRT/ONNX utility functions
class ONNXExporter:
	"""Utility class for exporting PyTorch models to ONNX format"""

	@staticmethod
	def export_model(model, input_tensor, onnx_path, input_names=None, output_names=None,
	                 dynamic_axes=None, opset_version=17):
		"""
		Export a PyTorch model to ONNX format

		Args:
			model: PyTorch model
			input_tensor: Example input tensor(s)
			onnx_path: Path to save the ONNX model
			input_names: Names for the input nodes
			output_names: Names for the output nodes
			dynamic_axes: Dictionary with dynamic axes for inputs/outputs
			opset_version: ONNX opset version
		"""
		if not input_names:
			input_names = ['input']
		if not output_names:
			output_names = ['output']

		print(f"Exporting model to ONNX: {onnx_path}")

		# Ensure the directory exists
		os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

		# Set the model to evaluation mode
		model.eval()

		# Export the model to ONNX
		torch.onnx.export(
			model,
			input_tensor,
			onnx_path,
			export_params=True,
			opset_version=opset_version,
			do_constant_folding=True,
			input_names=input_names,
			output_names=output_names,
			dynamic_axes=dynamic_axes,
			verbose=False
		)

		print(f"Model exported to {onnx_path}")

		# Verify the ONNX model
		onnx_model = onnx.load(onnx_path)
		onnx.checker.check_model(onnx_model)
		print("ONNX model verified successfully")

		return onnx_path

	@staticmethod
	def optimize_onnx_model(onnx_path, optimized_path=None):
		"""
		Optimize an ONNX model using ONNX Runtime

		Args:
			onnx_path: Path to the ONNX model
			optimized_path: Path to save the optimized model (default: same as input with _optimized suffix)

		Returns:
			Path to the optimized model
		"""
		if not optimized_path:
			optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')

		print(f"Optimizing ONNX model: {onnx_path}")

		# Use ONNX Runtime to optimize the model
		from onnxruntime.transformers import optimizer
		opt_options = optimizer.OptimizationOptions()
		optimized_model = optimizer.optimize_model(
			onnx_path,
			'bert',  # Use transformer-based optimization
			num_heads=12,  # Adjust based on your model
			hidden_size=768,  # Adjust based on your model
			opt_level=99,
			optimization_options=opt_options
		)

		optimized_model.save_model_to_file(optimized_path)
		print(f"Optimized model saved to {optimized_path}")

		return optimized_path


class TensorRTEngine:
	"""Utility class for converting ONNX models to TensorRT engines and inference"""

	def __init__(self):
		"""Initialize TensorRT-related utilities"""
		if TRT_AVAILABLE:
			self.logger = trt.Logger(trt.Logger.WARNING)
			self.runtime = trt.Runtime(self.logger)
			# Print TensorRT version info
			print(f"TensorRT version: {trt.__version__}")

			# Get supported device properties
			if hasattr(trt, 'get_device_properties'):
				for i in range(cudart.cudaGetDeviceCount()[1]):
					props = trt.get_device_properties(i)
					print(
						f"GPU {i}: {props.name}, Compute: {props.major}.{props.minor}, Memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
		else:
			self.logger = None
			self.runtime = None
			print("TensorRT not available, using ONNX Runtime for inference.")

	def build_engine(self, onnx_path, engine_path=None, fp16_mode=True,
	                 max_workspace_size=1 << 30, input_shapes=None,
	                 use_dla=False, dla_core=0, use_strict_types=False,
	                 force_rebuild=False, tactic_sources=None, avg_timing_iterations=8):
		"""
		Build a TensorRT engine from an ONNX model with advanced configuration

		Args:
			onnx_path: Path to the ONNX model
			engine_path: Path to save the TensorRT engine (default: same as input with .engine suffix)
			fp16_mode: Whether to enable FP16 mode
			max_workspace_size: Maximum workspace size (default: 1GB)
			input_shapes: Dictionary mapping input names to their shapes with min/opt/max values
			use_dla: Whether to use DLA (Deep Learning Accelerator) if available
			dla_core: Which DLA core to use
			use_strict_types: Whether to use strict types for TensorRT optimization
			force_rebuild: Whether to force rebuild even if engine exists
			tactic_sources: List of tactic sources to use for optimization
			avg_timing_iterations: Number of timing iterations for kernel selection

		Returns:
			Path to the TensorRT engine
		"""
		if not TRT_AVAILABLE:
			print("TensorRT not available, cannot build engine")
			return None

		if not engine_path:
			engine_path = onnx_path.replace('.onnx', '.engine')

		print(f"Building TensorRT engine from {onnx_path}")

		# Check if engine already exists
		if os.path.exists(engine_path) and not force_rebuild:
			print(f"Engine {engine_path} already exists. Loading...")
			with open(engine_path, 'rb') as f:
				engine_bytes = f.read()
			engine = self.runtime.deserialize_cuda_engine(engine_bytes)
			return engine_path

		# Create builder and network
		builder = trt.Builder(self.logger)
		network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
		config = builder.create_builder_config()
		parser = trt.OnnxParser(network, self.logger)

		# Set config properties
		config.max_workspace_size = max_workspace_size

		# Enable FP16 mode if requested and supported
		if fp16_mode and builder.platform_has_fast_fp16:
			config.set_flag(trt.BuilderFlag.FP16)
			print("Enabled FP16 mode")

		# Set timing iterations for kernel selection
		config.avg_timing_iterations = avg_timing_iterations

		# Enable strict types if requested
		if use_strict_types:
			config.set_flag(trt.BuilderFlag.STRICT_TYPES)
			print("Enabled strict types")

		# Configure DLA if requested and available
		if use_dla:
			if hasattr(builder, 'platform_has_fast_dla') and builder.platform_has_fast_dla:
				print(f"Using DLA core {dla_core}")
				config.default_device_type = trt.DeviceType.DLA
				config.DLA_core = dla_core
				# Also enable GPU fallback for operations not supported on DLA
				config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
			else:
				print("DLA not available, using GPU instead")

		# Configure tactic sources if specified
		if tactic_sources and hasattr(config, 'set_tactic_sources'):
			enabled_sources = 0
			if 'CUBLAS' in tactic_sources:
				enabled_sources |= 1 << int(trt.TacticSource.CUBLAS)
			if 'CUBLAS_LT' in tactic_sources:
				enabled_sources |= 1 << int(trt.TacticSource.CUBLAS_LT)
			if 'CUDNN' in tactic_sources:
				enabled_sources |= 1 << int(trt.TacticSource.CUDNN)
			if 'EDGE_MASK_CONVOLUTIONS' in tactic_sources:
				enabled_sources |= 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)

			config.set_tactic_sources(enabled_sources)
			print(f"Set tactic sources: {tactic_sources}")

		# Parse ONNX model
		with open(onnx_path, 'rb') as model:
			model_bytes = model.read()
			if not parser.parse(model_bytes):
				for error in range(parser.num_errors):
					print(f"ONNX parsing error: {parser.get_error(error)}")
				return None

		# Set optimization profiles for dynamic shapes
		if input_shapes:
			profile = builder.create_optimization_profile()
			for input_name, shapes in input_shapes.items():
				min_shape, opt_shape, max_shape = shapes
				profile.set_shape(input_name, min_shape, opt_shape, max_shape)
			config.add_optimization_profile(profile)

		# Print network info
		print(f"Network has {network.num_layers} layers and {network.num_inputs} inputs, {network.num_outputs} outputs")
		for i in range(network.num_inputs):
			input_tensor = network.get_input(i)
			print(f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
		for i in range(network.num_outputs):
			output_tensor = network.get_output(i)
			print(f"Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

		# Build and save engine
		print("Building TensorRT engine... This may take a while.")
		engine = builder.build_engine(network, config)
		if engine:
			with open(engine_path, 'wb') as f:
				engine_bytes = engine.serialize()
				f.write(engine_bytes)
			print(f"TensorRT engine saved to {engine_path}")
		else:
			print("Failed to build TensorRT engine")

		return engine_path

	def load_engine(self, engine_path):
		"""
		Load a TensorRT engine from file

		Args:
			engine_path: Path to the TensorRT engine

		Returns:
			TensorRT engine
		"""
		if not TRT_AVAILABLE:
			print("TensorRT not available, cannot load engine")
			return None

		print(f"Loading TensorRT engine from {engine_path}")

		with open(engine_path, 'rb') as f:
			engine_bytes = f.read()

		engine = self.runtime.deserialize_cuda_engine(engine_bytes)
		return engine

	def allocate_buffers(self, engine, batch_size=1):
		"""
		Allocate device buffers for TensorRT inference

		Args:
			engine: TensorRT engine
			batch_size: Batch size

		Returns:
			Dictionaries for input/output host and device buffers, and bindings
		"""
		if not TRT_AVAILABLE:
			return None, None, None

		inputs = []
		outputs = []
		bindings = []

		# Collect all input and output dimensions
		for binding in range(engine.num_bindings):
			binding_dims = tuple(engine.get_binding_shape(binding))
			binding_dims = (batch_size,) + binding_dims[1:] if engine.binding_is_input(binding) else binding_dims
			size = trt.volume(binding_dims) * engine.max_batch_size
			dtype = trt.nptype(engine.get_binding_dtype(binding))

			# Allocate host and device buffers
			host_mem = cuda.pagelocked_empty(size, dtype)
			device_mem = cuda.mem_alloc(host_mem.nbytes)

			# Append info to the appropriate lists
			bindings.append(int(device_mem))
			if engine.binding_is_input(binding):
				inputs.append({'host': host_mem, 'device': device_mem, 'name': engine.get_binding_name(binding)})
			else:
				outputs.append({'host': host_mem, 'device': device_mem, 'name': engine.get_binding_name(binding)})

		return inputs, outputs, bindings

	def infer(self, engine, inputs, outputs, bindings, input_data):
		"""
		Run inference using a TensorRT engine

		Args:
			engine: TensorRT engine
			inputs: List of input buffers
			outputs: List of output buffers
			bindings: List of binding points
			input_data: Dictionary mapping input names to numpy arrays

		Returns:
			Dictionary mapping output names to numpy arrays
		"""
		if not TRT_AVAILABLE:
			print("TensorRT not available, cannot run inference")
			return None

		# Create execution context
		context = engine.create_execution_context()

		# Transfer input data to the GPU
		for input_buffer in inputs:
			if input_buffer['name'] in input_data:
				data = input_data[input_buffer['name']]
				np.copyto(input_buffer['host'], data.ravel())
				cuda.memcpy_htod(input_buffer['device'], input_buffer['host'])

		# Run inference
		stream = cuda.Stream()
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

		# Transfer predictions back from GPU
		for output_buffer in outputs:
			cuda.memcpy_dtoh_async(output_buffer['host'], output_buffer['device'], stream)

		# Synchronize the stream
		stream.synchronize()

		# Create output dictionary
		output_data = {}
		for output_buffer in outputs:
			# Reshape the output data to match the expected dimensions
			output_data[output_buffer['name']] = np.reshape(
				output_buffer['host'],
				context.get_binding_shape(engine.get_binding_index(output_buffer['name']))
			)

		return output_data


# Custom FP8 optimization implementation matching ComfyUI approach
def fp8_linear_forward_custom(cls, original_dtype, input):
	"""
	Custom implementation of FP8 linear forward pass that follows ComfyUI's approach
	"""
	if hasattr(cls, "original_forward"):
		return cls.original_forward(input.to(original_dtype))
	else:
		return cls.forward(input.to(original_dtype))


def convert_fp8_linear(module, original_dtype, params_to_keep={}):
	"""
	Convert linear layers to use custom FP8 implementation
	Matches ComfyUI's approach for memory optimization
	"""
	# Flag to indicate we've applied FP8 optimization
	setattr(module, "fp8_matmul_enabled", True)

	for name, module in module.named_modules():
		if not any(keyword in name for keyword in params_to_keep):
			if isinstance(module, torch.nn.Linear):
				# Store the original forward function
				original_forward = module.forward
				setattr(module, "original_forward", original_forward)

				# Replace with our custom implementation
				setattr(module, "forward", lambda input, m=module: fp8_linear_forward_custom(m, original_dtype, input))


# Memory management utilities with aggressive cleanup
def print_memory(device):
	if torch.cuda.is_available():
		print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
		print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
		print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")


def soft_empty_cache():
	"""Aggressively clean GPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		gc.collect()


def add_noise_to_reference_video(image, ratio=None):
	"""Add noise to a reference video for better I2V results."""
	sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
	image_noise = torch.randn_like(image) * sigma[:, None, None, None]
	image_noise = torch.where(image == -1, torch.zeros_like(image), image_noise)
	image = image + image_noise
	return image


def load_torch_file(path, device="cpu", safe_load=True):
	"""Load a PyTorch model from file."""
	if path.endswith(".safetensors"):
		sd = load_file(path, device=device)
	else:
		sd = torch.load(path, map_location=device)
		if "state_dict" in sd:
			sd = sd["state_dict"]
	return sd


def common_upscale(samples, width, height, upscale_method, crop):
	"""Upscale image to target dimensions."""
	if crop == "disabled":
		s = samples.shape
		if len(s) == 3:
			samples = samples.unsqueeze(0)
			s = samples.shape

		if s[2] != height or s[3] != width:
			if upscale_method == "lanczos":
				# Fall back to bicubic which is supported
				samples = torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic",
				                                          align_corners=False)
			else:
				samples = torch.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)
		if s[0] == 1:
			samples = samples.squeeze(0)
	else:
		raise ValueError("Only 'disabled' crop mode is currently implemented")

	return samples


class WanVideoGenerator:
	def __init__(self, models_dir, device="cuda:0", t5_tokenizer_path=None, clip_tokenizer_path=None):
		"""
		Initialize the WanVideo generator.

		Args:
			models_dir (str): Directory containing model files
			device (str): Device to use for computation (default: "cuda:0")
			t5_tokenizer_path (str): Path to T5 tokenizer
			clip_tokenizer_path (str): Path to CLIP tokenizer
		"""
		self.models_dir = models_dir
		self.device = device

		# Set offload device
		self.offload_device = "cpu"

		# Set paths
		self.t5_tokenizer_path = t5_tokenizer_path or os.path.join(models_dir, "configs", "T5_tokenizer")
		self.clip_tokenizer_path = clip_tokenizer_path or os.path.join(models_dir, "configs", "clip")

		# Store loaded models
		self.loaded_models = {}

		# Directory for ONNX models and TensorRT engines
		self.onnx_dir = os.path.join(models_dir, "onnx")
		os.makedirs(self.onnx_dir, exist_ok=True)

		# Initialize TensorRT utilities
		self.trt_engine = TensorRTEngine() if TRT_AVAILABLE else None

		# Track exported models to avoid duplicate exports
		self.exported_models = {}

	def load_model(self, model_path, base_precision="bf16", load_device="main_device",
	               quantization="disabled", attention_mode="sdpa", blocks_to_swap=0):
		"""
		Load the WanVideo transformer model.

		Args:
			model_path (str): Path to model file
			base_precision (str): Model precision (fp32, bf16, fp16)
			load_device (str): Which device to load model to (main_device or offload_device)
			quantization (str): Optional quantization method
			attention_mode (str): Attention implementation to use
			blocks_to_swap (int): Number of blocks to swap to CPU for VRAM optimization

		Returns:
			dict: Loaded model and configuration
		"""
		print(f"Loading model from {model_path}...")
		soft_empty_cache()

		# Set up devices and precision
		transformer_load_device = self.device if load_device == "main_device" else self.offload_device

		# Map precision strings to PyTorch dtypes
		base_dtype_map = {
			"fp8_e4m3fn": torch.float16,  # Use fp16 as we're emulating fp8
			"fp8_e4m3fn_fast": torch.float16,
			"bf16": torch.bfloat16,
			"fp16": torch.float16,
			"fp32": torch.float32
		}
		base_dtype = base_dtype_map[base_precision]

		# Load model state dict
		sd = load_torch_file(model_path, device=transformer_load_device)

		# Determine model architecture based on state dict
		dim = sd["patch_embedding.weight"].shape[0]
		in_channels = sd["patch_embedding.weight"].shape[1]
		ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
		model_type = "i2v" if in_channels == 36 else "t2v"
		num_heads = 40 if dim == 5120 else 12
		num_layers = 40 if dim == 5120 else 30

		print(f"Model type: {model_type}, dim: {dim}, heads: {num_heads}, layers: {num_layers}")

		# Configure the transformer model
		TRANSFORMER_CONFIG = {
			"dim": dim,
			"ffn_dim": ffn_dim,
			"eps": 1e-06,
			"freq_dim": 256,
			"in_dim": in_channels,
			"model_type": model_type,
			"out_dim": 16,
			"text_len": 512,
			"num_heads": num_heads,
			"num_layers": num_layers,
			"attention_mode": attention_mode,
			"main_device": self.device,
			"offload_device": self.offload_device,
		}

		# Initialize empty model then fill with weights
		with init_empty_weights():
			transformer = WanModel(**TRANSFORMER_CONFIG)
		transformer.eval()

		# Load weights
		manual_offloading = True

		if not "torchao" in quantization:
			print("Loading model weights to device...")

			# Use fp16 for loading when using fp8 emulation
			dtype = torch.float16 if quantization in ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_scaled"] else base_dtype

			# Keys to keep in base precision
			params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb",
			                  "modulation"}

			# Load parameters with appropriate dtypes
			for name, param in transformer.named_parameters():
				dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype

				# Check if this is a block parameter - if so, keep on CPU for now if using block_swap
				if blocks_to_swap > 0 and "block" in name:
					load_dev = self.offload_device
				else:
					load_dev = transformer_load_device

				set_module_tensor_to_device(transformer, name, device=load_dev, dtype=dtype_to_use,
				                            value=sd[name])

			if load_device == "offload_device":
				transformer.to(self.offload_device)
		else:
			# Not implementing torchao quantization for simplicity
			pass

		# Prepare block swapping
		if blocks_to_swap > 0:
			block_swap_args = {"blocks_to_swap": blocks_to_swap}
		else:
			block_swap_args = None

		model_info = {
			"model": transformer,
			"dtype": base_dtype,
			"model_path": model_path,
			"model_name": os.path.basename(model_path),
			"manual_offloading": manual_offloading,
			"quantization": quantization,
			"block_swap_args": block_swap_args,
			"model_type": model_type
		}

		del sd
		gc.collect()
		soft_empty_cache()

		# Apply our custom FP8 optimization
		if quantization in ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_scaled"]:
			print("Applying custom FP8 optimization to linear layers...")
			params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb",
			                  "modulation"}
			convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
			print("FP8 optimization applied.")

		self.loaded_models["diffusion_model"] = model_info
		return model_info

	def load_vae(self, vae_path, precision="bf16"):
		"""Load the WanVideo VAE model."""
		print(f"Loading VAE from {vae_path}...")

		dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
		vae_sd = load_torch_file(vae_path)

		# Handle model prefix if needed
		has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
		if not has_model_prefix:
			vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}

		# Create and load VAE
		vae = WanVideoVAE(dtype=dtype)
		vae.load_state_dict(vae_sd)
		vae.eval()
		vae.to(device=self.offload_device, dtype=dtype)

		self.loaded_models["vae"] = vae
		return vae

	def load_t5_encoder(self, t5_path, precision="bf16", load_device="offload_device", quantization="disabled"):
		"""Load the T5 text encoder."""
		print(f"Loading T5 encoder from {t5_path}...")

		text_encoder_load_device = self.device if load_device == "main_device" else self.offload_device
		dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

		sd = load_torch_file(t5_path)

		T5_text_encoder = T5EncoderModel(
			text_len=512,
			dtype=dtype,
			device=text_encoder_load_device,
			state_dict=sd,
			tokenizer_path=self.t5_tokenizer_path,
			quantization=quantization
		)

		text_encoder = {
			"model": T5_text_encoder,
			"dtype": dtype,
		}

		self.loaded_models["t5_encoder"] = text_encoder
		return text_encoder

	def load_clip_encoder(self, clip_path, precision="fp16", load_device="offload_device"):
		"""Load the CLIP text/image encoder."""
		print(f"Loading CLIP encoder from {clip_path}...")

		text_encoder_load_device = self.device if load_device == "main_device" else self.offload_device
		dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

		sd = load_torch_file(clip_path)

		clip_model = CLIPModel(
			dtype=dtype,
			device=self.device,
			state_dict=sd,
			tokenizer_path=self.clip_tokenizer_path
		)

		clip_model.model.to(text_encoder_load_device)

		self.loaded_models["clip_model"] = clip_model
		return clip_model

	def encode_text(self, t5_encoder, positive_prompt, negative_prompt, force_offload=True, use_onnx=False):
		"""Encode text prompts using the T5 encoder."""
		print(f"Encoding prompts:\nPositive: {positive_prompt}\nNegative: {negative_prompt}")

		encoder = t5_encoder["model"]
		dtype = t5_encoder["dtype"]

		# Check if this model has been exported to ONNX
		onnx_path = os.path.join(self.onnx_dir, "t5_encoder.onnx")
		engine_path = onnx_path.replace(".onnx", ".engine")

		if use_onnx and not os.path.exists(onnx_path) and not "t5_encoder" in self.exported_models:
			# Export model to ONNX if requested
			print("Exporting T5 encoder to ONNX...")

			# First make sure the model is on the correct device
			encoder.model.to(self.device)

			# Create dummy input for ONNX export
			dummy_input = ["This is a test prompt"]

			# Define ONNX export for T5Encoder
			onnx_model = ONNXExporter.export_model(
				encoder,
				dummy_input,
				onnx_path,
				input_names=["input_text"],
				output_names=["output_embeds"],
				dynamic_axes={
					"input_text": {0: "batch_size"},
					"output_embeds": {0: "batch_size", 1: "sequence_length"}
				}
			)

			# Optimize the ONNX model
			optimized_onnx = ONNXExporter.optimize_onnx_model(onnx_path)

			# Build TensorRT engine if available
			if TRT_AVAILABLE:
				# Define input shapes for TensorRT engine
				input_shapes = {
					"input_text": [(1,), (1,), (8,)]  # min, opt, max batch sizes
				}

				self.trt_engine.build_engine(
					optimized_onnx,
					engine_path,
					fp16_mode=True,
					input_shapes=input_shapes
				)

			# Mark as exported
			self.exported_models["t5_encoder"] = onnx_path

		# Use ONNX/TensorRT inference if available and requested
		if use_onnx and (os.path.exists(onnx_path) or os.path.exists(engine_path)):
			if TRT_AVAILABLE and os.path.exists(engine_path):
				# Use TensorRT for inference
				print("Using TensorRT for T5 encoding")
				engine = self.trt_engine.load_engine(engine_path)
				inputs, outputs, bindings = self.trt_engine.allocate_buffers(engine)

				# Prepare input data
				input_data = {
					"input_text": np.array([positive_prompt])
				}

				# Run inference
				result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
				context = [torch.from_numpy(result["output_embeds"]).to(self.device)]

				# Run again for negative prompt
				input_data = {
					"input_text": np.array([negative_prompt])
				}
				result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
				context_null = [torch.from_numpy(result["output_embeds"]).to(self.device)]

			elif os.path.exists(onnx_path):
				# Use ONNX Runtime for inference
				print("Using ONNX Runtime for T5 encoding")
				session_options = ort.SessionOptions()
				session = ort.InferenceSession(onnx_path, sess_options=session_options)

				# Run for positive prompt
				ort_inputs = {
					"input_text": np.array([positive_prompt])
				}
				ort_outputs = session.run(None, ort_inputs)
				context = [torch.from_numpy(ort_outputs[0]).to(self.device)]

				# Run for negative prompt
				ort_inputs = {
					"input_text": np.array([negative_prompt])
				}
				ort_outputs = session.run(None, ort_inputs)
				context_null = [torch.from_numpy(ort_outputs[0]).to(self.device)]
		else:
			# Use original PyTorch implementation
			# Move encoder to device for processing
			encoder.model.to(self.device)

			with torch.autocast(device_type=self.device.split(':')[0], dtype=dtype, enabled=True):
				context = encoder([positive_prompt], self.device)
				context_null = encoder([negative_prompt], self.device)

			context = [t.to(self.device) for t in context]
			context_null = [t.to(self.device) for t in context_null]

			# Aggressively offload when done
			if force_offload:
				encoder.model.to(self.offload_device)
				soft_empty_cache()

		return {
			"prompt_embeds": context,
			"negative_prompt_embeds": context_null,
		}

	def encode_image_clip(self, clip, vae, image, num_frames, generation_width, generation_height,
	                      force_offload=True, noise_aug_strength=0.0, latent_strength=1.0, clip_embed_strength=1.0,
	                      use_onnx=False):
		"""Encode an image using CLIP and prepare it for I2V processing."""
		print(f"Encoding input image with CLIP, generating {num_frames} frames...")

		# CLIP mean and std values
		image_mean = [0.48145466, 0.4578275, 0.40821073]
		image_std = [0.26862954, 0.26130258, 0.27577711]

		# Configuration parameters
		patch_size = (1, 2, 2)
		vae_stride = (4, 8, 8)
		sp_size = 1  # no parallelism

		# Get image dimensions
		if isinstance(image, torch.Tensor):
			H, W = image.shape[1], image.shape[2]
		else:
			# Handle PIL Image
			W, H = image.size
			# Convert PIL image to tensor
			image = torch.tensor(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
			# Reorder dimensions to [B, H, W, C]
			if image.shape[-1] == 3:  # Check if channels are in the last dimension
				pass  # Already in the right format
			else:
				image = image.permute(0, 2, 3, 1)

		max_area = generation_width * generation_height

		# Process image through CLIP
		# Custom clip_preprocess implementation
		def clip_preprocess(image, size=224, mean=None, std=None, crop=True):
			"""Preprocess image for CLIP."""
			if mean is None:
				mean = image_mean
			if std is None:
				std = image_std

			# Convert to [B, C, H, W] for PyTorch
			if image.shape[-1] == 3:  # [B, H, W, C]
				image = image.permute(0, 3, 1, 2)

			# Resize
			if crop:
				scale = size / min(image.shape[-2], image.shape[-1])
				image = torch.nn.functional.interpolate(
					image,
					size=(int(image.shape[-2] * scale), int(image.shape[-1] * scale)),
					mode='bicubic',
					align_corners=False
				)

				# Center crop
				h, w = image.shape[-2], image.shape[-1]
				start_h = (h - size) // 2
				start_w = (w - size) // 2
				image = image[:, :, start_h:start_h + size, start_w:start_w + size]
			else:
				image = torch.nn.functional.interpolate(
					image,
					size=(size, size),
					mode='bicubic',
					align_corners=False
				)

			# Normalize
			mean = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
			std = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
			image = (image - mean) / std

			return image

		# Check if CLIP encoder has been exported to ONNX
		clip_onnx_path = os.path.join(self.onnx_dir, "clip_encoder.onnx")
		clip_engine_path = clip_onnx_path.replace(".onnx", ".engine")

		if use_onnx and not os.path.exists(clip_onnx_path) and not "clip_encoder" in self.exported_models:
			# Export CLIP model to ONNX
			print("Exporting CLIP visual encoder to ONNX...")

			# First make sure the model is on the correct device
			clip.model.to(self.device)

			# Create dummy input for ONNX export - a single 224x224 image
			dummy_input = torch.randn(1, 3, 224, 224, device=self.device)

			# Export the visual part of CLIP
			onnx_model = ONNXExporter.export_model(
				clip.visual,
				dummy_input,
				clip_onnx_path,
				input_names=["input_image"],
				output_names=["output_features"],
				dynamic_axes={
					"input_image": {0: "batch_size"},
					"output_features": {0: "batch_size"}
				}
			)

			# Optimize the ONNX model
			optimized_onnx = ONNXExporter.optimize_onnx_model(clip_onnx_path)

			# Build TensorRT engine if available
			if TRT_AVAILABLE:
				# Define input shapes for TensorRT engine
				input_shapes = {
					"input_image": [(1, 3, 224, 224), (1, 3, 224, 224), (8, 3, 224, 224)]  # min, opt, max dimensions
				}

				self.trt_engine.build_engine(
					optimized_onnx,
					clip_engine_path,
					fp16_mode=True,
					input_shapes=input_shapes
				)

			# Mark as exported
			self.exported_models["clip_encoder"] = clip_onnx_path

		# Process with CLIP or ONNX/TensorRT
		# Pre-process the image
		pixel_values = clip_preprocess(image.to(self.device), size=224, mean=image_mean, std=image_std,
		                               crop=True).float()

		# Use ONNX/TensorRT for inference if available and requested
		if use_onnx and (os.path.exists(clip_onnx_path) or os.path.exists(clip_engine_path)):
			if TRT_AVAILABLE and os.path.exists(clip_engine_path):
				# Use TensorRT for inference
				print("Using TensorRT for CLIP encoding")
				engine = self.trt_engine.load_engine(clip_engine_path)
				inputs, outputs, bindings = self.trt_engine.allocate_buffers(engine)

				# Prepare input data
				input_data = {
					"input_image": pixel_values.cpu().numpy()
				}

				# Run inference
				result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
				clip_context = torch.from_numpy(result["output_features"]).to(self.device)

			elif os.path.exists(clip_onnx_path):
				# Use ONNX Runtime for inference
				print("Using ONNX Runtime for CLIP encoding")
				session_options = ort.SessionOptions()
				session = ort.InferenceSession(clip_onnx_path, sess_options=session_options)

				# Run inference
				ort_inputs = {
					"input_image": pixel_values.cpu().numpy()
				}
				ort_outputs = session.run(None, ort_inputs)
				clip_context = torch.from_numpy(ort_outputs[0]).to(self.device)
		else:
			# Use original PyTorch model
			clip.model.to(self.device)
			with torch.no_grad(), torch.cuda.amp.autocast():
				clip_context = clip.visual(pixel_values)

		if clip_embed_strength != 1.0:
			clip_context *= clip_embed_strength

		# Immediately offload CLIP model
		if force_offload and not use_onnx:
			clip.model.to(self.offload_device)
			soft_empty_cache()

		# Calculate dimensions
		aspect_ratio = H / W
		lat_h = round(
			np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
			patch_size[1] * patch_size[1])
		lat_w = round(
			np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
			patch_size[2] * patch_size[2])
		h = lat_h * vae_stride[1]
		w = lat_w * vae_stride[2]

		# Create mask for I2V
		mask = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
		mask[:, 1:] = 0

		# Repeat first frame 4 times and concatenate with remaining frames
		first_frame_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
		mask = torch.concat([first_frame_repeated, mask[:, 1:]], dim=1)

		# Reshape mask into groups of 4 frames
		mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)

		# Transpose dimensions and select first batch
		mask = mask.transpose(1, 2)[0]

		# Calculate maximum sequence length
		frames_per_stride = (num_frames - 1) // vae_stride[0] + 1
		patches_per_frame = lat_h * lat_w // (patch_size[1] * patch_size[2])
		raw_seq_len = frames_per_stride * patches_per_frame

		# Round up to nearest multiple of sp_size
		max_seq_len = int(math.ceil(raw_seq_len / sp_size)) * sp_size

		# Check if VAE encoder has been exported to ONNX
		vae_encoder_onnx_path = os.path.join(self.onnx_dir, "vae_encoder.onnx")
		vae_encoder_engine_path = vae_encoder_onnx_path.replace(".onnx", ".engine")

		if use_onnx and not os.path.exists(vae_encoder_onnx_path) and not "vae_encoder" in self.exported_models:
			# Export VAE encoder to ONNX
			print("Exporting VAE encoder to ONNX...")

			# First make sure the VAE is on the correct device
			vae.to(self.device)

			# Create dummy input for ONNX export
			# Shape should match expected input [C, B, H, W]
			dummy_input = [torch.randn(3, 1, h, w, device=self.device)]

			# Create a wrapper model for the encoder part
			class VAEEncoderWrapper(torch.nn.Module):
				def __init__(self, vae):
					super().__init__()
					self.vae = vae

				def forward(self, x):
					return self.vae.encode(x, self.vae.device)[0]

			vae_encoder_wrapper = VAEEncoderWrapper(vae)

			# Export the VAE encoder
			onnx_model = ONNXExporter.export_model(
				vae_encoder_wrapper,
				dummy_input,
				vae_encoder_onnx_path,
				input_names=["input_frames"],
				output_names=["latent"],
				dynamic_axes={
					"input_frames": {1: "batch", 2: "height", 3: "width"},
					"latent": {1: "batch", 2: "latent_height", 3: "latent_width"}
				}
			)

			# Optimize the ONNX model
			optimized_onnx = ONNXExporter.optimize_onnx_model(vae_encoder_onnx_path)

			# Build TensorRT engine if available
			if TRT_AVAILABLE:
				# Define input shapes for TensorRT engine
				input_shapes = {
					"input_frames": [(3, 1, 64, 64), (3, 1, h, w), (3, 16, 1024, 1024)]  # min, opt, max dimensions
				}

				self.trt_engine.build_engine(
					optimized_onnx,
					vae_encoder_engine_path,
					fp16_mode=True,
					input_shapes=input_shapes
				)

			# Mark as exported
			self.exported_models["vae_encoder"] = vae_encoder_onnx_path

		# Process through VAE
		vae.to(self.device)

		if isinstance(image, torch.Tensor):
			if image.shape[-1] == 3:  # [B, H, W, C]
				# Convert from [B, H, W, C] to [B, C, H, W]
				img = image.permute(0, 3, 1, 2)
			else:
				img = image
		else:
			# Convert PIL image to tensor
			img = torch.tensor(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).permute(0, 3, 1, 2)

		# Resize to target dimensions
		img = torch.nn.functional.interpolate(img, size=(h, w), mode="bicubic", align_corners=False)
		# Convert to [-1, 1] range
		img = img * 2 - 1
		# Reshape to [C, B, H, W] format needed by the model
		resized_image = img.squeeze(0).unsqueeze(1).to(self.device)  # Shape: [C, 1, H, W]

		if noise_aug_strength > 0.0:
			resized_image = add_noise_to_reference_video(resized_image, ratio=noise_aug_strength)

		# Create zero padding frames with correct shape
		zero_frames = torch.zeros(3, num_frames - 1, h, w, device=self.device)

		# Now both tensors have compatible dimensions for concatenation
		concatenated = torch.concat([resized_image, zero_frames], dim=1).to(dtype=vae.dtype)
		concatenated *= latent_strength

		# Use ONNX/TensorRT for VAE encoding if available and requested
		if use_onnx and (os.path.exists(vae_encoder_onnx_path) or os.path.exists(vae_encoder_engine_path)):
			if TRT_AVAILABLE and os.path.exists(vae_encoder_engine_path):
				# Use TensorRT for inference
				print("Using TensorRT for VAE encoding")
				engine = self.trt_engine.load_engine(vae_encoder_engine_path)
				inputs, outputs, bindings = self.trt_engine.allocate_buffers(engine)

				# Prepare input data - need to reshape to match expected input
				input_data = {
					"input_frames": [concatenated.cpu().numpy()]
				}

				# Run inference
				result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
				y = torch.from_numpy(result["latent"]).to(self.device)

			elif os.path.exists(vae_encoder_onnx_path):
				# Use ONNX Runtime for inference
				print("Using ONNX Runtime for VAE encoding")
				session_options = ort.SessionOptions()
				session = ort.InferenceSession(vae_encoder_onnx_path, sess_options=session_options)

				# Run inference
				ort_inputs = {
					"input_frames": [concatenated.cpu().numpy()]
				}
				ort_outputs = session.run(None, ort_inputs)
				y = torch.from_numpy(ort_outputs[0]).to(self.device)
		else:
			# Use original PyTorch VAE encoder
			# Use mixed precision encoding to save memory
			with torch.cuda.amp.autocast():
				y = vae.encode([concatenated], self.device)[0]

		# Combine mask and encoded latent
		y = torch.concat([mask, y])

		# Clear VAE cache and move back to offload device
		vae.model.clear_cache()
		vae.to(self.offload_device)
		soft_empty_cache()

		return {
			"image_embeds": y,
			"clip_context": clip_context,
			"max_seq_len": max_seq_len,
			"num_frames": num_frames,
			"lat_h": lat_h,
			"lat_w": lat_w,
		}

	def create_empty_embeds(self, width, height, num_frames):
		"""Create empty embeds for T2V models."""
		print(f"Creating empty embeds for {num_frames} frames at {width}x{height}...")

		patch_size = (1, 2, 2)
		vae_stride = (4, 8, 8)

		target_shape = (16, (num_frames - 1) // vae_stride[0] + 1,
		                height // vae_stride[1],
		                width // vae_stride[2])

		seq_len = math.ceil((target_shape[2] * target_shape[3]) /
		                    (patch_size[1] * patch_size[2]) *
		                    target_shape[1])

		return {
			"max_seq_len": seq_len,
			"target_shape": target_shape,
			"num_frames": num_frames
		}

	def export_unet_to_onnx(self, model_info, max_seq_len, num_frames, lat_h, lat_w, use_dynamic_axes=True):
		"""Export the diffusion UNet model to ONNX."""
		print("Exporting diffusion model to ONNX...")

		transformer = model_info["model"]
		model_type = model_info["model_type"]
		base_name = os.path.splitext(os.path.basename(model_info["model_path"]))[0]

		# Create output directory for the ONNX model
		onnx_path = os.path.join(self.onnx_dir, f"{base_name}_unet.onnx")
		engine_path = onnx_path.replace(".onnx", ".engine")

		# Check if already exported
		if "diffusion_model" in self.exported_models:
			return self.exported_models["diffusion_model"]

		# Check if files already exist
		if os.path.exists(onnx_path) or os.path.exists(engine_path):
			if os.path.exists(engine_path):
				self.exported_models["diffusion_model"] = engine_path
				return engine_path
			else:
				self.exported_models["diffusion_model"] = onnx_path
				return onnx_path

		# Move model to device
		transformer.to(self.device)
		transformer.eval()

		# Create dummy inputs based on model type
		if model_type == "i2v":
			# For Image to Video, we need image embeddings and CLIP context
			# Create dummy tensors for each input
			latent = torch.randn(1, 16, num_frames // 4, lat_h, lat_w, device=self.device)
			t = torch.tensor([500], device=self.device)  # Example timestep
			clip_fea = torch.randn(1, 1024, device=self.device)  # CLIP features
			context = [torch.randn(1, 512, 2048, device=self.device)]  # T5 text context
			y = [torch.randn(5, num_frames // 4, lat_h, lat_w, device=self.device)]  # Image embeddings

			# Configure dynamic axes for ONNX export
			dynamic_axes = {
				"latent": {0: "batch_size", 2: "frames", 3: "height", 4: "width"},
				"timestep": {},  # Scalar
				"clip_features": {0: "batch_size"},
				"context": {0: "batch_size", 1: "seq_len"},
				"image_embeddings": {1: "frames", 2: "height", 3: "width"},
				"output": {0: "batch_size", 2: "frames", 3: "height", 4: "width"}
			}

			# Create a wrapper class for ONNX export
			class UNetWrapperI2V(torch.nn.Module):
				def __init__(self, model):
					super().__init__()
					self.model = model

				def forward(self, latent, timestep, clip_features, context, image_embeddings):
					# Call the model with appropriate arguments
					return self.model(
						latent,
						t=timestep,
						clip_fea=clip_features,
						context=[context],
						y=[image_embeddings],
						seq_len=max_seq_len,
						device=latent.device,
						freqs=None  # Will be generated inside the model
					)

			# Create the wrapper model
			wrapper_model = UNetWrapperI2V(transformer)

			# Input names and example inputs
			input_names = ["latent", "timestep", "clip_features", "context", "image_embeddings"]
			output_names = ["output"]
			example_inputs = (latent, t, clip_fea, context[0], y[0])

		else:  # t2v
			# For Text to Video, we need only text context
			# Create dummy tensors
			latent = torch.randn(1, 16, num_frames // 4, lat_h, lat_w, device=self.device)
			t = torch.tensor([500], device=self.device)  # Example timestep
			context = [torch.randn(1, 512, 2048, device=self.device)]  # T5 text context

			# Configure dynamic axes for ONNX export
			dynamic_axes = {
				"latent": {0: "batch_size", 2: "frames", 3: "height", 4: "width"},
				"timestep": {},  # Scalar
				"context": {0: "batch_size", 1: "seq_len"},
				"output": {0: "batch_size", 2: "frames", 3: "height", 4: "width"}
			}

			# Create a wrapper class for ONNX export
			class UNetWrapperT2V(torch.nn.Module):
				def __init__(self, model):
					super().__init__()
					self.model = model

				def forward(self, latent, timestep, context):
					# Call the model with appropriate arguments
					return self.model(
						latent,
						t=timestep,
						context=[context],
						seq_len=max_seq_len,
						device=latent.device,
						freqs=None  # Will be generated inside the model
					)

			# Create the wrapper model
			wrapper_model = UNetWrapperT2V(transformer)

			# Input names and example inputs
			input_names = ["latent", "timestep", "context"]
			output_names = ["output"]
			example_inputs = (latent, t, context[0])

		# Export the model to ONNX
		if not use_dynamic_axes:
			dynamic_axes = None

		try:
			onnx_path = ONNXExporter.export_model(
				wrapper_model,
				example_inputs,
				onnx_path,
				input_names=input_names,
				output_names=output_names,
				dynamic_axes=dynamic_axes
			)

			# Optimize the ONNX model
			optimized_onnx = ONNXExporter.optimize_onnx_model(onnx_path)

			# Build TensorRT engine if available
			if TRT_AVAILABLE:
				# Define input shapes - example only, would need adjustment for real use
				if model_type == "i2v":
					input_shapes = {
						"latent": [(1, 16, num_frames // 4, lat_h, lat_w),
						           (1, 16, num_frames // 4, lat_h, lat_w),
						           (2, 16, num_frames // 4 * 2, lat_h * 2, lat_w * 2)],
						"clip_features": [(1, 1024), (1, 1024), (2, 1024)],
						"context": [(1, 512, 2048), (1, 512, 2048), (2, 512, 2048)],
						"image_embeddings": [(5, num_frames // 4, lat_h, lat_w),
						                     (5, num_frames // 4, lat_h, lat_w),
						                     (5, num_frames // 4 * 2, lat_h * 2, lat_w * 2)]
					}
				else:
					input_shapes = {
						"latent": [(1, 16, num_frames // 4, lat_h, lat_w),
						           (1, 16, num_frames // 4, lat_h, lat_w),
						           (2, 16, num_frames // 4 * 2, lat_h * 2, lat_w * 2)],
						"context": [(1, 512, 2048), (1, 512, 2048), (2, 512, 2048)]
					}

				self.trt_engine.build_engine(
					optimized_onnx,
					engine_path,
					fp16_mode=True,
					input_shapes=input_shapes
				)

				if os.path.exists(engine_path):
					path = engine_path
				else:
					path = optimized_onnx
			else:
				path = optimized_onnx

			self.exported_models["diffusion_model"] = path
			return path

		except Exception as e:
			print(f"Error exporting UNet model to ONNX: {e}")
			return None

	def export_vae_decoder_to_onnx(self, vae, latents_shape):
		"""Export the VAE decoder to ONNX."""
		print("Exporting VAE decoder to ONNX...")

		vae_decoder_onnx_path = os.path.join(self.onnx_dir, "vae_decoder.onnx")
		vae_decoder_engine_path = vae_decoder_onnx_path.replace(".onnx", ".engine")

		# Check if already exported
		if "vae_decoder" in self.exported_models:
			return self.exported_models["vae_decoder"]

		# Check if files already exist
		if os.path.exists(vae_decoder_onnx_path) or os.path.exists(vae_decoder_engine_path):
			if os.path.exists(vae_decoder_engine_path):
				self.exported_models["vae_decoder"] = vae_decoder_engine_path
				return vae_decoder_engine_path
			else:
				self.exported_models["vae_decoder"] = vae_decoder_onnx_path
				return vae_decoder_onnx_path

		# Move VAE to device
		vae.to(self.device)
		vae.eval()

		# Create dummy input for the decoder
		dummy_latent = torch.randn(*latents_shape, device=self.device)

		# Create a wrapper class for the decoder
		class VAEDecoderWrapper(torch.nn.Module):
			def __init__(self, vae):
				super().__init__()
				self.vae = vae

			def forward(self, latent):
				return self.vae.decode(latent, self.vae.device)[0]

		vae_decoder_wrapper = VAEDecoderWrapper(vae)

		# Configure dynamic axes
		dynamic_axes = {
			"latent": {0: "batch_size", 2: "frames", 3: "latent_height", 4: "latent_width"},
			"output": {0: "batch_size", 2: "frames", 3: "height", 4: "width"}
		}

		try:
			# Export the model to ONNX
			onnx_path = ONNXExporter.export_model(
				vae_decoder_wrapper,
				dummy_latent,
				vae_decoder_onnx_path,
				input_names=["latent"],
				output_names=["output"],
				dynamic_axes=dynamic_axes
			)

			# Optimize the ONNX model
			optimized_onnx = ONNXExporter.optimize_onnx_model(vae_decoder_onnx_path)

			# Build TensorRT engine if available
			if TRT_AVAILABLE:
				# Define input shapes
				input_shapes = {
					"latent": [
						(1, 16, 1, 8, 8),  # min
						latents_shape,  # opt
						(2, 16, 40, 64, 64)  # max - would adjust based on your use case
					]
				}

				self.trt_engine.build_engine(
					optimized_onnx,
					vae_decoder_engine_path,
					fp16_mode=True,
					input_shapes=input_shapes
				)

				if os.path.exists(vae_decoder_engine_path):
					path = vae_decoder_engine_path
				else:
					path = optimized_onnx
			else:
				path = optimized_onnx

			self.exported_models["vae_decoder"] = path
			return path

		except Exception as e:
			print(f"Error exporting VAE decoder to ONNX: {e}")
			return None

	def sample(self, model_info, text_embeds, image_embeds, shift, steps, cfg, seed=None,
	           scheduler="dpm++", riflex_freq_index=0, force_offload=True,
	           samples=None, denoise_strength=1.0, use_onnx=False):
		"""
		Run the sampling process to generate the video.

		Args:
			model_info: The loaded diffusion model info
			text_embeds: Encoded text prompts
			image_embeds: Encoded image or empty embeds
			shift: Shift parameter for the scheduler
			steps: Number of denoising steps
			cfg: Classifier-free guidance scale
			seed: Random seed (if None, a random seed will be generated)
			scheduler: Sampling scheduler to use
			riflex_freq_index: Frequency index for RIFLEX
			force_offload: Whether to offload models after use
			samples: Initial latents for video2video
			denoise_strength: Strength of denoising (for video2video)
			use_onnx: Whether to use ONNX/TensorRT for inference

		Returns:
			Generated video latents
		"""
		print(f"Running sampling for {steps} steps with {scheduler} scheduler...")

		if seed is None:
			seed = int(torch.randint(0, 2147483647, (1,)).item())

		print(f"Using seed: {seed}")

		transformer = model_info["model"]
		model_type = model_info["model_type"]

		# Adjust steps based on denoise strength
		steps = int(steps / denoise_strength)

		# Initialize scheduler
		if scheduler == 'unipc':
			sample_scheduler = FlowUniPCMultistepScheduler(
				num_train_timesteps=1000,
				shift=shift,
				use_dynamic_shifting=False)
			sample_scheduler.set_timesteps(
				steps, device=self.device, shift=shift)
			timesteps = sample_scheduler.timesteps
		elif 'dpm++' in scheduler:
			if scheduler == 'dpm++_sde':
				algorithm_type = "sde-dpmsolver++"
			else:
				algorithm_type = "dpmsolver++"
			sample_scheduler = FlowDPMSolverMultistepScheduler(
				num_train_timesteps=1000,
				shift=shift,
				use_dynamic_shifting=False,
				algorithm_type=algorithm_type)
			sampling_sigmas = get_sampling_sigmas(steps, shift)
			timesteps, _ = retrieve_timesteps(
				sample_scheduler,
				device=self.device,
				sigmas=sampling_sigmas)
		else:
			raise NotImplementedError(f"Unsupported scheduler: {scheduler}")

		# Adjust timesteps for denoise strength
		if denoise_strength < 1.0:
			steps = int(steps * denoise_strength)
			timesteps = timesteps[-(steps + 1):]

		# Prepare initial noise
		seed_g = torch.Generator(device=torch.device("cpu"))
		seed_g.manual_seed(seed)

		if model_type == "i2v":
			lat_h = image_embeds.get("lat_h", None)
			lat_w = image_embeds.get("lat_w", None)
			if lat_h is None or lat_w is None:
				raise ValueError("Clip encoded image embeds must be provided for I2V (Image to Video) model")

			noise = torch.randn(
				16,
				(image_embeds["num_frames"] - 1) // 4 + 1,
				lat_h,
				lat_w,
				dtype=torch.float32,
				generator=seed_g,
				device=torch.device("cpu"))

			seq_len = image_embeds["max_seq_len"]
		else:  # t2v
			target_shape = image_embeds.get("target_shape", None)
			if target_shape is None:
				raise ValueError("Empty image embeds must be provided for T2V (Text to Video)")

			seq_len = image_embeds["max_seq_len"]
			noise = torch.randn(
				target_shape[0],
				target_shape[1],
				target_shape[2],
				target_shape[3],
				dtype=torch.float32,
				device=torch.device("cpu"),
				generator=seed_g)

		# Handle initial latents for video2video
		if samples is not None:
			latent_timestep = timesteps[:1].to(noise)
			noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * samples["samples"].squeeze(0).to(
				noise)

		# Initialize latent on CPU first, then move to GPU when needed
		latent = noise

		# Export model to ONNX if using ONNX and not already exported
		if use_onnx and not "diffusion_model" in self.exported_models:
			if model_type == "i2v":
				self.export_unet_to_onnx(
					model_info,
					seq_len,
					image_embeds["num_frames"],
					image_embeds["lat_h"],
					image_embeds["lat_w"]
				)
			else:
				# For T2V, use target_shape parameters
				self.export_unet_to_onnx(
					model_info,
					seq_len,
					image_embeds["num_frames"],
					image_embeds["target_shape"][2],  # height
					image_embeds["target_shape"][3]  # width
				)

		# Set up rope parameters
		d = transformer.dim // transformer.num_heads
		freqs = torch.cat([
			rope_params(1024, d - 4 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index),
			rope_params(1024, 2 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index),
			rope_params(1024, 2 * (d // 6), L_test=latent.shape[2], k=riflex_freq_index)
		], dim=1)

		# Handle cfg
		if not isinstance(cfg, list):
			cfg = [cfg] * (steps + 1)

		# Prepare arguments
		base_args = {
			'clip_fea': image_embeds.get('clip_context', None),
			'seq_len': seq_len,
			'device': self.device,
			'freqs': freqs,
		}

		if model_type == "i2v":
			base_args.update({
				'y': [image_embeds["image_embeds"]],
			})

		arg_c = base_args.copy()
		arg_c.update({'context': [text_embeds["prompt_embeds"][0]]})

		arg_null = base_args.copy()
		arg_null.update({'context': text_embeds["negative_prompt_embeds"]})

		# Check if we should use ONNX/TensorRT for inference
		use_engine = use_onnx and "diffusion_model" in self.exported_models

		if use_engine:
			engine_path = self.exported_models["diffusion_model"]
			if TRT_AVAILABLE and engine_path.endswith(".engine"):
				print("Using TensorRT engine for diffusion model inference")
				engine = self.trt_engine.load_engine(engine_path)
				inputs, outputs, bindings = self.trt_engine.allocate_buffers(engine)
			elif engine_path.endswith(".onnx"):
				print("Using ONNX Runtime for diffusion model inference")
				session_options = ort.SessionOptions()
				session = ort.InferenceSession(engine_path, sess_options=session_options)

		# Handle block swapping - crucial for memory efficiency in ComfyUI
		if not use_engine:
			if model_info["block_swap_args"] is not None:
				# Only load non-block parameters to GPU
				for name, param in transformer.named_parameters():
					if "block" not in name:
						param.data = param.data.to(self.device)

				# Use block swapping (loads only necessary blocks during inference)
				transformer.block_swap(
					model_info["block_swap_args"]["blocks_to_swap"] - 1,
				)
			else:
				if model_info["manual_offloading"]:
					transformer.to(self.device)

		soft_empty_cache()

		# Turn off gradients for memory efficiency
		with torch.no_grad():
			# Enable mixed precision for memory efficiency
			with torch.autocast(device_type=self.device.split(':')[0], dtype=model_info["dtype"], enabled=True):
				for i, t in enumerate(tqdm(timesteps)):
					# Only move latent to GPU during processing - like ComfyUI
					latent_gpu = latent.to(self.device)
					latent_model_input = [latent_gpu]
					timestep = torch.tensor([t], device=self.device)

					# Get conditional noise prediction using PyTorch or ONNX/TensorRT
					if use_engine:
						if model_type == "i2v":
							if TRT_AVAILABLE and engine_path.endswith(".engine"):
								# Use TensorRT for inference
								input_data = {
									"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
									"timestep": timestep.cpu().numpy(),
									"clip_features": image_embeds["clip_context"].cpu().numpy(),
									"context": text_embeds["prompt_embeds"][0].cpu().numpy(),
									"image_embeddings": image_embeds["image_embeds"].cpu().numpy()
								}
								result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
								noise_pred_cond = torch.from_numpy(result["output"]).squeeze(0).to(self.device)
							else:
								# Use ONNX Runtime for inference
								ort_inputs = {
									"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
									"timestep": timestep.cpu().numpy(),
									"clip_features": image_embeds["clip_context"].cpu().numpy(),
									"context": text_embeds["prompt_embeds"][0].cpu().numpy(),
									"image_embeddings": image_embeds["image_embeds"].cpu().numpy()
								}
								ort_outputs = session.run(None, ort_inputs)
								noise_pred_cond = torch.from_numpy(ort_outputs[0]).squeeze(0).to(self.device)
						else:  # t2v
							if TRT_AVAILABLE and engine_path.endswith(".engine"):
								# Use TensorRT for inference
								input_data = {
									"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
									"timestep": timestep.cpu().numpy(),
									"context": text_embeds["prompt_embeds"][0].cpu().numpy()
								}
								result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
								noise_pred_cond = torch.from_numpy(result["output"]).squeeze(0).to(self.device)
							else:
								# Use ONNX Runtime for inference
								ort_inputs = {
									"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
									"timestep": timestep.cpu().numpy(),
									"context": text_embeds["prompt_embeds"][0].cpu().numpy()
								}
								ort_outputs = session.run(None, ort_inputs)
								noise_pred_cond = torch.from_numpy(ort_outputs[0]).squeeze(0).to(self.device)
					else:
						# Use original PyTorch model
						noise_pred_cond = transformer(
							latent_model_input, t=timestep, **arg_c)[0].to(self.offload_device)

					# Free GPU memory
					del latent_gpu
					soft_empty_cache()

					if cfg[i] != 1.0:
						# Move latent back to GPU for unconditional pass
						latent_gpu = latent.to(self.device)
						latent_model_input = [latent_gpu]

						# Get unconditional noise prediction
						if use_engine:
							if model_type == "i2v":
								if TRT_AVAILABLE and engine_path.endswith(".engine"):
									# Use TensorRT for inference
									input_data = {
										"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
										"timestep": timestep.cpu().numpy(),
										"clip_features": image_embeds["clip_context"].cpu().numpy(),
										"context": text_embeds["negative_prompt_embeds"][0].cpu().numpy(),
										"image_embeddings": image_embeds["image_embeds"].cpu().numpy()
									}
									result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
									noise_pred_uncond = torch.from_numpy(result["output"]).squeeze(0).to(self.device)
								else:
									# Use ONNX Runtime for inference
									ort_inputs = {
										"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
										"timestep": timestep.cpu().numpy(),
										"clip_features": image_embeds["clip_context"].cpu().numpy(),
										"context": text_embeds["negative_prompt_embeds"][0].cpu().numpy(),
										"image_embeddings": image_embeds["image_embeds"].cpu().numpy()
									}
									ort_outputs = session.run(None, ort_inputs)
									noise_pred_uncond = torch.from_numpy(ort_outputs[0]).squeeze(0).to(self.device)
							else:  # t2v
								if TRT_AVAILABLE and engine_path.endswith(".engine"):
									# Use TensorRT for inference
									input_data = {
										"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
										"timestep": timestep.cpu().numpy(),
										"context": text_embeds["negative_prompt_embeds"][0].cpu().numpy()
									}
									result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
									noise_pred_uncond = torch.from_numpy(result["output"]).squeeze(0).to(self.device)
								else:
									# Use ONNX Runtime for inference
									ort_inputs = {
										"latent": latent_model_input[0].unsqueeze(0).cpu().numpy(),
										"timestep": timestep.cpu().numpy(),
										"context": text_embeds["negative_prompt_embeds"][0].cpu().numpy()
									}
									ort_outputs = session.run(None, ort_inputs)
									noise_pred_uncond = torch.from_numpy(ort_outputs[0]).squeeze(0).to(self.device)
						else:
							# Use original PyTorch model
							noise_pred_uncond = transformer(
								latent_model_input, t=timestep, **arg_null)[0].to(self.offload_device)

						# Free GPU memory again
						del latent_gpu
						soft_empty_cache()

						# Compute weighted noise prediction
						noise_pred = noise_pred_uncond + cfg[i] * (
								noise_pred_cond - noise_pred_uncond)
					else:
						noise_pred = noise_pred_cond

					# Move needed tensors to GPU for sampling step
					noise_pred_gpu = noise_pred.to(self.device)
					latent_gpu = latent.to(self.device)

					# Perform sampling step
					temp_x0 = sample_scheduler.step(
						noise_pred_gpu.unsqueeze(0),
						t,
						latent_gpu.unsqueeze(0),
						return_dict=False,
						generator=seed_g)[0]

					# Update latent and keep on CPU
					latent = temp_x0.squeeze(0).cpu()

					# Free GPU memory
					del noise_pred_gpu, latent_gpu, latent_model_input, timestep
					soft_empty_cache()

		# Move final result to GPU for one last processing
		latent_gpu = latent.to(self.device)
		result = latent_gpu.unsqueeze(0).cpu()

		# Clean up
		del latent_gpu
		if force_offload and not use_engine:
			if model_info["manual_offloading"]:
				transformer.to(self.offload_device)
			soft_empty_cache()

		return {"samples": result}

	def decode(self, vae, samples, enable_vae_tiling=True,
	           tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128,
	           use_onnx=False):
		"""Decode latents into video frames."""
		print("Decoding video frames...")

		soft_empty_cache()
		latents = samples["samples"]

		# Export VAE decoder to ONNX if using ONNX and not already exported
		if use_onnx and not "vae_decoder" in self.exported_models:
			self.export_vae_decoder_to_onnx(vae, latents.shape)

		# Check if we should use ONNX/TensorRT for decoding
		use_engine = use_onnx and "vae_decoder" in self.exported_models

		if use_engine:
			engine_path = self.exported_models["vae_decoder"]
			if TRT_AVAILABLE and engine_path.endswith(".engine"):
				print("Using TensorRT engine for VAE decoding")
				engine = self.trt_engine.load_engine(engine_path)
				inputs, outputs, bindings = self.trt_engine.allocate_buffers(engine)

				# Move latents to device
				latents_gpu = latents.to(device=self.device, dtype=vae.dtype)

				# Run inference
				input_data = {
					"latent": latents_gpu.cpu().numpy()
				}
				result = self.trt_engine.infer(engine, inputs, outputs, bindings, input_data)
				image = torch.from_numpy(result["output"]).to(self.device)

			elif engine_path.endswith(".onnx"):
				print("Using ONNX Runtime for VAE decoding")
				session_options = ort.SessionOptions()
				session = ort.InferenceSession(engine_path, sess_options=session_options)

				# Move latents to device
				latents_gpu = latents.to(device=self.device, dtype=vae.dtype)

				# Run inference
				ort_inputs = {
					"latent": latents_gpu.cpu().numpy()
				}
				ort_outputs = session.run(None, ort_inputs)
				image = torch.from_numpy(ort_outputs[0]).to(self.device)
		else:
			# Use original PyTorch VAE decoder
			# Move VAE to device for decoding
			vae.to(self.device)

			# Move latents to device
			latents = latents.to(device=self.device, dtype=vae.dtype)

			# Use mixed precision for decoding to save memory
			with torch.cuda.amp.autocast():
				# Decode in tiles for memory efficiency
				image = vae.decode(
					latents,
					device=self.device,
					tiled=enable_vae_tiling,
					tile_size=(tile_x, tile_y),
					tile_stride=(tile_stride_x, tile_stride_y)
				)[0]

			# Clean up
			vae.to(self.offload_device)
			vae.model.clear_cache()

		# Normalize and format
		image = (image - image.min()) / (image.max() - image.min())
		image = torch.clamp(image, 0.0, 1.0)
		image = image.permute(1, 2, 3, 0).cpu().float()

		soft_empty_cache()

		return image

	def generate_video(self,
	                   # Model parameters
	                   model_path=None,
	                   vae_path=None,
	                   t5_path=None,
	                   clip_path=None,

	                   # Precision settings
	                   base_precision="fp16",  # Use fp16 as default for better memory
	                   vae_precision="bf16",
	                   t5_precision="bf16",
	                   clip_precision="fp16",

	                   # Text prompts
	                   positive_prompt="a man standing in front of a ford gt surprised",
	                   negative_prompt="bad quality video",

	                   # Video settings
	                   width=512,
	                   height=512,
	                   num_frames=81,

	                   # Generation parameters
	                   steps=10,
	                   cfg=6.0,
	                   shift=5.0,
	                   seed=None,
	                   scheduler="dpm++",
	                   riflex_freq_index=0,

	                   # Resource optimization
	                   quantization="fp8_e4m3fn_fast",  # Using our custom FP8 implementation
	                   attention_mode="sdpa",
	                   blocks_to_swap=30,  # More aggressive block swapping
	                   force_offload=True,

	                   # VAE settings
	                   enable_vae_tiling=True,
	                   tile_x=272,
	                   tile_y=272,
	                   tile_stride_x=144,
	                   tile_stride_y=128,

	                   # Image to Video settings
	                   input_image=None,
	                   noise_aug_strength=0.0,
	                   latent_strength=1.0,
	                   clip_embed_strength=1.0,

	                   # Video to Video settings
	                   input_video=None,
	                   denoise_strength=1.0,

	                   # Output settings
	                   save_path=None,
	                   output_format="mp4",
	                   fps=16,

	                   # ONNX/TensorRT settings
	                   use_onnx=False,
	                   export_onnx=False
	                   ):
		"""
		Generate a video using the WanVideo model.
		"""
		print(f"Generating video...")
		print(f"Prompt: {positive_prompt}")

		# IMPORTANT FIX: Adjust num_frames to satisfy the model's requirement
		# For the mask reshaping to work, (3 + num_frames) must be divisible by 4
		# In other words, num_frames must be congruent to 1 modulo 4 (num_frames % 4 == 1)
		original_num_frames = num_frames
		while num_frames % 4 != 1:
			num_frames += 1

		if original_num_frames != num_frames:
			print(
				f"Adjusted num_frames from {original_num_frames} to {num_frames} to ensure compatibility with the model")

		# Ensure we have aggressive memory cleanup
		soft_empty_cache()

		# Find models if not explicitly provided
		if model_path is None:
			model_files = glob.glob(os.path.join(self.models_dir, "*.safetensors"))
			for file in model_files:
				if "I2V" in file and input_image is not None:
					model_path = file
					break
				elif "T2V" in file and input_image is None:
					model_path = file
					break
			if model_path is None and model_files:
				model_path = model_files[0]

		# Look in the subdirectories if not found directly
		if vae_path is None:
			vae_path = os.path.join(self.models_dir, "vae")
			if os.path.exists(vae_path):
				vae_files = glob.glob(os.path.join(vae_path, "*.safetensors"))
				if vae_files:
					vae_path = vae_files[0]

		# Same for t5 path
		if t5_path is None:
			t5_path = os.path.join(self.models_dir, "t5")
			if os.path.exists(t5_path):
				t5_files = glob.glob(os.path.join(t5_path, "*.safetensors"))
				if t5_files:
					t5_path = t5_files[0]

		# Same for clip path
		if clip_path is None and input_image is not None:
			clip_path = os.path.join(self.models_dir, "clip")
			if os.path.exists(clip_path):
				clip_files = glob.glob(os.path.join(clip_path, "*.safetensors"))
				if clip_files:
					clip_path = clip_files[0]

		# Ensure models are found
		if model_path is None:
			raise ValueError("No diffusion model found. Please specify model_path.")
		if vae_path is None:
			raise ValueError("No VAE model found. Please specify vae_path.")
		if t5_path is None:
			raise ValueError("No T5 encoder found. Please specify t5_path.")
		if input_image is not None and clip_path is None:
			raise ValueError("No CLIP encoder found for I2V. Please specify clip_path.")

		# 1. Load all required models
		print("Loading models...")
		model_info = self.load_model(
			model_path=model_path,
			base_precision=base_precision,
			load_device="main_device",
			quantization=quantization,
			attention_mode=attention_mode,
			blocks_to_swap=blocks_to_swap
		)

		vae = self.load_vae(vae_path=vae_path, precision=vae_precision)

		t5_encoder = self.load_t5_encoder(
			t5_path=t5_path,
			precision=t5_precision,
			load_device="offload_device",
			quantization="disabled"
		)

		# Determine if we need CLIP encoder
		is_i2v = model_info["model_type"] == "i2v"
		need_clip = is_i2v and input_image is not None

		clip_model = None
		if need_clip:
			clip_model = self.load_clip_encoder(
				clip_path=clip_path,
				precision=clip_precision,
				load_device="offload_device"
			)

		# 2. Encode text prompts
		text_embeds = self.encode_text(
			t5_encoder=t5_encoder,
			positive_prompt=positive_prompt,
			negative_prompt=negative_prompt,
			force_offload=force_offload,
			use_onnx=use_onnx
		)

		# 3. Prepare image embeds or empty embeds
		if is_i2v and input_image is not None:
			image_embeds = self.encode_image_clip(
				clip=clip_model,
				vae=vae,
				image=input_image,
				num_frames=num_frames,
				generation_width=width,
				generation_height=height,
				force_offload=force_offload,
				noise_aug_strength=noise_aug_strength,
				latent_strength=latent_strength,
				clip_embed_strength=clip_embed_strength,
				use_onnx=use_onnx
			)
		else:
			image_embeds = self.create_empty_embeds(
				width=width,
				height=height,
				num_frames=num_frames
			)

		# 4. Run the sampler
		samples = self.sample(
			model_info=model_info,
			text_embeds=text_embeds,
			image_embeds=image_embeds,
			shift=shift,
			steps=steps,
			cfg=cfg,
			seed=seed,
			scheduler=scheduler,
			riflex_freq_index=riflex_freq_index,
			force_offload=force_offload,
			samples=input_video,
			denoise_strength=denoise_strength,
			use_onnx=use_onnx
		)

		# 5. Decode the video frames
		video_frames = self.decode(
			vae=vae,
			samples=samples,
			enable_vae_tiling=enable_vae_tiling,
			tile_x=tile_x,
			tile_y=tile_y,
			tile_stride_x=tile_stride_x,
			tile_stride_y=tile_stride_y,
			use_onnx=use_onnx
		)

		print(f"Video generated successfully! Shape: {video_frames.shape}")

		# If original_num_frames is different from adjusted num_frames, trim the extra frames
		if original_num_frames != num_frames and video_frames.shape[0] > original_num_frames:
			video_frames = video_frames[:original_num_frames]
			print(f"Trimmed output to requested {original_num_frames} frames")

		# Save video if path is provided
		if save_path:
			try:
				self.save_video(video_frames, save_path, fps=fps, format=output_format)
				print(f"Video saved to {save_path}")
			except Exception as e:
				print(f"Error saving video: {e}")

		return video_frames


def main():
	parser = argparse.ArgumentParser(description="Generate videos with WanVideo models")
	parser.add_argument("--models_dir", type=str, required=True, help="Directory with model files")
	parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu)")
	parser.add_argument("--positive_prompt", type=str, default="a man standing in front of a ford gt surprised",
	                    help="Positive prompt")
	parser.add_argument("--negative_prompt", type=str, default="bad quality video", help="Negative prompt")
	parser.add_argument("--width", type=int, default=512, help="Output width")
	parser.add_argument("--height", type=int, default=512, help="Output height")
	parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
	parser.add_argument("--steps", type=int, default=10, help="Number of sampling steps")
	parser.add_argument("--cfg", type=float, default=6.0, help="Classifier-free guidance scale")
	parser.add_argument("--shift", type=float, default=3.0, help="Scheduler shift parameter")
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("--scheduler", type=str, default="dpm++", choices=["dpm++", "dpm++_sde", "unipc"],
	                    help="Sampling scheduler")
	parser.add_argument("--blocks_to_swap", type=int, default=30, help="Number of blocks to swap to CPU to save VRAM")
	parser.add_argument("--input_image", type=str, default=None, help="Optional input image for I2V")
	parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Noise augmentation strength for I2V")
	parser.add_argument("--output", type=str, required=True, help="Output path for the video")
	parser.add_argument("--fps", type=int, default=16, help="Frames per second for output video")
	args = parser.parse_args()

	# Specify paths to match ComfyUI workflow
	model_path = os.path.join(args.models_dir, "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors")
	vae_path = os.path.join(args.models_dir, "vae", "Wan2_1_VAE_fp32.safetensors")
	t5_path = os.path.join(args.models_dir, "text_encoders", "umt5-xxl-enc-fp8_e4m3fn.safetensors")
	clip_path = os.path.join(args.models_dir, "clip", "open-clip-xlm-roberta-large-vit-huge-14_fp16.safetensors")

	# Define a fixed path for ONNX models
	onnx_models_dir = os.path.join(args.models_dir, "onnx_models")
	os.makedirs(onnx_models_dir, exist_ok=True)

	# Check if files exist, otherwise search for them
	if not os.path.exists(model_path):
		model_files = glob.glob(os.path.join(args.models_dir, "*.safetensors"))
		if model_files:
			model_path = model_files[0]
			print(f"Using model: {model_path}")

	if not os.path.exists(vae_path):
		vae_files = glob.glob(os.path.join(args.models_dir, "vae", "*.safetensors"))
		if vae_files:
			vae_path = vae_files[0]
			print(f"Using VAE: {vae_path}")

	if not os.path.exists(t5_path):
		t5_files = glob.glob(os.path.join(args.models_dir, "text_encoders", "*.safetensors"))
		if not t5_files:
			t5_files = glob.glob(os.path.join(args.models_dir, "t5", "*.safetensors"))
		if t5_files:
			t5_path = t5_files[0]
			print(f"Using T5: {t5_path}")

	if not os.path.exists(clip_path) and args.input_image:
		clip_files = glob.glob(os.path.join(args.models_dir, "clip", "*.safetensors"))
		if clip_files:
			clip_path = clip_files[0]
			print(f"Using CLIP: {clip_path}")

	# Clear CUDA cache before starting
	torch.cuda.empty_cache()
	gc.collect()
	print(f"Initial VRAM: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

	# Create generator with explicit device and use our custom ONNX dir
	generator = WanVideoGenerator(
		models_dir=args.models_dir,
		device=args.device
	)
	# Set the ONNX directory
	generator.onnx_dir = onnx_models_dir

	# Check if ONNX models exist, if not, create them
	onnx_models_exist = all([
		any(glob.glob(os.path.join(onnx_models_dir, "*unet*.onnx"))) or
		any(glob.glob(os.path.join(onnx_models_dir, "*unet*.engine"))),
		any(glob.glob(os.path.join(onnx_models_dir, "*vae_decoder*.onnx"))) or
		any(glob.glob(os.path.join(onnx_models_dir, "*vae_decoder*.engine"))),
		any(glob.glob(os.path.join(onnx_models_dir, "*t5_encoder*.onnx"))) or
		any(glob.glob(os.path.join(onnx_models_dir, "*t5_encoder*.engine"))),
	])

	# Add CLIP check only if we're using an image input
	if args.input_image:
		onnx_models_exist = onnx_models_exist and (
				any(glob.glob(os.path.join(onnx_models_dir, "*clip*.onnx"))) or
				any(glob.glob(os.path.join(onnx_models_dir, "*clip*.engine")))
		)

	# If ONNX models don't exist, export them
	if not onnx_models_exist:
		print("ONNX models not found. Converting models to ONNX format...")
		generator.export_models(
			model_path=model_path,
			vae_path=vae_path,
			t5_path=t5_path,
			clip_path=clip_path if args.input_image else None,
			width=args.width,
			height=args.height,
			num_frames=args.num_frames,
			export_path=onnx_models_dir
		)
		print("ONNX conversion complete.")

	# Load input image if provided
	input_img = None
	if args.input_image:
		input_img = Image.open(args.input_image).convert("RGB")

	# Determine whether to use ONNX/TensorRT based on availability
	use_onnx = TRT_AVAILABLE or onnx_models_exist

	# Generate video with our TensorRT/ONNX implementation if available, otherwise use PyTorch
	generator.generate_video(
		model_path=model_path,
		vae_path=vae_path,
		t5_path=t5_path,
		clip_path=clip_path,
		base_precision="fp16",  # Use fp16 for base precision
		vae_precision="bf16",
		t5_precision="bf16",
		clip_precision="fp16",
		quantization="disabled" if use_onnx else "fp8_e4m3fn",  # Disable custom FP8 if using ONNX
		attention_mode="sdpa",
		blocks_to_swap=args.blocks_to_swap,
		positive_prompt=args.positive_prompt,
		negative_prompt=args.negative_prompt,
		width=args.width,
		height=args.height,
		num_frames=args.num_frames,
		steps=args.steps,
		cfg=args.cfg,
		shift=args.shift,
		seed=args.seed,
		scheduler=args.scheduler,
		force_offload=True,  # Always offload models to CPU when not in use
		input_image=input_img,
		noise_aug_strength=args.noise_aug_strength,
		save_path=args.output,
		fps=args.fps,
		use_onnx=use_onnx  # Enable ONNX/TensorRT if available
	)


if __name__ == "__main__":
	main()