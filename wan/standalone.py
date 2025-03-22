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

# Add after imports
if torch.cuda.is_available():
	# Set optimal TF32 mode for Ampere+ GPUs
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	# Enable cuDNN benchmarking for best performance
	torch.backends.cudnn.benchmark = True

	# Disable gradient synchronization
	torch.cuda.set_sync_debug_mode(0)

	# Set device to high priority stream
	current_stream = torch.cuda.current_stream()
	current_stream.priority = -1  # High priority

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
	Enhanced FP8 optimization with better memory scaling
	"""
	setattr(module, "fp8_matmul_enabled", True)

	# Cache for intermediate calculations
	if not hasattr(module, "_fp8_cache"):
		module._fp8_cache = {}

	for name, sub_module in module.named_modules():
		if not any(keyword in name for keyword in params_to_keep):
			if isinstance(sub_module, torch.nn.Linear):
				# Store the original forward function
				original_forward = sub_module.forward
				setattr(sub_module, "original_forward", original_forward)

				# Use better precision for small matrices
				weight_size = sub_module.weight.numel()
				use_fp16 = weight_size < 1000000  # Use fp16 for smaller matrices

				# Enhanced forward with caching for common input shapes
				def enhanced_forward(input, m=sub_module, use_fp16=use_fp16):
					input_shape = input.shape
					cache_key = (input_shape, input.device)

					# Check for cached calculations
					if hasattr(m, "_fp8_cache") and cache_key in m._fp8_cache:
						cached_dtype = m._fp8_cache[cache_key]
					else:
						cached_dtype = torch.float16 if use_fp16 else original_dtype
						if hasattr(m, "_fp8_cache"):
							m._fp8_cache[cache_key] = cached_dtype

					# Run with appropriate precision
					return m.original_forward(input.to(cached_dtype))

				setattr(sub_module, "forward", enhanced_forward)

# Memory management utilities with aggressive cleanup
def print_memory(device):
	if torch.cuda.is_available():
		print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
		print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
		print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")


def soft_empty_cache():
    """Aggressively clean GPU memory with better timing"""
    if torch.cuda.is_available():
        # Add synchronization before cleanup
        torch.cuda.synchronize()
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


class BlockSwapManager:
	def __init__(self, model, blocks_to_swap, device, offload_device="cpu"):
		self.model = model
		self.max_active_blocks = blocks_to_swap
		self.device = device
		self.offload_device = offload_device
		self.active_blocks = set()
		self.block_map = {}

		# Map all block parameters
		for name, param in model.named_parameters():
			if "blocks." in name:
				block_id = int(name.split("blocks.")[1].split(".")[0])
				if block_id not in self.block_map:
					self.block_map[block_id] = []
				self.block_map[block_id].append(name)

	def activate_block(self, block_id):
		"""Move a specific block to GPU"""
		if block_id in self.active_blocks:
			return

		# Ensure we don't exceed max active blocks
		while len(self.active_blocks) >= self.max_active_blocks:
			# Remove oldest block (simple FIFO)
			oldest = min(self.active_blocks)
			self.deactivate_block(oldest)

		# Move block to device
		if block_id in self.block_map:
			for param_name in self.block_map[block_id]:
				param = self.model.get_parameter(param_name)
				param.data = param.data.to(self.device)
			self.active_blocks.add(block_id)

	def deactivate_block(self, block_id):
		"""Move a specific block to CPU"""
		if block_id not in self.active_blocks:
			return

		if block_id in self.block_map:
			for param_name in self.block_map[block_id]:
				param = self.model.get_parameter(param_name)
				param.data = param.data.to(self.offload_device)
			self.active_blocks.remove(block_id)

	def activate_sequential_blocks(self, start_block, count):
		"""Activate a sequence of blocks for processing"""
		for i in range(start_block, min(start_block + count, max(self.block_map.keys()) + 1)):
			self.activate_block(i)

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

	def load_model(self, model_path, base_precision="bf16", load_device="main_device",
	               quantization="disabled", attention_mode="sdpa", blocks_to_swap=0):
		"""
		Load the WanVideo transformer model with enhanced memory management.
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

			# Use blocks_to_swap if provided, otherwise use default
			effective_blocks_to_swap = blocks_to_swap if blocks_to_swap > 0 else self.default_blocks_to_swap

			# Load parameters with appropriate dtypes
			for name, param in transformer.named_parameters():
				dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype

				# Check if this is a block parameter - if so, keep on CPU for now if using block_swap
				if effective_blocks_to_swap > 0 and "blocks." in name:
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
		block_swap_args = None
		if blocks_to_swap > 0 or self.default_blocks_to_swap > 0:
			effective_blocks = blocks_to_swap if blocks_to_swap > 0 else self.default_blocks_to_swap
			block_swap_args = {"blocks_to_swap": effective_blocks}

			# Initialize block manager
			self.block_manager = BlockSwapManager(
				transformer,
				max_active_blocks=effective_blocks,
				device=self.device,
				offload_device=self.offload_device
			)

		model_info = {
			"model": transformer,
			"dtype": base_dtype,
			"model_path": model_path,
			"model_name": os.path.basename(model_path),
			"manual_offloading": manual_offloading,
			"quantization": quantization,
			"block_swap_args": block_swap_args,
			"model_type": model_type,
			"num_layers": num_layers
		}

		del sd
		gc.collect()
		soft_empty_cache()

		# Apply our custom FP8 optimization
		if quantization in ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_scaled"]:
			print("Applying optimized FP8 optimization to linear layers...")
			params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb",
			                  "modulation"}
			convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
			print("FP8 optimization applied.")

		# Use flash attention if available on compatible hardware
		if torch.cuda.get_device_capability(0)[0] >= 8 and attention_mode == "sdpa":
			print("Enabling Flash Attention optimizations")
			transformer.use_flash_attention = True

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

	def encode_text(self, t5_encoder, positive_prompt, negative_prompt, force_offload=True):
		"""Encode text prompts using the T5 encoder."""
		print(f"Encoding prompts:\nPositive: {positive_prompt}\nNegative: {negative_prompt}")

		encoder = t5_encoder["model"]
		dtype = t5_encoder["dtype"]

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
	                      force_offload=True, noise_aug_strength=0.0, latent_strength=1.0, clip_embed_strength=1.0):
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

		# Process with CLIP and immediately offload
		clip.model.to(self.device)
		pixel_values = clip_preprocess(image.to(self.device), size=224, mean=image_mean, std=image_std,
		                               crop=True).float()

		# Use CLIP to get embeddings
		with torch.no_grad(), torch.cuda.amp.autocast():
			clip_context = clip.visual(pixel_values)

		if clip_embed_strength != 1.0:
			clip_context *= clip_embed_strength

		# Immediately offload CLIP model
		if force_offload:
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
		# Use mixed precision encoding to save memory
		with torch.cuda.amp.autocast():
			concatenated = torch.concat([resized_image, zero_frames], dim=1).to(dtype=vae.dtype)
			concatenated *= latent_strength
			y = vae.encode([concatenated], self.device)[0]
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

	def sample(self, model_info, text_embeds, image_embeds, shift, steps, cfg, seed=None,
	           scheduler="dpm++", riflex_freq_index=0, force_offload=True,
	           samples=None, denoise_strength=1.0):
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

        Returns:
            Generated video latents
        """
		print(f"Running sampling for {steps} steps with {scheduler} scheduler...")

		if seed is None:
			seed = int(torch.randint(0, 2147483647, (1,)).item())

		print(f"Using seed: {seed}")

		transformer = model_info["model"]

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

		if transformer.model_type == "i2v":
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

		if transformer.model_type == "i2v":
			base_args.update({
				'y': [image_embeds["image_embeds"]],
			})

		arg_c = base_args.copy()
		arg_c.update({'context': [text_embeds["prompt_embeds"][0]]})

		arg_null = base_args.copy()
		arg_null.update({'context': text_embeds["negative_prompt_embeds"]})

		# Handle block swapping - crucial for memory efficiency in ComfyUI
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
		# Replace your current sampling loop with this optimized version
		with torch.no_grad():
			with torch.autocast(device_type=self.device.split(':')[0], dtype=model_info["dtype"], enabled=True):
				# Pre-allocate tensors and initialize block manager
				latent_gpu = torch.zeros_like(latent, device=self.device)
				block_manager = BlockSwapManager(transformer, blocks_to_swap, self.device, self.offload_device)

				# Process timesteps in batches where possible
				timestep_batches = []
				batch_size = min(4, len(timesteps))  # Group timesteps into small batches
				for i in range(0, len(timesteps), batch_size):
					timestep_batches.append(timesteps[i:i + batch_size])

				for batch_idx, timestep_batch in enumerate(tqdm(timestep_batches)):
					# Prepare for batch processing
					start_layer = 0
					end_layer = transformer.num_layers

					# Activate early blocks for this timestep batch
					block_manager.activate_sequential_blocks(0, min(10, transformer.num_layers))

					# Process each timestep in small batches
					for i, t in enumerate(timestep_batch):
						# Copy latent to GPU
						latent_gpu.copy_(latent.to(self.device))
						latent_model_input = [latent_gpu]
						timestep = torch.tensor([t], device=self.device)

						# Get conditional prediction
						noise_pred_cond = transformer(
							latent_model_input, t=timestep, **arg_c)[0].to(self.offload_device)

						# Apply CFG if needed
						if cfg[batch_idx * batch_size + i] != 1.0:
							# Reuse latent_gpu for unconditional pass
							latent_gpu.copy_(latent.to(self.device))
							latent_model_input = [latent_gpu]

							# Get unconditional prediction
							noise_pred_uncond = transformer(
								latent_model_input, t=timestep, **arg_null)[0].to(self.offload_device)

							# Apply CFG
							noise_pred = noise_pred_uncond + cfg[batch_idx * batch_size + i] * (
									noise_pred_cond - noise_pred_uncond)
						else:
							noise_pred = noise_pred_cond

						# Sample and update latent
						noise_pred = noise_pred.to(self.device)
						latent_gpu.copy_(latent.to(self.device))

						# Perform sampling step
						latent = sample_scheduler.step(
							noise_pred.unsqueeze(0),
							t,
							latent_gpu.unsqueeze(0),
							return_dict=False,
							generator=seed_g)[0].squeeze(0).cpu()

						# Clear intermediate tensors
						del noise_pred
						torch.cuda.synchronize()

		# Move final result to GPU for one last processing
		latent_gpu = latent.to(self.device)
		result = latent_gpu.unsqueeze(0).cpu()

		# Clean up
		del latent_gpu
		if force_offload:
			if model_info["manual_offloading"]:
				transformer.to(self.offload_device)
			soft_empty_cache()

		return {"samples": result}

	def decode(self, vae, samples, enable_vae_tiling=True,
	           tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128):
		"""Optimized decode function with overlap handling"""
		print("Decoding video frames...")

		soft_empty_cache()
		latents = samples["samples"]

		# Move VAE to device for decoding
		vae.to(self.device)

		# Calculate optimal tile configuration
		frames, channels, height, width = latents.shape

		# Use adaptive tiling based on available memory
		if enable_vae_tiling:
			free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(
				self.device)
			free_memory_gb = free_memory / (1024 ** 3)

			# Adjust tile size based on available memory
			if free_memory_gb > 8:
				# More memory available, use larger tiles
				tile_x = min(tile_x * 2, width)
				tile_y = min(tile_y * 2, height)
			elif free_memory_gb < 2:
				# Very limited memory, use smaller tiles
				tile_x = max(tile_x // 2, 64)
				tile_y = max(tile_y // 2, 64)

			# Ensure stride is appropriate for tile size
			tile_stride_x = min(tile_stride_x, tile_x // 2)
			tile_stride_y = min(tile_stride_y, tile_y // 2)

		# Process in frame batches to reduce memory peaks
		batch_size = 1
		if frames <= 16:
			batch_size = frames  # Process all at once for short videos
		elif frames <= 64:
			batch_size = 4  # Process in batches of 4 for medium videos
		else:
			batch_size = 2  # Process in batches of 2 for long videos

		results = []

		# Process in batches
		for i in range(0, frames, batch_size):
			end_idx = min(i + batch_size, frames)
			batch = latents[i:end_idx].to(device=self.device, dtype=vae.dtype)

			# Use mixed precision for decoding
			with torch.cuda.amp.autocast():
				decoded_batch = vae.decode(
					batch,
					device=self.device,
					tiled=enable_vae_tiling,
					tile_size=(tile_x, tile_y),
					tile_stride=(tile_stride_x, tile_stride_y)
				)[0]

			# Normalize and format
			decoded_batch = (decoded_batch - decoded_batch.min()) / (decoded_batch.max() - decoded_batch.min())
			decoded_batch = torch.clamp(decoded_batch, 0.0, 1.0)
			decoded_batch = decoded_batch.permute(1, 2, 3, 0).cpu().float()

			results.append(decoded_batch)

			# Clear batch from GPU
			del batch, decoded_batch
			soft_empty_cache()

		# Concatenate results
		image = torch.cat(results, dim=0)

		# Clean up
		vae.to(self.offload_device)
		vae.model.clear_cache()
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
	                   fps=16
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

		# Rest of the method remains unchanged
		# ...

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
			force_offload=force_offload
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
				clip_embed_strength=clip_embed_strength
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
			denoise_strength=denoise_strength
		)

		# 5. Decode the video frames
		video_frames = self.decode(
			vae=vae,
			samples=samples,
			enable_vae_tiling=enable_vae_tiling,
			tile_x=tile_x,
			tile_y=tile_y,
			tile_stride_x=tile_stride_x,
			tile_stride_y=tile_stride_y
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

	def save_video(self, video_frames, save_path, fps=16, format="mp4"):
		"""Save video frames to a file."""
		frames = (video_frames * 255).to(torch.uint8).numpy()

		if format == "frames":
			os.makedirs(save_path, exist_ok=True)
			for i, frame in enumerate(frames):
				img = Image.fromarray(frame)
				img.save(os.path.join(save_path, f"frame_{i:05d}.png"))
			return

		try:
			import imageio

			if format == "gif":
				imageio.mimsave(save_path, frames, fps=fps)
			elif format == "mp4":
				imageio.mimsave(save_path, frames, fps=fps, codec="libx264", quality=7)
			else:
				raise ValueError(f"Unsupported format: {format}")
		except ImportError:
			print("Please install imageio for video saving: pip install imageio imageio-ffmpeg")
			# Fall back to saving frames
			dir_path = os.path.splitext(save_path)[0] + "_frames"
			os.makedirs(dir_path, exist_ok=True)
			for i, frame in enumerate(frames):
				img = Image.fromarray(frame)
				img.save(os.path.join(dir_path, f"frame_{i:05d}.png"))


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
	parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift parameter")
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

	# Clear CUDA cache before starting
	torch.cuda.empty_cache()
	gc.collect()

	print(f"Initial VRAM: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

	# Create generator with explicit device
	generator = WanVideoGenerator(
		models_dir=args.models_dir,
		device=args.device
	)

	# Load input image if provided
	input_img = None
	if args.input_image:
		input_img = Image.open(args.input_image).convert("RGB")

	# Generate video with our custom FP8 implementation
	generator.generate_video(
		model_path=model_path,
		vae_path=vae_path,
		t5_path=t5_path,
		clip_path=clip_path,
		base_precision="fp16",  # Use fp16 for base precision
		vae_precision="bf16",
		t5_precision="bf16",
		clip_precision="fp16",
		quantization="fp8_e4m3fn",  # Use our custom FP8 implementation
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
		fps=args.fps
	)


if __name__ == "__main__":
	main()
