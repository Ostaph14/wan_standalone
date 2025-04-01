#!/usr/bin/env python3
import os
import argparse
import json
import time
from PIL import Image

# Import the WanVideo Generator
from testwan import WanVideoGenerator


class BatchGenerator:
	def __init__(self, models_dir, device="cuda:0"):
		"""Initialize the batch generator with a persistent WanVideoGenerator instance"""
		self.models_dir = models_dir
		self.device = device

		# Create the generator once
		self.generator = WanVideoGenerator(
			models_dir=models_dir,
			device=device
		)

		# Standard paths for models
		self.model_paths = {
			"i2v": os.path.join(models_dir, "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"),
			"t2v": os.path.join(models_dir, "Wan2_1-T2V-14B-480P_fp8_e4m3fn.safetensors"),
			"vae": os.path.join(models_dir, "vae", "Wan2_1_VAE_fp32.safetensors"),
			"t5": os.path.join(models_dir, "text_encoders", "umt5-xxl-enc-fp8_e4m3fn.safetensors"),
			"clip": os.path.join(models_dir, "clip", "open-clip-xlm-roberta-large-vit-huge-14_fp16.safetensors"),
		}

		# Keep track of what models are loaded to avoid unnecessary reloading
		self.loaded_models = {
			"diffusion_model": None,  # Track model type: "i2v" or "t2v"
			"vae": False,
			"t5_encoder": False,
			"clip_model": False
		}

	def generate_from_config(self, config):
		"""Generate a video from a configuration dictionary"""
		# Determine model type and path
		model_type = config.get("model_type", "t2v" if config.get("input_image") is None else "i2v")
		model_path = self.model_paths[model_type]

		# Special handling for model loading - only reload if we need a different type
		load_diffusion = (self.loaded_models["diffusion_model"] != model_type)

		# Load input image if specified
		input_image = None
		if "input_image" in config and config["input_image"]:
			input_image = Image.open(config["input_image"]).convert("RGB")

		# Update config with defaults and model paths
		generation_config = {
			# Model paths
			"model_path": model_path if load_diffusion else None,  # Only provide if we need to load
			"vae_path": self.model_paths["vae"] if not self.loaded_models["vae"] else None,
			"t5_path": self.model_paths["t5"] if not self.loaded_models["t5_encoder"] else None,
			"clip_path": self.model_paths["clip"] if not self.loaded_models[
				"clip_model"] and model_type == "i2v" else None,

			# Default settings
			"base_precision": "fp16",
			"vae_precision": "bf16",
			"t5_precision": "bf16",
			"clip_precision": "fp16",
			"quantization": "fp8_e4m3fn_fast",
			"attention_mode": "sdpa",
			"blocks_to_swap": 30,
			"force_offload": False,  # Don't offload models between generations

			# Pass through user config
			**config,

			# Always use the prepared input_image
			"input_image": input_image,
		}

		# Generate the video
		start_time = time.time()
		print(f"Starting generation for output: {config.get('save_path')}")
		print(f"Prompt: {config.get('positive_prompt', 'Default prompt')}")

		video_frames = self.generator.generate_video(**generation_config)

		# Update the loaded model tracking
		if load_diffusion:
			self.loaded_models["diffusion_model"] = model_type
		if not self.loaded_models["vae"] and generation_config["vae_path"]:
			self.loaded_models["vae"] = True
		if not self.loaded_models["t5_encoder"] and generation_config["t5_path"]:
			self.loaded_models["t5_encoder"] = True
		if not self.loaded_models["clip_model"] and generation_config["clip_path"]:
			self.loaded_models["clip_model"] = True

		elapsed = time.time() - start_time
		print(f"Generation completed in {elapsed:.2f} seconds")

		return video_frames

	def generate_batch(self, batch_file):
		"""Generate videos from a batch configuration file"""
		with open(batch_file, 'r') as f:
			configs = json.load(f)

		results = []
		for i, config in enumerate(configs):
			print(f"\n--- Generating video {i + 1}/{len(configs)} ---")
			result = self.generate_from_config(config)
			results.append(result)

		return results

	def cleanup(self):
		"""Clean up resources"""
		import gc
		import torch

		# Clear the model cache
		self.generator.loaded_models = {}
		self.loaded_models = {k: False for k in self.loaded_models}
		self.loaded_models["diffusion_model"] = None

		# Force garbage collection
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		print("Models unloaded and memory cleared")


def main():
	parser = argparse.ArgumentParser(description="Batch generate videos with WanVideo models")
	parser.add_argument("--models_dir", type=str, required=True, help="Directory with model files")
	parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu)")
	parser.add_argument("--config_file", type=str, required=True, help="JSON file with batch configuration")

	args = parser.parse_args()

	# Create the batch generator
	batch_generator = BatchGenerator(
		models_dir=args.models_dir,
		device=args.device
	)

	try:
		# Generate all videos in the batch
		batch_generator.generate_batch(args.config_file)
	finally:
		# Make sure we clean up even if an error occurs
		batch_generator.cleanup()


if __name__ == "__main__":
	main()