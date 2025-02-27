# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from safetensors import safe_open
import os
import json

# Set higher log level
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.set_printoptions(profile="full")

# Add error handler for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    import traceback
    print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    # Force exit
    os._exit(1)

sys.excepthook = handle_exception


from safetensors.torch import load_file
# Create model with config or defaults
from .modules.model import WanModel
# Define a custom loader function for the quantized model
def load_quantized_wan_model(checkpoint_dir, model_type='i2v'):
    """Load the quantized model with explicit dimensions to match the weights"""
    import psutil  # Add this import

    # Get local rank for logging
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Debug memory usage
    if local_rank == 0:
        print(f"Memory before model creation: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

    # Create model with explicit dimensions matching the quantized weights
    model = WanModel(
        model_type='i2v',
        in_dim=36,  # Match input tensor channels
        dim=5120,  # Hidden dimension
        ffn_dim=13824,  # FFN dimension
        out_dim=16,  # Output channels
        num_layers=40,  # Match layer count
        # Keep other params default
    )

    if local_rank == 0:
        print(f"Memory after model creation: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

    # Load weights with chunking to reduce memory usage
    quant_file = os.path.join(checkpoint_dir, "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors")
    if os.path.exists(quant_file):
        if local_rank == 0:
            print(f"Loading from: {quant_file}")

        # Use chunk-based loading to minimize memory usage
        from safetensors import safe_open
        model_state = model.state_dict()

        # Group parameters by block for efficient loading
        param_groups = {}
        for name in model_state.keys():
            parts = name.split('.')
            if parts[0] == 'blocks' and len(parts) > 1:
                group = f"blocks.{parts[1]}"
            else:
                group = "base"

            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append(name)

        # Load parameter groups
        total_groups = len(param_groups)
        for i, (group, params) in enumerate(param_groups.items()):
            if local_rank == 0 and i % 10 == 0:
                print(f"Loading group {i + 1}/{total_groups}: {group}")
                print(f"Current memory: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

            try:
                # Load just this group's parameters
                with safe_open(quant_file, framework="pt", device="cpu") as f:
                    for param_name in params:
                        if param_name in f.keys():
                            tensor = f.get_tensor(param_name)
                            if tensor.shape == model_state[param_name].shape:
                                model_state[param_name].copy_(tensor)
            except Exception as e:
                print(f"Error loading group {group}: {e}")

            # Force cleanup after each group
            gc.collect()
            torch.cuda.empty_cache()

        if local_rank == 0:
            print(f"Memory after loading: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

    return model


# First, inspect the shard_model function - this is likely causing our issues
# Add this function at the top of the file to override FSDP settings
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def create_fsdp_model(model, device_id):
    """Create a properly configured FSDP model that works across PyTorch versions"""
    import torch.distributed.fsdp as fsdp
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, default_auto_wrap_policy

    # Define which layers to wrap
    from .modules.model import WanAttentionBlock

    # Create auto_wrap_policy (compatible with different PyTorch versions)
    auto_wrap_policy = default_auto_wrap_policy
    if hasattr(transformer_auto_wrap_policy, "__call__"):
        # For newer PyTorch versions
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={WanAttentionBlock},
        )
    else:
        # For older PyTorch versions
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={WanAttentionBlock},
        )

    # Mixed precision config
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    # Wrapped model with proper FSDP parameters
    fsdp_model = FullyShardedDataParallel(
        model,
        device_id=device_id,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,  # Apply policy here instead
        limit_all_gathers=True,
        use_orig_params=False,
        sync_module_states=True,
        forward_prefetch=True,
        backward_prefetch=True,
    )

    return fsdp_model

class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device="cpu")

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device="cpu",
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if dit_fsdp:
            # Skip FSDP and use a simpler approach that will work reliably
            if self.rank == 0:
                print("Creating model with correct dimensions (skipping FSDP)")

            # Create model with correct parameters
            self.model = WanModel(
                model_type='i2v',
                in_dim=36,
                dim=5120,
                ffn_dim=13824,
                out_dim=16,
                num_layers=40,
            )

            # Convert to half precision for memory efficiency
            self.model = self.model.half()

            if self.rank == 0:
                print("Loading weights directly")

            # Load weights directly
            quant_file = os.path.join(checkpoint_dir, "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors")

            # Load weights efficiently
            try:
                from safetensors.torch import load_file
                weights = load_file(quant_file)

                # Convert to half precision
                for k in weights:
                    weights[k] = weights[k].half()

                self.model.load_state_dict(weights, strict=False)

                if self.rank == 0:
                    print("Successfully loaded weights")

            except Exception as e:
                if self.rank == 0:
                    print(f"Error loading weights: {e}")

            # Move to device
            if not init_on_cpu:
                if self.rank == 0:
                    print(f"Moving model to {self.device}")
                self.model.to(self.device)
                if self.rank == 0:
                    print("Model successfully moved to device")
        else:
            # Normal loading for non-FSDP
            self.model = load_quantized_wan_model(checkpoint_dir)
        if self.rank == 0:
            print("Model loaded, setting to eval mode")
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if self.rank == 0:
            print("Configuring distributed setup")
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if self.rank == 0:
            print("Before moving to device")
        if dit_fsdp:
            if self.rank == 0:
                print("Applying FSDP sharding")
            self.model = shard_fn(self.model)
            if self.rank == 0:
                print("FSDP sharding complete")
        else:
            if not init_on_cpu:
                if self.rank == 0:
                    print(f"Moving model to {self.device}")
                self.model.to(self.device)
                if self.rank == 0:
                    print("Model successfully moved to device")

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        # Add at the top of the method:
        import psutil
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if local_rank == 0:
            print(f"Starting generation with memory: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # seeding random noise
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            21,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        # creating mask
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

            # After creating tensors and before model inference:
            if local_rank == 0:
                print(f"Input shapes: noise={noise.shape}, msk={msk.shape}, y={y.shape}")
                print(f"Memory before inference: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

        # preprocess, y creation
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, 80, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        # After creating tensors and before model inference
        if local_rank == 0:
            print(f"Input shapes: noise={noise.shape}, msk={msk.shape}, y={y.shape}")
            print(f"Memory before inference: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")


            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                # Inference loop
                if local_rank == 0 and t % 5 == 0:
                    print(f"Step {t}/{len(timesteps)}, Memory: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if self.rank == 0:
            print("Before distributed barrier")
        if dist.is_initialized():
            dist.barrier()
            if self.rank == 0:
                print("After distributed barrier")

        return videos[0] if self.rank == 0 else None
