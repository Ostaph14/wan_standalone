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


class WanI2V:

    def __init__(
            self,
            config,
            checkpoint_dir,
            device_id=1,
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
        primary_device = "cuda:0"
        secondary_device = "cuda:1"
        self.device = torch.device(f"cuda:{device_id}")
        self.primary_device = primary_device
        self.secondary_device = secondary_device

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
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        # Modified approach to load model across two GPUs
        if not (t5_fsdp or dit_fsdp or use_usp):
            logging.info(f"Creating split WanModel from {checkpoint_dir} across cuda:0 and cuda:1")
            # Monkey patch the WanModel.from_pretrained method to load directly to distributed devices
            original_from_pretrained = WanModel.from_pretrained

            def distributed_from_pretrained(cls, checkpoint_dir, **kwargs):
                # First load model to CPU to analyze structure
                # Correctly call the original method with just the right arguments
                model = original_from_pretrained(checkpoint_dir, **kwargs)
                model.eval().requires_grad_(False)

                # Get number of blocks
                total_blocks = len(model.blocks)
                split_idx = total_blocks // 2

                # Create a new distributed forward method
                original_forward = model.forward

                def custom_forward(self, x, t=None, context=None, clip_fea=None, seq_len=None, y=None):
                    """Custom forward that handles device transitions"""
                    # Ensure inputs are on the right device
                    if isinstance(x, list):
                        x_dev = [tensor.to(primary_device) for tensor in x]
                    else:
                        x_dev = x.to(primary_device)

                    t_dev = t.to(primary_device) if t is not None else None

                    if context is not None and isinstance(context, list):
                        context_dev = [tensor.to(primary_device) for tensor in context]
                    else:
                        context_dev = context.to(primary_device) if context is not None else None

                    clip_fea_dev = clip_fea.to(primary_device) if clip_fea is not None else None

                    if y is not None and isinstance(y, list):
                        y_dev = [tensor.to(primary_device) for tensor in y]
                    else:
                        y_dev = y.to(primary_device) if y is not None else None

                    # Call the original forward with our device-specific versions
                    # This is a partial forward pass we'll intercept with hooks
                    result = original_forward(self, x_dev, t_dev, context_dev, clip_fea_dev, seq_len, y_dev)

                    return result

                # Replace forward method
                model.forward = types.MethodType(custom_forward, model)

                # Track execution flow through model
                activation_dict = {}
                hooks = []

                # Capture input to first block on cuda:0
                def pre_hook_first_block(module, input):
                    activation_dict['first_block_input'] = input
                    return input

                # Capture output from last block on cuda:0
                def hook_middle_block(module, input, output):
                    activation_dict['middle_output'] = output
                    # Transfer to cuda:1
                    return output.to(secondary_device)

                # Register hooks
                hooks.append(model.blocks[0].register_forward_pre_hook(pre_hook_first_block))
                hooks.append(model.blocks[split_idx - 1].register_forward_hook(hook_middle_block))

                # Now distribute the model across devices
                # First half of blocks to cuda:0
                for i in range(split_idx):
                    model.blocks[i] = model.blocks[i].to(primary_device)

                # Second half of blocks to cuda:1
                for i in range(split_idx, total_blocks):
                    model.blocks[i] = model.blocks[i].to(secondary_device)

                # Any remaining top-level components split based on dependency
                # This requires model-specific knowledge
                # For now, assuming the model has a simple structure where only blocks are important

                # Clean up hooks after initialization
                for hook in hooks:
                    hook.remove()

                return model

            # Replace the from_pretrained method temporarily
            WanModel.from_pretrained = classmethod(distributed_from_pretrained)

            # Now create the model using our distributed loading
            self.model = WanModel.from_pretrained(checkpoint_dir)

            # Restore original method
            WanModel.from_pretrained = original_from_pretrained

            # Now create our custom forward implementation that handles the split
            original_forward = self.model.forward

            def distributed_forward(self, x, t=None, context=None, clip_fea=None, seq_len=None, y=None):
                """Distributed forward pass implementation"""
                # Process inputs on primary device
                if isinstance(x, list):
                    x_primary = [tensor.to(primary_device) for tensor in x]
                else:
                    x_primary = x.to(primary_device)

                t_primary = t.to(primary_device) if t is not None else None

                if context is not None and isinstance(context, list):
                    context_primary = [tensor.to(primary_device) for tensor in context]
                    context_secondary = [tensor.to(secondary_device) for tensor in context]
                else:
                    context_primary = context.to(primary_device) if context is not None else None
                    context_secondary = context.to(secondary_device) if context is not None else None

                clip_fea_primary = clip_fea.to(primary_device) if clip_fea is not None else None
                y_primary = y[0].to(primary_device) if y is not None and isinstance(y, list) else y

                # We need to capture intermediate state at the split point
                activation_dict = {}

                def hook_middle_block(module, input, output):
                    activation_dict['middle_output'] = output
                    # Transfer to cuda:1
                    return output.to(secondary_device)

                # Add hook to the last block in the first half
                total_blocks = len(self.blocks)
                split_idx = total_blocks // 2
                hook = self.blocks[split_idx - 1].register_forward_hook(hook_middle_block)

                # Run forward pass (will be intercepted by our hook)
                result = original_forward(self, x_primary, t_primary, context_primary,
                                          clip_fea_primary, seq_len, y_primary)

                # Remove hook
                hook.remove()

                return result

            # Replace forward with our distributed version
            self.model.forward = types.MethodType(distributed_forward, self.model)
        else:
            # Standard loading for FSDP or USP
            logging.info(f"Creating WanModel from {checkpoint_dir}")
            self.model = WanModel.from_pretrained(checkpoint_dir)
            self.model.eval().requires_grad_(False)

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

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        # We've handled model loading and distribution ourselves

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

        # preprocess
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
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
