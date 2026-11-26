"""Native Ref2VA reference encoding and video decoding."""

import torch
import numpy as np
from accelerate import dispatch_model
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep


class Encoder:
    def __init__(self, model_path, device, text_devices):
        blocks = (
            ModularPipeline.from_pretrained(model_path).blocks.get_workflow("ref2va").sub_blocks
        )
        stages = ("before_encode", "text_encoder", "vae_encoder")
        self.conditioned = SequentialPipelineBlocks.from_blocks_dict(
            {name: blocks[name] for name in stages}
        ).init_pipeline(model_path, components_manager=ComponentsManager())
        self.conditioned.load_components(dtype=torch.bfloat16)
        encoder = self.conditioned.text_encoder
        encoder.set_attn_implementation("sdpa")
        mapping = {
            "model.visual": device,
            "model.language_model.embed_tokens": device,
            "model.language_model.norm": device,
            "model.language_model.rotary_emb": device,
            "lm_head": device,
        }
        layers = len(encoder.model.language_model.layers)
        for index in range(layers):
            mapping[f"model.language_model.layers.{index}"] = text_devices[
                index * len(text_devices) // layers
            ]
        dispatch_model(encoder, device_map=mapping, main_device=device, force_hooks=True)
        self.conditioned.vae.to(device)

    @torch.no_grad()
    def conditions(self, reference, proxy, metadata, prompt):
        self.conditioned.register_to_config(
            canvas_short_edge=min(metadata["width"], metadata["height"]),
            canvas_max_pixels=metadata["width"] * metadata["height"],
        )
        state = self.conditioned(
            prompt=prompt,
            references=[
                MiniMaxH3ImageReference(image=reference),
                MiniMaxH3VideoReference(frames=np.asarray(proxy), fps=24),
            ],
            height=metadata["height"],
            width=metadata["width"],
            num_frames=metadata["frames"],
        )
        return {
            key: state.get(key)
            for key in (
                "prompt_embeds",
                "text_token_tags",
                "condition_latents",
                "audio_condition_latents",
                "normalized_references",
            )
        }


class Decoder:
    def __init__(self, model_path, device):
        self.pipe = SequentialPipelineBlocks.from_blocks_dict(
            {"decode": MiniMaxH3VideoDecodeStep()}
        ).init_pipeline(model_path)
        self.pipe.load_components(dtype=torch.float32)
        self.pipe.vae.to(device)

    @torch.no_grad()
    def __call__(self, latents):
        return self.pipe(latents=latents, output_type="pil").get("videos")[0]
