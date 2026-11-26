"""Prepare reference latents and denoise video and audio with H3."""

import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3Ref2VACoreDenoiseStep,
)

from .temporal import latent_frames


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def unpack(rows, frames, height, width):
    return (
        rows.reshape(1, frames, height // 2, width // 2, 24, 1, 2, 2)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(1, 24, frames, height, width)
    )


class NativeRuntime:
    def __init__(self, model_path, device, steps=50):
        blocks = MiniMaxH3Ref2VACoreDenoiseStep().sub_blocks
        names = (
            "prepare_layout",
            "prepare_condition_latents",
            "prepare_latents",
            "prepare_latents_ref2va",
            "set_timesteps",
        )
        self.pipe = SequentialPipelineBlocks.from_blocks_dict(
            {name: blocks[name] for name in names}
        ).init_pipeline(model_path)
        self.pipe.load_components()
        self.device, self.steps = device, steps
        self.pipe.scheduler.set_timesteps(steps + 1, device=device)
        self.pipe.audio_scheduler.set_timesteps(steps + 1, device=device)

    def prepare(self, raw, metadata, seed, *, output_device=None):
        inputs = {key: value for key, value in raw.items() if key != "prompt_embeds"}
        state = self.pipe(
            **move(inputs, "cpu"),
            height=metadata["height"],
            width=metadata["width"],
            num_frames=metadata["frames"],
            num_inference_steps=self.steps + 1,
            generator=torch.Generator().manual_seed(seed),
        )
        keys = (
            "latents",
            "audio_latents",
            "token_tags",
            "position_ids",
            "video_indices",
            "audio_indices",
            "text_indices",
            "row_timestep_plan",
            "timesteps",
            "audio_timesteps",
            "num_condition_video_rows",
            "num_condition_audio_rows",
        )
        prepared = {key: state.get(key) for key in keys}
        prepared["prompt_embeds"] = raw["prompt_embeds"]
        return move(prepared, self.device if output_device is None else output_device)

    @staticmethod
    def predict(model, state, video, audio, index):
        timestep, inverse = state["row_timestep_plan"][index]
        return model(
            hidden_states=video[None],
            audio_hidden_states=audio[None],
            encoder_hidden_states=state["prompt_embeds"],
            timestep=timestep,
            timestep_indices=inverse,
            return_dict=False,
            **{
                key: state[key]
                for key in (
                    "token_tags",
                    "position_ids",
                    "video_indices",
                    "audio_indices",
                    "text_indices",
                )
            },
        )

    @torch.no_grad()
    def sample(self, model, prepared, metadata):
        self.pipe.scheduler.set_timesteps(self.steps + 1, device=self.device)
        self.pipe.audio_scheduler.set_timesteps(self.steps + 1, device=self.device)
        state = move(prepared, self.device)
        video, audio = state["latents"].clone(), state["audio_latents"].clone()
        nv, na = state["num_condition_video_rows"], state["num_condition_audio_rows"]
        for index, (tv, ta) in enumerate(zip(state["timesteps"], state["audio_timesteps"])):
            vp, ap = self.predict(model, state, video, audio, index)
            video[nv:] = self.pipe.scheduler.step(
                vp[0, nv:].float(), tv, video[nv:], return_dict=False
            )[0]
            audio[na:] = self.pipe.audio_scheduler.step(
                ap[0, na:].float(), ta, audio[na:], return_dict=False
            )[0]
        result = unpack(
            video[nv:],
            latent_frames(metadata["frames"]),
            metadata["height"] // 16,
            metadata["width"] // 16,
        )
        return result, audio
