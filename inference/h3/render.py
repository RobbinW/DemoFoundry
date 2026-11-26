"""Render one camera view with the native MiniMax-H3 Ref2VA workflow."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .media import load_proxy, write_video
from .temporal import pad_tail, rollout_windows


def render_windows(proxy, reference, seed, sample):
    for index, window in enumerate(rollout_windows(len(proxy))):
        clip = pad_tail(proxy[window.start : window.start + window.valid], window.frames)
        frames = sample(reference, clip, seed + index)
        if len(frames) < window.valid:
            raise ValueError("Decoded video is shorter than the requested window.")
        reference = frames[window.valid - 1]
        yield from frames[int(index > 0) : window.valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MiniMax-H3 model directory")
    parser.add_argument("--adapter", required=True, help="DemoFoundry H3 task LoRA (.safetensors)")
    parser.add_argument("--proxy", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", nargs=2, required=True, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(value <= 0 or value % 32 for value in args.size):
        parser.error("width and height must be positive multiples of 32")

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor, distribute_tensor

    from .encode import Decoder, Encoder
    from .model import adapter_parameters, load_model
    from .runtime import NativeRuntime

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    group = shard_group = None
    if world > 1:
        dist.init_process_group("nccl")
        group, shard_group = dist.new_group(), dist.new_group()
    leader = local_rank == 0
    width, height = args.size
    metadata = dict(width=width, height=height, frames=158)
    prompt = Path(__file__).with_name("prompt.txt").read_text().strip()

    with torch.no_grad():
        model = load_model(
            args.model, device, group=group, shard=world > 1, shard_group=shard_group
        ).eval()
        weights = load_file(args.adapter)
        params = adapter_parameters(model)
        if weights.keys() != params.keys():
            raise ValueError("LoRA keys do not match the native Ref2VA attention adapter.")
        for name, parameter in params.items():
            value = weights[name].to(device)
            if isinstance(parameter, DTensor):
                value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
            parameter.copy_(value)
        del weights
        runtime = NativeRuntime(args.model, device)
        if leader:
            devices = [f"cuda:{i}" for i in range(world)]
            encoder = Encoder(args.model, devices[0], devices)
            decoder = Decoder(args.model, device)
            proxy = load_proxy(args.proxy, width, height)
            reference = Image.open(args.reference).convert("RGB")
        length = [len(proxy) if leader else None]
        if world > 1:
            dist.broadcast_object_list(length, src=0, group=group)

        def denoise(prepared):
            packet = [prepared]
            if world > 1:
                dist.broadcast_object_list(packet, src=0, group=group)
            latent, _ = runtime.sample(model, packet[0], metadata)
            return latent

        if leader:

            def sample(ref, clip, seed):
                conditions = encoder.conditions(ref, clip, metadata, prompt)
                state = runtime.prepare(
                    conditions, metadata, seed, output_device="cpu" if world > 1 else device
                )
                return decoder(denoise(state))

            args.output.parent.mkdir(parents=True, exist_ok=True)
            frames = render_windows(proxy, reference, args.seed, sample)
            with TemporaryDirectory(dir=args.output.parent) as staging:
                rendered = Path(staging) / args.output.name
                write_video(frames, rendered, (width, height))
                rendered.replace(args.output)
        else:
            for _ in rollout_windows(length[0]):
                denoise(None)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
