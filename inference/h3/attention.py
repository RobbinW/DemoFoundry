"""SDPA for the native H3 attention layout, optionally sharded across ranks."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distributed_nn
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb


class Processor:
    def __init__(self, parallel=None):
        self.parallel = parallel

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("This processor serves unmasked native H3 packed sequences")
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            q, k = (_apply_rotary_emb(x, *rotary_emb) for x in (q, k))
        if self.parallel is not None:
            q, k, v = self.parallel.to_heads(torch.cat((q, k, v), dim=-1)).chunk(3, dim=-1)
        output = torch.nn.functional.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in (q, k, v))
        )
        output = output.transpose(1, 2)
        if self.parallel is not None:
            output = self.parallel.to_sequence(output)
        return attn.to_out[1](attn.to_out[0](output.flatten(2)))


class SequenceParallel:
    """Uneven Ulysses exchange: sequence-sharded FFNs, head-sharded attention."""

    def __init__(self, model, group=None):
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.length = 0
        block_attention = {block.attn for block in model.transformer_blocks}
        for module in model.modules():
            if module in block_attention:
                module.set_processor(Processor(self if self.size > 1 else None))
            elif type(module).__name__ == "MiniMaxH3Attention":
                module.set_processor(Processor())
        if self.size > 1:
            model.rope.register_forward_hook(self.rope)
            model.transformer_blocks[0].register_forward_pre_hook(self.first_block)
            for block in model.transformer_blocks:
                block.register_forward_pre_hook(self.block)
            model.norm_out.register_forward_pre_hook(self.norm)
            model.proj_out.register_forward_hook(self.output)
            model.audio_proj_out.register_forward_hook(self.output)

    def split(self, tensor, dim=0):
        return tensor.tensor_split(self.size, dim=dim)[self.rank].contiguous()

    def lengths(self):
        return [
            self.length // self.size + (rank < self.length % self.size) for rank in range(self.size)
        ]

    def to_heads(self, tensor):
        batch, local, heads, channels = tensor.shape
        if heads % self.size:
            raise ValueError("H3 attention heads must divide the sequence group")
        part = heads // self.size
        send = (
            tensor.reshape(batch, local, self.size, part, channels)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
            .flatten()
        )
        lengths = self.lengths()
        receive_counts = [batch * length * part * channels for length in lengths]
        received = distributed_nn.all_to_all_single(
            send.new_empty(sum(receive_counts)),
            send,
            output_split_sizes=receive_counts,
            input_split_sizes=[batch * local * part * channels] * self.size,
            group=self.group,
        )
        return torch.cat(
            [
                piece.reshape(batch, length, part, channels)
                for piece, length in zip(received.split(receive_counts), lengths)
            ],
            dim=1,
        )

    def to_sequence(self, tensor):
        batch, _, heads, channels = tensor.shape
        lengths = self.lengths()
        local = lengths[self.rank]
        send_counts = [batch * length * heads * channels for length in lengths]
        send = torch.cat(
            [piece.contiguous().flatten() for piece in tensor.split(lengths, dim=1)]
        )
        received = distributed_nn.all_to_all_single(
            send.new_empty(batch * local * heads * channels * self.size),
            send,
            output_split_sizes=[batch * local * heads * channels] * self.size,
            input_split_sizes=send_counts,
            group=self.group,
        )
        return (
            received.reshape(self.size, batch, local, heads, channels)
            .permute(1, 2, 0, 3, 4)
            .reshape(batch, local, heads * self.size, channels)
        )

    def gather(self, tensor, dim):
        lengths = self.lengths()
        shape = list(tensor.shape)
        shape[dim] = max(lengths) - tensor.shape[dim]
        padded = torch.cat((tensor, tensor.new_zeros(shape)), dim=dim)
        parts = distributed_nn.all_gather(padded.contiguous(), group=self.group)
        return torch.cat(
            [part.narrow(dim, 0, length) for part, length in zip(parts, lengths)], dim=dim
        )

    def rope(self, module, args, output):
        self.length = output[0].shape[0]
        return tuple(self.split(value) for value in output)

    def first_block(self, module, args):
        return (self.split(args[0], 1), *args[1:])

    def block(self, module, args):
        hidden, temb, indices, rotary = args
        return hidden, temb, self.split(indices), rotary

    def norm(self, module, args):
        hidden, temb, indices = args
        return hidden, temb, self.split(indices)

    def output(self, module, args, output):
        return self.gather(output, 1)
