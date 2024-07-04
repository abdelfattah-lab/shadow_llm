# Copyright (c) 2022, Tri Dao.

import math
from functools import partial
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
except ImportError:
    flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func = None, None

try:
    from flash_attn.ops.flash_attn_triton import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
    )
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

try:
    from src.ops.fused_dense_sparse_dejavu import (
        FusedDense,
        ColumnParallelLinear,
        RowParallelLinear,
        RowParallelLinearNoReduce,
    )
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear, RowParallelLinearNoReduce = (
        None,
        None,
        None,
        None,
    )

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    import ft_attention
except ImportError:
    ft_attention = None

from flash_attn.utils.distributed import all_reduce


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self, causal=False, softmax_scale=None, attention_dropout=0.0, triton=False
    ):
        super().__init__()
        if attention_dropout != 0.0 or not triton:
            assert (
                flash_attn_unpadded_qkvpacked_func is not None
            ), "FlashAttention is not installed"
        if attention_dropout == 0.0 and triton:
            assert (
                flash_attn_qkvpacked_func is not None
            ), "FlashAttention Triton is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.triton = triton

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_unpadded_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        else:
            batch_size, seqlen = qkv.shape[0], qkv.shape[1]
            # Triton version doesn't support dropout
            if self.triton and (self.dropout_p == 0 or not self.training):
                output = flash_attn_qkvpacked_func(
                    qkv, None, causal, self.softmax_scale
                )
            else:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_seqlen = seqlen
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seqlen,
                    step=seqlen,
                    dtype=torch.int32,
                    device=qkv.device,
                )
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            return output


class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self, causal=False, softmax_scale=None, attention_dropout=0.0, triton=False
    ):
        super().__init__()
        if attention_dropout != 0.0 or not triton:
            assert (
                flash_attn_unpadded_kvpacked_func is not None
            ), "FlashAttention is not installed"
        if attention_dropout == 0.0 and triton:
            assert (
                flash_attn_kvpacked_func is not None
            ), "FlashAttention Triton is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.triton = triton

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_unpadded_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert (
                kv.shape[0] == batch_size
                and kv.shape[3] == q.shape[2]
                and kv.shape[4] == q.shape[3]
            )
            if self.triton and (
                self.dropout_p == 0.0 or not self.training
            ):  # Triton version doesn't support dropout
                output = flash_attn_kvpacked_func(
                    q, kv, None, causal, self.softmax_scale
                )
            else:
                q = rearrange(q, "b s ... -> (b s) ...")
                kv = rearrange(kv, "b s ... -> (b s) ...")
                cu_seqlens_q = torch.arange(
                    0,
                    (batch_size + 1) * seqlen_q,
                    step=seqlen_q,
                    dtype=torch.int32,
                    device=q.device,
                )
                cu_seqlens_k = torch.arange(
                    0,
                    (batch_size + 1) * seqlen_k,
                    step=seqlen_k,
                    dtype=torch.int32,
                    device=kv.device,
                )
                output = flash_attn_unpadded_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqlen_q,
                    seqlen_k,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            return output


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert (
            kv.shape[0] == batch_size
            and kv.shape[3] == q.shape[2]
            and kv.shape[4] == q.shape[3]
        )
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(
                torch.full((seqlen_q, seqlen_k), -10000.0, device=scores.device), 1
            )
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            # For FT, k_cache has shape (b, h, headdim / packsize, s, packsize)
            # where packsize = 4 if fp32, 8 if fp16 or bf16.
            # v_cache has shape (b, h, s, headdim)
            k_cache, v_cache = inference_params.key_value_memory_dict[layer_idx]
            kv_cache = None
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (
        kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0]
    )
    assert sequence_end <= (
        kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2]
    )
    # Copy key and values.
    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        kv = kv_cache[batch_start:batch_end, :sequence_end, ...]
        return kv
    else:
        assert inference_params.sequence_len_offset == 0
        # FT kernel requires different layouts for the k_cache and v_cache.
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        packsize = 4 if kv.dtype == torch.float32 else 8
        if kv_cache is not None:
            kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
            k_cache = rearrange(
                kv_cache[:, :, 0],
                "b s h (d packsize) -> b h d s packsize",
                packsize=packsize,
            ).contiguous()
            v_cache = rearrange(kv_cache[:, :, 1], "b s h d -> b h s d").contiguous()
            inference_params.key_value_memory_dict[layer_idx] = (k_cache, v_cache)
        else:
            k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
                kv[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize
            )
            v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(
                kv[:, :, 1], "b s h d -> b h s d"
            )
        return kv


def split_topk(input, k, sorted=False, num_chunks=2):
    N0, N1 = list(input.shape)
    NN0, NN1 = N0 * num_chunks, N1 // num_chunks
    part_input = input.reshape([NN0, NN1])
    part_val, part_idx = torch.topk(part_input, k=k, dim=1)
    key = (N0, N1)
    if not key in self.part_idx_offset:
        pio = (
            torch.tensor(
                [NN1 * i for i in range(num_chunks)] * N0,
                dtype=part_idx.dtype,
                pin_memory=True,
            )
            .to(device="cuda", non_blocking=True)
            .reshape([NN0, 1])
        )
        self.part_idx_offset[key] = pio
    else:
        pio = self.part_idx_offset[key]
    part_idx.add_(pio)
    part_val = part_val.reshape([N0, k * num_chunks])
    part_idx = part_idx.reshape([N0, k * num_chunks])
    top_val, idx2 = torch.topk(part_val, k=k, sorted=sorted, dim=1)
    top_idx = torch.gather(part_idx, 1, idx2)
    return top_val, top_idx


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        cross_attn=False,
        bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_scale_base=0,
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
        sp_kwargs=None,
    ) -> None:
        """
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        assert sp_kwargs == None, "sparse predictor not support in MHA"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert (
                not cross_attn
            ), "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual
            if not fused_bias_fc
            else partial(FusedDense, return_residual=True)
        )
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        if not self.cross_attn:
            if not self.return_residual:
                self.Wqkv = linear_cls(
                    embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
                )
            else:
                self.Wqkv = linear_resid_cls(
                    embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
                )
            if self.dwconv:
                self.dwconv_qkv = nn.Conv1d(
                    3 * embed_dim,
                    3 * embed_dim,
                    kernel_size=3,
                    padding=2,
                    groups=3 * embed_dim,
                )
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            if not self.return_residual:
                self.Wkv = linear_cls(
                    embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs
                )
            else:
                self.Wkv = linear_resid_cls(
                    embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs
                )
            if self.dwconv:
                self.dwconv_q = nn.Conv1d(
                    embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
                )
                self.dwconv_kv = nn.Conv1d(
                    2 * embed_dim,
                    2 * embed_dim,
                    kernel_size=3,
                    padding=2,
                    groups=2 * embed_dim,
                )
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert (
            self.layer_idx is not None
        ), "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        if not self.cross_attn:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            qkv = rearrange(
                qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
            )
            if inference_params is None:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv)
                if not self.checkpointing:
                    context = self.inner_attn(qkv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(
                        self.inner_attn, qkv, **kwargs
                    )
            else:
                if (
                    not inference_params.fused_ft_kernel
                ) or inference_params.sequence_len_offset == 0:
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(
                            qkv, seqlen_offset=inference_params.sequence_len_offset
                        )
                    q = qkv[:, :, 0]
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
                    # If we're processing the prompt, causal=None (use self.causal).
                    # If we're decoding, then causal=False.
                    causal = (
                        None if inference_params.sequence_len_offset == 0 else False
                    )
                    context = self.inner_cross_attn(q, kv, causal=causal)
                else:
                    assert inference_params.fused_ft_kernel
                    assert ft_attention is not None
                    context = ft_attention.single_query_attention(
                        *rearrange(qkv, "b 1 three h d -> b three h d").unbind(dim=1),
                        *inference_params.key_value_memory_dict[self.layer_idx],
                        inference_params.lengths_per_sample,
                        inference_params.sequence_len_offset,
                        self.rotary_emb_dim,
                    )
                    context = rearrange(context, "b h d -> b 1 h d")
        else:
            if not self.return_residual:
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                kv = self.Wkv(x_kv if x_kv is not None else x)
            else:
                if x_kv is not None:
                    kv, x_kv = self.Wkv(x_kv)
                else:
                    kv, x = self.Wkv(x)
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two h d) -> ... two h d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            if inference_params is None:
                if not self.checkpointing:
                    context = self.inner_cross_attn(q, kv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(
                        self.inner_cross_attn, q, kv, **kwargs
                    )
            else:
                kv = self._update_kv_cache(kv)
                context = self.inner_cross_attn(q, kv, causal=False)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)


class ParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_scale_base=0,
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
        sp_kwargs=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert sp_kwargs == None, "sparse predictor not support in ParallelMHA"
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            process_group,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.num_active_heads = self.num_heads // process_group.size()
        self.process_group = process_group

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        if (
            inference_params is not None
            and inference_params.sequence_len_offset > 0
            and self.num_active_heads
            != self.num_heads // self.Wqkv.process_group.size()
        ):
            active_dim = self.num_active_heads * self.head_dim
            # from flash_attn.utils.distributed import all_reduce
            # return all_reduce(x, self.out_proj.process_group)
            # return x
            qkv = F.linear(
                x, self.Wqkv.weight[: 3 * active_dim], self.Wqkv.bias[: 3 * active_dim]
            )
        else:
            qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(
                qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim
            )
        else:
            qkv = rearrange(
                qkv,
                "(b s) (three h d) -> b s three h d",
                s=seqlen,
                three=3,
                d=self.head_dim,
            )
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_attn, qkv, **kwargs
                )
        else:
            if (
                not inference_params.fused_ft_kernel
            ) or inference_params.sequence_len_offset == 0:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=inference_params.sequence_len_offset
                    )
                q = qkv[:, :, 0]
                assert (
                    self.layer_idx is not None
                ), "Generation requires layer_idx in the constructor"
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                # If we're processing the prompt, causal=None (use self.causal).
                # If we're decoding, then causal=False.
                causal = None if inference_params.sequence_len_offset == 0 else False
                context = self.inner_cross_attn(q, kv, causal=causal)
            else:
                assert inference_params.fused_ft_kernel
                assert ft_attention is not None
                k_cache, v_cache = inference_params.key_value_memory_dict[
                    self.layer_idx
                ]
                if (
                    self.num_active_heads
                    != self.num_heads // self.Wqkv.process_group.size()
                ):
                    k_cache = k_cache[:, : self.num_active_heads]
                    v_cache = v_cache[:, : self.num_active_heads]
                context = ft_attention.single_query_attention(
                    *rearrange(qkv, "b 1 three h d -> b three h d").unbind(dim=1),
                    # *inference_params.key_value_memory_dict[self.layer_idx],
                    k_cache,
                    v_cache,
                    inference_params.lengths_per_sample,
                    inference_params.sequence_len_offset,
                    self.rotary_emb_dim,
                )
                context = rearrange(context, "b h d -> b 1 h d")
        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")
        if (
            inference_params is not None
            and inference_params.sequence_len_offset > 0
            and self.num_active_heads
            != self.num_heads // self.Wqkv.process_group.size()
        ):
            active_dim = self.num_active_heads * self.head_dim
            out = F.linear(
                context, self.out_proj.weight[:, :active_dim], self.out_proj.bias
            )
            # Emma added: have to do all reduce after out projection
            return all_reduce(out, self.process_group)
        else:
            out = self.out_proj(context)
            return out


class ParallelMHADejavu(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        sp_kwargs=None,
        bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_scale_base=0,
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.num_head_per_node = self.num_heads // process_group.size()
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinearNoReduce(
            embed_dim,
            embed_dim,
            process_group,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.process_group = process_group

        self.sp_stream = torch.cuda.Stream(device="cuda", priority=0)
        self.event_out = torch.cuda.Event(enable_timing=False, blocking=False)
        self.event_mlp_sp = torch.cuda.Event(enable_timing=False, blocking=False)

    def forward(
        self, x, mlp_sp_logit=None, seqlen=None, inference_params=None, **kwargs
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
            mlp_sp_logit: (b, 4*hidden_dim), calculate topk neuron to activate for MLP
            head_idx: (b, k), k is the number of selected heads.
        """
        curr_stream = torch.cuda.current_stream()
        qkv = self.Wqkv(x)

        if seqlen is None:
            qkv = rearrange(
                qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim
            )
        else:
            qkv = rearrange(
                qkv,
                "(b s) (three h d) -> b s three h d",
                s=seqlen,
                three=3,
                d=self.head_dim,
            )
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_attn, qkv, **kwargs
                )
        else:
            if (
                not inference_params.fused_ft_kernel
            ) or inference_params.sequence_len_offset == 0:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=inference_params.sequence_len_offset
                    )
                q = qkv[:, :, 0]
                assert (
                    self.layer_idx is not None
                ), "Generation requires layer_idx in the constructor"
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                # If we're processing the prompt, causal=None (use self.causal).
                # If we're decoding, then causal=False.
                causal = None if inference_params.sequence_len_offset == 0 else False
                context = self.inner_cross_attn(q, kv, causal=causal)
            else:
                assert inference_params.fused_ft_kernel
                assert ft_attention is not None
                k_cache, v_cache = inference_params.key_value_memory_dict[
                    self.layer_idx
                ]
                context = ft_attention.single_query_attention(
                    *rearrange(qkv, "b 1 three h d -> b three h d").unbind(dim=1),
                    # *inference_params.key_value_memory_dict[self.layer_idx],
                    k_cache,
                    v_cache,
                    inference_params.lengths_per_sample,
                    inference_params.sequence_len_offset,
                    self.rotary_emb_dim,
                )
                context = rearrange(context, "b h d -> b 1 h d")
        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")

        out = self.out_proj(context)
        curr_stream.record_event(self.event_out)

        out = all_reduce(out, self.process_group)

        mlp_idx = None
        with torch.cuda.stream(self.sp_stream):
            self.sp_stream.wait_event(self.event_out)
            if mlp_sp_logit != None:
                print(mlp_sp_logit.size())
                _, mlp_idx = mlp_sp_logit.topk(self.mlp_k, sorted=False)

            self.sp_stream.record_event(self.event_mlp_sp)

        curr_stream.wait_event(self.event_mlp_sp)

        return out, mlp_idx

class ParallelSP(nn.Module):
    """
    A Near Neighbor Classifier
    """

    def __init__(
        self,
        layer_idx=None,
        embed_dim=None,
        low_rank_dim=None,
        out_dim=None,
        K=None,
        process_group=None,
        sequence_parallel=False,
        device=None,
        dtype=None,
        replace_att_predictor=False,
        skip_att_predictor=False,
        num_hidden_layers=1,
    ) -> None:
        super().__init__()
        assert (
            process_group is not None
        ), "sparse predictor only implemented with parallel for now"

        factory_kwargs = {"device": device, "dtype": dtype}
        self.process_group = process_group
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.fc0 = nn.Linear(
            embed_dim, low_rank_dim, bias=False, device=device, dtype=dtype
        )
        self.out = out_dim // self.process_group.size()
        if replace_att_predictor and layer_idx == 0:
            out_dim = out_dim * (num_hidden_layers - 1)
        self.fc1 = ColumnParallelLinear(
            low_rank_dim,
            out_dim,
            process_group,
            bias=False,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.K = K // self.process_group.size()
        self.replace_att_predictor = replace_att_predictor
        self.skip_att_predictor = skip_att_predictor

    def forward(self, x):
        if self.skip_att_predictor:
            return torch.squeeze(x)[:self.out]
        if self.replace_att_predictor:
            if self.layer_idx == 0:
                x = self.fc0(x.view(self.embed_dim))
                x = self.fc1(x)
                return x
            else:
                return torch.squeeze(x)[:self.out]
        else:
            x = F.relu(self.fc0(x.view(self.embed_dim)))  # b x 1000
            x = self.fc1(x)
            return x


class ParallelTracker(nn.Module):
    def __init__(self, process_group, num_head_per_node, seq_len, device=None) -> None:
        self.process_group = process_group
        self.num_head_per_node = num_head_per_node
        self.tracker = torch.arange(
            0, seq_len, dtype=torch.int32, device=device
        ).repeat(self.num_head_per_node, 1)
        super().__init__()

    def get_batch_idx(self, head_idx, seq_idx):
        self.tracker = self.tracker.to(head_idx.device)
        return self.tracker[head_idx][:, :seq_idx]

    def update(self, head_idx, seq_idx, compute_idx):
        # mark all compute position as -1

        temp = self.tracker[:, : seq_idx + 1][head_idx]
        temp[compute_idx != -1] = -1
        self.tracker[:, : seq_idx + 1][head_idx] = temp

        # self.tracker[:, seq_idx][head_idx] = -1

    def reset(self, device):
        self.tracker = torch.arange(0, 16, dtype=torch.int32, device=device).repeat(
            self.num_head_per_node, 1
        )


class ParallelMHASparseAttMlp(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        sp_kwargs,
        bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_scale_base=0,
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert sp_kwargs != None, "sparse predictor parameters are not passed in."
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.sp_kwargs = sp_kwargs
        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.num_head_per_node = self.num_heads // process_group.size()
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=bias,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinearNoReduce(
            embed_dim,
            embed_dim,
            process_group,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.process_group = process_group

        """sparse related"""

        # sparse predictor related
        (
            self.head_idx_qkv_idx_mapping,
            self.head_idx_out_idx_mapping,
        ) = self.generate_idx_mapping(device)

        self.out_proj_weight_t = self.register_buffer("out_proj_weight_t", None)

        self.skip_att_predictor = sp_kwargs["skip_att_predictor"]
        self.replace_att_predictor = sp_kwargs["replace_att_predictor"]
        self.need_predictor = not(self.skip_att_predictor) and (layer_idx == 0 or not(self.replace_att_predictor))

        if self.need_predictor:
            self.sp = ParallelSP(
                layer_idx=layer_idx,
                device=device,
                dtype=dtype,
                process_group=process_group,
                sequence_parallel=sequence_parallel,
                **sp_kwargs,
            )
        else:
            self.att_k = sp_kwargs["K"] // process_group.size()
        # cache to store previous x for up to past 16 tokens to recompute KV for skiped head
        self.x_cache = torch.empty((16, embed_dim), dtype=torch.float16, device=device)

        self.to_compute_token_idx = ParallelTracker(
            process_group=process_group,
            num_head_per_node=self.num_head_per_node,
            seq_len=16,
        )
        # generation counter used to clear x_cache
        self.counter = 0
        # generation counter used to copy kv cache
        self.cache_offset = None

        self.sp_stream = torch.cuda.Stream(device="cuda", priority=0)
        self.event_qkv = torch.cuda.Event(enable_timing=False, blocking=False)
        self.event_out = torch.cuda.Event(enable_timing=False, blocking=False)
        self.event_mlp_sp = torch.cuda.Event(enable_timing=False, blocking=False)
        self.event_att_sp = torch.cuda.Event(enable_timing=False, blocking=False)

    def generate_idx_mapping(self, device):
        # assert == 128 * (96/8) * 3, 12288
        q_idx = torch.arange(
            0,
            self.embed_dim // torch.distributed.get_world_size(),
            dtype=torch.long,
            device=device,
        ).reshape(-1, self.head_dim)
        k_idx = q_idx.clone() + 1 * (
            self.embed_dim // torch.distributed.get_world_size()
        )

        v_idx = q_idx.clone() + 2 * (
            self.embed_dim // torch.distributed.get_world_size()
        )

        out_idx = q_idx.clone()

        qkv_idx = torch.cat((q_idx, k_idx, v_idx), dim=1)

        return qkv_idx, out_idx

    def forward(
        self,
        x,
        residual,
        att_predictions=None,
        head_idx=None,
        mlp_sp_logit=None,
        seqlen=None,
        inference_params=None,
        **kwargs
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
            mlp_sp_logit: (b, 4*hidden_dim), calculate topk neuron to activate for MLP
            head_idx: (b, k), k is the number of selected heads.
        """
        do_token_generation = x.size(1) == 1

        # prompting sequence length
        if x.size(1) != 1:
            self.cache_offset = torch.tensor([x.size(1)], dtype=torch.int32).to(
                x.device
            )

        curr_stream = torch.cuda.current_stream()

        if (
            inference_params is not None
            and inference_params.sequence_len_offset > 0
            and head_idx != None
        ):
            assert x.size(1) == 1
            self.x_cache[self.counter] = x
            # get the token index to compute
            if self.counter == 15:
                # computer key value for past 16 tokens to clear x cache
                self.to_compute_token_idx.reset(device=x.device)
                head_idx = torch.arange(
                    0, self.num_head_per_node, dtype=torch.int32, device=x.device
                )

            head_recompute_token_idx = self.to_compute_token_idx.get_batch_idx(
                head_idx, self.counter + 1
            )

            from src.ops.triton.attention_proj_sparse import qkv_proj_sparse

            qkv = qkv_proj_sparse(
                self.x_cache[: self.counter + 1],
                rearrange(
                    self.Wqkv.weight,
                    "(three n m) d -> three n m d",
                    three=3,
                    n=self.num_head_per_node,
                    m=self.head_dim,
                    d=self.embed_dim,
                ),
                head_idx,
                head_recompute_token_idx,
                rearrange(
                    self.Wqkv.bias,
                    "(three n m) -> three n m",
                    three=3,
                    n=self.num_head_per_node,
                    m=self.head_dim,
                ),
            )

            # update trackers
            if self.counter != 15:
                self.to_compute_token_idx.update(
                    head_idx, self.counter, head_recompute_token_idx
                )
            self.counter += 1
        else:
            qkv = self.Wqkv(x)
        curr_stream.record_event(self.event_qkv)

        # mlp sp topk
        mlp_idx = None
        with torch.cuda.stream(self.sp_stream):
            self.sp_stream.wait_event(self.event_qkv)
            if mlp_sp_logit != None:
                if self.layer_idx == self.sp_kwargs["num_hidden_layers"] - 1:
                    mlp_idx = None
                else:
                    _, mlp_idx = mlp_sp_logit[:self.Wqkv.in_features * 4 // self.process_group.size()].topk(self.mlp_k, sorted=False)
            self.sp_stream.record_event(self.event_mlp_sp)


        if seqlen is None and head_idx is None:
            qkv = rearrange(
                qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim
            )
        elif seqlen is not None:
            qkv = rearrange(
                qkv,
                "(b s) (three h d) -> b s three h d",
                s=seqlen,
                three=3,
                d=self.head_dim,
            )

        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_attn, qkv, **kwargs
                )
        else:
            if (
                not inference_params.fused_ft_kernel
            ) or inference_params.sequence_len_offset == 0:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=inference_params.sequence_len_offset
                    )
                q = qkv[:, :, 0]
                assert (
                    self.layer_idx is not None
                ), "Generation requires layer_idx in the constructor"
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                # If we're processing the prompt, causal=None (use self.causal).
                # If we're decoding, then causal=False.
                causal = None if inference_params.sequence_len_offset == 0 else False
                context = self.inner_cross_attn(q, kv, causal=causal)
            else:
                assert inference_params.fused_ft_kernel
                assert ft_attention is not None
                k_cache, v_cache = inference_params.key_value_memory_dict[
                    self.layer_idx
                ]

                from src.ops.triton.attention_proj_sparse import k_cache_copy_sparse
                from src.ops.triton.attention_proj_sparse import v_cache_copy_sparse

                # copy calculate KV to kvcahce
                if head_idx != None:
                    assert x.size(1) == 1
                    # qkv:  b, 3, h d
                    k_cache_copy_sparse(
                        qkv[:, 1],
                        k_cache,
                        head_idx,
                        head_recompute_token_idx,
                        self.cache_offset,
                    )
                    v_cache_copy_sparse(
                        qkv[:, 2],
                        v_cache,
                        head_idx,
                        head_recompute_token_idx,
                        self.cache_offset,
                    )
                    context = ft_attention.single_query_attention(
                        *qkv[-1:, :, :].unbind(dim=1),
                        k_cache,
                        v_cache,
                        inference_params.lengths_per_sample,
                        inference_params.sequence_len_offset,
                        self.rotary_emb_dim,
                    )
                else:
                    context = ft_attention.single_query_attention(
                        *rearrange(qkv, "b 1 three h d -> b three h d").unbind(dim=1),
                        # *inference_params.key_value_memory_dict[self.layer_idx],
                        k_cache,
                        v_cache,
                        inference_params.lengths_per_sample,
                        inference_params.sequence_len_offset,
                        self.rotary_emb_dim,
                    )
                    context = rearrange(context, "b h d -> b 1 h d")
        if seqlen is None:
            if head_idx == None:
                context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")

        if (
            inference_params is not None
            and inference_params.sequence_len_offset > 0
            and head_idx != None
        ):
            assert context.size(0) == 1
            from src.ops.triton.attention_proj_sparse import out_proj_sparse

            out = out_proj_sparse(
                context,
                rearrange(
                    self.out_proj.weight,
                    "d (n m) -> d n m",
                    n=self.num_head_per_node,
                    m=self.head_dim,
                ),
                head_idx,
                self.out_proj.bias,
            )

            # clear token cache and tracker
            if self.counter == 16:
                self.counter = 0
                self.x_cache = torch.empty(
                    (16, self.embed_dim),
                    device=x.device,
                    dtype=torch.float16,
                )
                self.cache_offset += 16
        else:
            out = self.out_proj(context)

        curr_stream.record_event(self.event_out)

        out = all_reduce(out, self.process_group)
        att_idx = None
        att_sp_logit = None
        if self.need_predictor:
            with torch.cuda.stream(self.sp_stream):
                self.sp_stream.wait_event(self.event_out)
                if do_token_generation:
                    att_sp_logit = self.sp(residual)
                    _, att_idx = att_sp_logit[:(self.num_heads // self.Wqkv.process_group.size())].topk(self.att_k, sorted=False)
                    att_idx = att_idx.to(torch.int32)
                self.sp_stream.record_event(self.event_att_sp)
        else:
            if do_token_generation:
                if self.skip_att_predictor:
                    if self.layer_idx == self.sp_kwargs["num_hidden_layers"] - 1:
                        att_idx = torch.tensor([], device=x.device, dtype=torch.int32)
                    else:
                        att_idx = torch.randint(0, self.num_heads // self.Wqkv.process_group.size(), (self.att_k,), device=x.device, dtype=torch.int32)
                # Last layer doesn't need to predict anything.
                elif self.layer_idx == self.sp_kwargs["num_hidden_layers"] - 1:
                    att_idx = torch.tensor([], device=x.device, dtype=torch.int32)
                else:
                    _, att_idx = att_predictions[:(self.num_heads // self.Wqkv.process_group.size())].topk(self.att_k, sorted=False)
                    att_idx = att_idx.to(torch.int32)

        curr_stream.wait_event(self.event_mlp_sp)
        curr_stream.wait_event(self.event_att_sp)
        # Make sure we're not returning predictions for any other mode.
        if not(self.replace_att_predictor):
            att_sp_logit = None
        return out, mlp_idx, att_idx, att_sp_logit
