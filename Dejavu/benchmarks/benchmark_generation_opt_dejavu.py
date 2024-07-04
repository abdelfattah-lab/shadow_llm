import os
import re
import argparse
import time

import torch
import pytest


from transformers import OPTConfig, AutoTokenizer

from src.models.gpt_dejavu import GPTLMHeadModel

from flash_attn.models.opt import opt_config_to_gpt2_config
from flash_attn.utils.generation import update_graph_cache
from src.utils.dejavu_util import remap_state_dict_opt, shard_state_dict_tp

MODEL_PATH = {
    "facebook/opt-30b": "/root/LSHOPT/opt-30b/",
    "facebook/opt-13b": "/root/LSHOPT/opt-13b/",
    "facebook/opt-6.7b": "/root/LSHOPT/opt-6.7b/",
    "facebook/opt-1.3b": "/root/LSHOPT/opt-1.3b/",
}

parser = argparse.ArgumentParser(description="OPT generation benchmarking")
parser.add_argument("--promptlen", type=int, default=128)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument(
    "--mlp-K",
    type=int,
    default=49152,
)
parser.add_argument(
    "--model-name",
    type=str,
    default="facebook/opt-30b",
    help="model-name",
)
parser.add_argument(
    "--att-K1",
    type=int,
    default=96,
)
parser.add_argument(
    "--att-K2",
    type=int,
    default=96,
)
parser.add_argument(
    "--skip-ffn-predictor",
    action="store_true",
)
parser.add_argument(
    "--skip-att-predictor",
    action="store_true",
)
parser.add_argument(
    "--replace-ffn-predictor",
    action="store_true",
)
parser.add_argument(
    "--replace-att-predictor",
    action="store_true",
)
args = parser.parse_args()


os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
torch.distributed.init_process_group(backend="nccl", init_method="env://")
device = f"cuda:{torch.distributed.get_rank()}"
world_size = torch.distributed.get_world_size()
# Need this, otherwise when we capture the graph the process for GPU 1 would run on both
# GPU0 and GPU1 and things would hang
torch.cuda.set_device(device)

repeats = 5
dtype = torch.float16
device = "cuda"
rtol, atol = 3e-3, 3e-1
fused_ft_kernel = True

if args.model_name == "opt-175b":
    print("Loading 175b config")
    config = OPTConfig.from_pretrained("facebook/opt-66b")
    config.hidden_size = 12 * 1024
    config.word_embed_proj_dim = config.hidden_size
    config.ffn_dim = 4 * config.hidden_size
    config.num_attention_heads = 96
    config.num_hidden_layers = 96
else:
    config = OPTConfig.from_pretrained(args.model_name)
config = opt_config_to_gpt2_config(config)
# Only prenorm supports residual_in_fp32
# config.residual_in_fp32 = getattr(config, 'prenorm', True)
config.use_flash_attn = True
config.fused_bias_fc = True
config.fused_mlp = True
# config.fused_dropout_add_ln = True
config.pad_vocab_size_multiple = 8 * world_size
config.sequence_parallel = False  # Need to set this to False for generation
config.mlp_sparse = False
config.att_sparse = False
""" 
sparse mlp config
"""
# import pdb; pdb.set_trace()
# print(args.mlp_K)
if args.mlp_K <= config.n_inner:
    config.mlp_sparse = True
    config.mlp_K = args.mlp_K
    config.mlp_sp_dim = 2048

"""
sparse att config
"""
if args.att_K1 <= config.n_head or args.att_K2 <= config.n_head:
    print("set sparse attention config")
    config.att_sparse = True
    config.att_K = [args.att_K1, args.att_K2]
    config.att_sp_dim = 2048

"""
Skip predictors configuration
"""
if args.replace_ffn_predictor:
    config.replace_ffn_predictor = True
else:
    config.replace_ffn_predictor = False

if args.skip_ffn_predictor:
    config.skip_ffn_predictor = True
else:
    config.skip_ffn_predictor = False

if args.replace_att_predictor:
    config.replace_att_predictor = True
else:
    config.replace_att_predictor = False

if args.skip_att_predictor:
    config.skip_att_predictor = True
else:
    config.skip_att_predictor = False

assert config.att_sparse and config.mlp_sparse == True

from apex.transformer import parallel_state

parallel_state.initialize_model_parallel(tensor_model_parallel_size_=world_size)
rank = parallel_state.get_tensor_model_parallel_rank()
process_group = parallel_state.get_tensor_model_parallel_group()

model = GPTLMHeadModel(config, device=device, dtype=dtype, process_group=process_group)

# # load model
# print(f"Loading {args.model_name} model")
# ckt = torch.load(
#     os.path.join(
#         MODEL_PATH[args.model_name],
#         f"full.pt",
#     ),
#     map_location="cpu",
# )
# ckt = remap_state_dict_opt(ckt, config)
# ckt = shard_state_dict_tp(ckt, config, world_size, rank)
# model.load_state_dict(ckt, strict=False)
# del ckt

model.eval()

print(
    f"rank {rank}, Num params {sum(p.numel() for p in model.parameters()) * world_size}"
)

torch.manual_seed(0)
# OPT tokenizer requires use_fast=False
# https://huggingface.co/docs/transformers/model_doc/opt
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b", use_fast=False)

input_ids = torch.randint(0, 100, (1, args.promptlen), dtype=torch.long, device="cuda")
# input_ids = tokenizer("Hello, my dog is cute and", return_tensors="pt").input_ids.to(
#     device=device
# )
max_length = input_ids.shape[1] + args.genlen

# print("Without CUDA graph")
# fn = lambda: model.generate(
#     input_ids=input_ids,
#     max_length=max_length,
#     fused_ft_kernel=True,
#     vocab_size=config.vocab_size,
#     return_dict_in_generate=True,
#     output_scores=True,
#     timing=False,
# )

# fn()
# torch.cuda.synchronize()
# torch.distributed.barrier()
# start = time.time()
# for _ in range(repeats):
#     out = model.generate(
#         input_ids=input_ids,
#         max_length=max_length,
#         fused_ft_kernel=True,
#         vocab_size=config.vocab_size,
#         return_dict_in_generate=True,
#         output_scores=True,
#         timing=False,
#     )
#     # with profile(activities=[ProfilerActivity.CUDA]) as prof:
#     # fn()
# torch.cuda.synchronize()
# # if rank == 0:
# #     prof.export_chrome_trace(f"opt-175b_generation_tp{world_size}.json")
# print(
#     f"Prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms"
# )
# if rank == 0:
#     print(tokenizer.batch_decode(out.sequences.tolist()))

# Capture graph outside the timing loop
batch_size, seqlen_og = input_ids.shape
# We need to pass tensor_parallel here, otherwise the kv_cache will have the wrong shape
model._decoding_cache = update_graph_cache(
    model, None, batch_size, seqlen_og, max_length, tensor_parallel=world_size
)
print("Warmup")
out_cg = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    fused_ft_kernel=True,
    cg=True,
    vocab_size=config.vocab_size,
    return_dict_in_generate=True,
    output_scores=True,
    timing=False,
)

torch.cuda.synchronize()
torch.distributed.barrier()
start = time.time()
for i in range(repeats):
    # import pdb; pdb.set_trace()
    out_cg = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        fused_ft_kernel=True,
        cg=True,
        vocab_size=config.vocab_size,
        return_dict_in_generate=True,
        output_scores=True,
        timing=False,
    )
torch.cuda.synchronize()
end = time.time()
print(
    f"Prompt processing + decoding time: {(end - start) / repeats * 1000:.0f}ms"
)
# Write time to file time.txt
with open("time.txt", "w") as f:
    f.write(f"{(end - start) / repeats * 1000:.0f}")

# print(tokenizer.batch_decode(out_cg.sequences.tolist()))
if rank == 0:
    print(tokenizer.batch_decode(out_cg.sequences.tolist()))

# If we don't delete the cache, it would hang and not exit. Maybe because the CUDA graph is still
# around and the NCCL connection is not closed?
del model._decoding_cache
