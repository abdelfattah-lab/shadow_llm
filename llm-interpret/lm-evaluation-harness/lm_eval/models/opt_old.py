import transformers
import torch
import os
import random
import pickle
from lm_eval.base import BaseLM
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed

class SequenceModel(nn.Module):
    def __init__(self, embedding_dim=1024, output_dim=16):
        super(SequenceModel, self).__init__()
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        
        # 2-layer MLP
        self.fc1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.fc2 = nn.Linear(embedding_dim//2, output_dim)
    
    def generate_attention_mask(self, input_seq):
        # Assuming padding value is 0, create a mask for non-padded values
        # input_seq shape is assumed to be [B, L, E]
        mask = input_seq.ne(0).any(dim=2)  # Resulting shape [B, L]
        # Inverting the mask for attention: True for positions to attend, False for positions to ignore
        mask = ~mask  # Now mask is True where values should be ignored
        # No need to multiply by -1e9 here; we'll use this boolean mask directly
        return mask

    def forward(self, x):
        # Self-attention: [L, E] -> [L, L, E]
        attention_mask = self.generate_attention_mask(x)
        
        # Apply self-attention
        x, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)

        # Aggregating to [1, E]
        x = x.mean(dim=1, keepdim=True)
        
        # Passing through 2-layer MLP
        x = F.relu(self.fc1(x.squeeze(1)))
        x = self.fc2(x)
        
        return x
        
class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        zcp_calc=None,
        pretrained="facebook/opt-125m",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        model_cache_dir = None,
        tokenizer_cache_dir = None,
        mask_single_head=0,
        mask_heads=0,
        mask_fc=0,
        mask_iterative_fc=0,
        head_percent_mask=0,
        head_importance_path=None,
        fc_percent_mask=0,
        fc_importance_path=None,
        maskmethod="original",
        predictor_="piqa"
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        device_map = 'auto'
        if '66b' in pretrained:
            device_map = eval(open('/home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/lm_eval/models/66b_device_map.txt', 'r').readlines()[0])
            for key in device_map:
                if 'layers' in key:
                    layer_num = int(key.split('.')[-1])
                    if layer_num <= 3:
                        device_map[key] = 0
                    elif layer_num >= 60:
                        device_map[key] = 'cpu'
                    else:    
                        device_map[key] = ((layer_num - 4) // 8) + 1

        self.opt = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map = device_map,
            cache_dir = model_cache_dir,
            torch_dtype=torch.float16
        )

        self.opt.get_decoder().embed_tokens.weight.requires_grad = False
        self.opt.get_decoder().embed_positions.weight.requires_grad = False
        # self.opt = transformers.AutoModelForCausalLM.from_pretrained(
        #     pretrained,
        #     cache_dir = model_cache_dir,
        # )
        # self.ds_engine = deepspeed.initialize(model=self.opt, model_parameters = self.opt.parameters(), config_params=ds_config)[0]
        # self.ds_engine.module.eval()  # inference

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            cache_dir = tokenizer_cache_dir if tokenizer_cache_dir else 'tokenizer_cache/',
            use_fast = False
        ) if tokenizer is None else tokenizer

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        num_hidden_layers = self.opt.config.num_hidden_layers
        num_heads = self.opt.config.num_attention_heads
        self.head_mask = torch.ones(num_hidden_layers * num_heads, dtype = torch.half)
        self.fc_mask = torch.ones(num_hidden_layers, dtype = torch.half)
        self.head_percent_mask = head_percent_mask
        if int(mask_heads):
            with open(head_importance_path, 'rb') as f:
                importance = pickle.load(f)
            if "oracle" in head_importance_path:
                self.head_mask = np.asarray(importance.tolist())
            else:
                _, head_indices = torch.sort(importance.view(-1))
                head_indices = list(head_indices.numpy())
                head_indices = head_indices[: int(head_percent_mask) * len(head_indices) // 100]
                self.head_mask[head_indices] = 0. 
        elif int(mask_single_head): #Only performing it on OPT125M
            self.head_mask[int(mask_single_head)-1] = 0.
        self.head_mask = torch.Tensor(self.head_mask).view(num_hidden_layers, num_heads).contiguous()
        # Change its type to half
        self.head_mask = self.head_mask.half()
        
        if mask_fc:
            self.fc_mask[int(mask_fc)] = 0.
        elif int(mask_iterative_fc):
            with open(fc_importance_path, 'rb') as f:
                fc_indices = list(pickle.load(f))
            fc_indices = fc_indices[: int(fc_percent_mask) * len(fc_indices) // 100] 
            self.fc_mask[fc_indices] = 0.
        self.maskmethod = maskmethod
        # if self.maskmethod == "shadowllm":
        self.headpredictor_dict = {}
        if self.maskmethod != "original":
        # load all predictors from model_opt1.3b for each layer as model_layer_{layer_idx}.pt
            for layer in range(num_hidden_layers):
                predm_ = SequenceModel(embedding_dim=2048 , output_dim=num_heads)
                predm_.load_state_dict(torch.load(f'./model_opt1.3b_{predictor_}/model_layer_{layer}.pt'))
                predm_ = predm_.to(self._device)
                self.headpredictor_dict[layer] = predm_
        self.tokenencoder = self.opt.model.decoder.embed_tokens
        self._temp_tracker = 0


    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.opt.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def get_tokenizer(self):
        return self.tokenizer

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, attn_mask = None, labels = None):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """

        if labels == None:
            # we take inps; feed it to our predictor for each layer, modify head_mask to be a per-item dict
            # modify opt forward pass to use the headmask corresponding to the item
            with torch.no_grad():
                if self.maskmethod == "original":
                    self.newmask = self.head_mask
                    # # import pdb; pdb.set_trace()
                    # curr_dindex = self._temp_tracker // 2
                    # # Use the curr_dindex to index the self.headact_trace to get per-token head-layer mask
                    # self.newmask = self.headact_trace[self._temp_tracker][1]
                    # # normalize masks
                    # _, head_indices = torch.sort(self.newmask.view(-1))
                    # head_indices = list(head_indices.numpy())
                    # head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                    # self.newmask = torch.ones_like(self.newmask).view(-1)
                    # self.newmask[head_indices] = 0.
                    # self.newmask = self.newmask.view(self.head_mask.shape[0], self.head_mask.shape[1]).contiguous().half()
                elif self.maskmethod == "predictor":
                    self.newmask = torch.zeros_like(self.head_mask)
                    enc_inp = self.tokenencoder(inps)
                    for layer_ in range(self.head_mask.shape[0]):
                        self.newmask[layer_] = self.headpredictor_dict[layer_](enc_inp.float()).squeeze()
                    # normalize masks
                    _, head_indices = torch.sort(self.newmask.view(-1))
                    head_indices = list(head_indices.numpy())
                    head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                    self.newmask = torch.ones_like(self.newmask).view(-1)
                    self.newmask[head_indices] = 0.
                    self.newmask = self.newmask.view(self.head_mask.shape[0], self.head_mask.shape[1]).contiguous()
                elif self.maskmethod == "kmeans":
                    raise NotImplementedError
                else:
                    raise ValueError("Invalid maskmethod")
                # import pdb; pdb.set_trace()
                self._temp_tracker += 1
                # import pdb; pdb.set_trace()
                return self.opt(input_ids = inps, head_mask = self.newmask, fc_mask = self.fc_mask)[0][:, :, :50265]
                # rank = int(os.getenv("LOCAL_RANK", "0"))
                # return self.ds_engine.module(input_ids = inps, head_mask = self.head_mask.to(rank))[0][:, :, :50265]
        else:
            return self.opt(input_ids = inps, attention_mask = attn_mask, labels = labels).loss
            # return self.ds_engine.module(input_ids = inps, attention_mask = attn_mask, labels = labels).loss
            
    def _model_generate(self, context, max_length, eos_token_id):
        return self.opt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
OPTLM = HFLM