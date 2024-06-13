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

glob_n = 2048
glob_n_h = 2048
class B1EPredModelFFN(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1EPredModelFFN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n)
        self.fc2 = nn.Linear(glob_n, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class B1ESeqPredModelFFN(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1ESeqPredModelFFN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n)
        self.fc2 = nn.Linear(glob_n, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x


class B1EPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1EPredModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n_h)
        self.fc2 = nn.Linear(glob_n_h, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x


class B1ESeqPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1ESeqPredModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n_h)
        self.fc2 = nn.Linear(glob_n_h, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x

        
class HFLM(BaseLM):
    def __init__(
        self,
        aggr_all=False,
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
        ffn_percent_mask=0,
        fc_importance_path=None,
        maskmethod="original",
        fcmaskmethod="fc",
        predictor_="piqa",
        prune_style="perlayer",
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
        self.maskmethod = maskmethod
        self.head_percent_mask = int(head_percent_mask)
        self.ffn_percent_mask = int(ffn_percent_mask)
        if int(mask_heads):
            if self.maskmethod != "original":
                importance = torch.randn(num_hidden_layers, num_heads)
            else:
                with open(head_importance_path, 'rb') as f:
                    importance = pickle.load(f)
            if "oracle" in head_importance_path:
                self.head_mask = np.asarray(importance.tolist())
            else:
                try:
                    _, head_indices = torch.sort(importance.view(-1))
                except:
                    _, head_indices = torch.sort(torch.Tensor(importance).view(-1))

                head_indices = list(head_indices.numpy())
                head_indices = head_indices[: int(head_percent_mask) * len(head_indices) // 100]
                self.head_mask[head_indices] = 0. 
        elif int(mask_single_head): #Only performing it on OPT125M
            self.head_mask[int(mask_single_head)-1] = 0.
        self.head_mask = self.head_mask.view(num_hidden_layers, num_heads).contiguous()

        # Change its type to half
        self.head_mask = self.head_mask.half()
        # Create a self.fc_mask like head_mask, but with shape (num_hidden_layers, self.opt.config.ffn_dim)
        self.ffn_fc_mask = torch.ones(num_hidden_layers, self.opt.config.ffn_dim, dtype = torch.half)
        self.fcmaskmethod = fcmaskmethod
        self.prune_style = prune_style
        self.num_min_heads = 1          # Cannot enforce more than 93.75% sparsity if its   2, 96.87% if its   1
        self.num_min_ffn_neurons = 256  # Cannot enforce more than 93.75% sparsity if its 512, 96.87% if its 256
        if mask_fc:
            self.fc_mask[int(mask_fc)] = 0.
        elif int(mask_iterative_fc):
            with open(fc_importance_path, 'rb') as f:
                fc_indices = list(pickle.load(f))
            fc_indices = fc_indices[: int(fc_percent_mask) * len(fc_indices) // 100] 
            self.fc_mask[fc_indices] = 0.
        # if self.maskmethod == "shadowllm":
        model_size = pretrained.split('-')[-1]
        self.headpredictor_dict = {}
        self.fcpredictor_dict = {}
        if self.maskmethod == "predictor":
            self.headpred_model = B1EPredModel(embedding_dim=self.opt.config.ffn_dim//4, output_dim=num_hidden_layers*num_heads)
            model_path = f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1e.pt'
            self.headpred_model.load_state_dict(torch.load(model_path))
            self.headpred_model = nn.DataParallel(self.headpred_model)  # DataParallel wrapping
            # cuda it
            self.headpred_model = self.headpred_model.to(self._device)
            if self.fcmaskmethod == "fc":
                self.fc_model = B1EPredModelFFN(embedding_dim=self.opt.config.ffn_dim//4, output_dim=self.opt.config.ffn_dim*num_hidden_layers)
                model_size = pretrained.split('-')[-1]
                model_path = f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1e_fc1.pt'
                self.fc_model.load_state_dict(torch.load(model_path))
                self.fc_model = nn.DataParallel(self.fc_model)  # DataParallel wrapping
                # cuda it
                self.fc_model = self.fc_model.to(self._device)
        if self.maskmethod == "predictorL":
            self.headpred_model = B1EPredModel(embedding_dim=self.opt.config.ffn_dim//4, output_dim=num_hidden_layers*num_heads)
            model_path = f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1eL.pt'
            self.headpred_model.load_state_dict(torch.load(model_path))
            self.headpred_model = nn.DataParallel(self.headpred_model)  # DataParallel wrapping
            # cuda it
            self.headpred_model = self.headpred_model.to(self._device)
            if self.fcmaskmethod == "fc":
                self.fc_model = B1EPredModelFFN(embedding_dim=self.opt.config.ffn_dim//4, output_dim=self.opt.config.ffn_dim*num_hidden_layers)
                model_size = pretrained.split('-')[-1]
                model_path = f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1eL_fc1.pt'
                self.fc_model.load_state_dict(torch.load(model_path))
                self.fc_model = nn.DataParallel(self.fc_model)  # DataParallel wrapping
                # cuda it
                self.fc_model = self.fc_model.to(self._device)
        if self.maskmethod == "dejavu":
            for layer in range(num_hidden_layers):
                predm_ = B1ESeqPredModel(embedding_dim=self.opt.config.ffn_dim//4 , output_dim=num_heads)
                predm_.load_state_dict(torch.load(f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1e_seq_{layer}.pt'))
                predm_ = nn.DataParallel(predm_)
                predm_ = predm_.to(self._device)
                self.headpredictor_dict[layer] = predm_
            if self.fcmaskmethod == "fc":
                for layer in range(num_hidden_layers):
                    predm_ = B1ESeqPredModelFFN(embedding_dim=self.opt.config.ffn_dim//4 , output_dim=self.opt.config.ffn_dim)
                    predm_.load_state_dict(torch.load(f'pred_models_{model_size}/{zcp_calc}/{predictor_}/b1e_seq_fc1_{layer}.pt'))
                    predm_ = nn.DataParallel(predm_)
                    predm_ = predm_.to(self._device)
                    self.fcpredictor_dict[layer] = predm_

        # load all predictors from model_opt1.3b for each layer as model_layer_{layer_idx}.pt
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
                old_head_shape, old_fc_shape = self.head_mask.shape, self.ffn_fc_mask.shape
                if self.maskmethod == "original":
                    self.newmask = self.head_mask
                    self.ffn_fc_mask = self.ffn_fc_mask
                elif self.maskmethod == "predictor" or self.maskmethod == "predictorL":
                    self.newmask = torch.ones_like(self.head_mask)
                    self.ffn_fc_mask = torch.ones_like(self.ffn_fc_mask)
                    _, context_layer_val = self.opt.get_decoder().forward_first_layer(
                        input_ids=inps,
                        attention_mask=attn_mask
                    )
                    pred_inp = context_layer_val[:, -1, :].squeeze().float()
                    pred_inp = (pred_inp - pred_inp.min()) / (pred_inp.max() - pred_inp.min())
                    pred_inp = pred_inp if len(pred_inp.shape) == 2 else pred_inp.unsqueeze(0)
                    himp_vs = self.headpred_model(pred_inp)
                    
                    if self.prune_style == "perlayer":
                        for layer_idx, himp_v in enumerate(himp_vs.view(-1, self.opt.config.num_attention_heads)):
                            _, head_indices = torch.sort(himp_v.view(-1))
                            head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                            local_headmask = torch.ones_like(self.head_mask[layer_idx])
                            local_headmask[head_indices] = 0.
                            self.newmask[layer_idx] = local_headmask
                    elif self.prune_style == "global":
                        self.newmask = self.newmask.view(-1)
                        _, head_indices = torch.sort(himp_vs.view(-1))
                        head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                        self.newmask[head_indices] = 0.
                        self.newmask = self.newmask.view(old_head_shape)
                    elif self.prune_style == "hybrid":
                        self.newmask = self.newmask.view(-1)
                        _, head_indices = torch.sort(himp_vs.view(-1))
                        imp_head_indices = head_indices[int(self.head_percent_mask) * len(head_indices) // 100 : ]
                        head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                        self.newmask[head_indices] = 0.
                        self.newmask = self.newmask.view(old_head_shape)
                        # find index of newmask where .sum(dim=1) < self.num_min_heads code below
                        zeromask_ix = torch.where(self.newmask.sum(dim=1) < self.num_min_heads)[0]
                        num_heads_to_activate = len(zeromask_ix) * self.num_min_heads
                        # deactivate least important head indices from imp_head_indices  (its already sorted, use num_heads_to_activate)
                        self.newmask.view(-1)[imp_head_indices[:num_heads_to_activate]] = 0.
                        # Now, for each layer in zeromask_ix, activate self.num_min_heads heads
                        for layer_idx in zeromask_ix:
                            _, head_indices = torch.sort(himp_vs.view(old_head_shape)[layer_idx])
                            # take the indices of the MOST important heads
                            top_head_indices = head_indices[-self.num_min_heads:]
                            self.newmask[layer_idx][top_head_indices] = 1.
                    else:
                        raise ValueError("Invalid prune_style")
                    if self.fcmaskmethod == "fc":
                        fimp_vs = self.fc_model(pred_inp)
                        if self.prune_style == "perlayer":
                            for layer_idx, fimp_v in enumerate(fimp_vs.view(-1, self.opt.config.ffn_dim)):
                                _, f_indices = torch.sort(fimp_v.view(-1))
                                f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                                local_fcmask = torch.ones_like(self.ffn_fc_mask[layer_idx])
                                local_fcmask[f_indices] = 0.
                                self.ffn_fc_mask[layer_idx] = local_fcmask
                        elif self.prune_style == "global":
                            self.ffn_fc_mask = self.ffn_fc_mask.view(-1)
                            _, f_indices = torch.sort(fimp_vs.view(-1))
                            f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                            self.ffn_fc_mask[f_indices] = 0.
                            self.ffn_fc_mask = self.ffn_fc_mask.view(old_fc_shape)
                        elif self.prune_style == "hybrid":
                            self.ffn_fc_mask = self.ffn_fc_mask.view(-1)
                            _, f_indices = torch.sort(fimp_vs.view(-1))
                            imp_ffn_indices = f_indices[int(self.ffn_percent_mask) * len(f_indices) // 100 : ]
                            f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                            self.ffn_fc_mask[f_indices] = 0.
                            self.ffn_fc_mask = self.ffn_fc_mask.view(old_fc_shape)
                            # find index of newmask where .sum(dim=1) < self.num_min_ffn_neurons code below
                            zeromask_ix = torch.where(self.ffn_fc_mask.sum(dim=1) < self.num_min_ffn_neurons)[0]
                            num_ffn_to_activate = len(zeromask_ix) * self.num_min_ffn_neurons
                            # deactivate least important ffn indices from imp_ffn_indices  (its already sorted, use num_ffn_to_activate)
                            self.ffn_fc_mask.view(-1)[imp_ffn_indices[:num_ffn_to_activate]] = 0.
                            # Now, for each layer in zeromask_ix, activate self.num_min_ffn_neurons
                            for layer_idx in zeromask_ix:
                                _, f_indices = torch.sort(fimp_vs.view(old_fc_shape)[layer_idx])
                                # take the indices of the MOST important ffn neurons
                                top_ffn_indices = f_indices[-self.num_min_ffn_neurons:]
                                self.ffn_fc_mask[layer_idx][top_ffn_indices] = 1.
                        else:
                            raise ValueError("Invalid prune_style")
                elif self.maskmethod == "dejavu":
                    self.newmask = torch.ones_like(self.head_mask)
                    self.ffn_fc_mask = torch.ones_like(self.ffn_fc_mask)
                    _, context_layer_vals = self.opt.get_decoder().forward_all_layers(
                        input_ids=inps,
                        attention_mask=attn_mask
                    )
                    if self.prune_style == "perlayer":
                        for layer_idx, context_layer_val in enumerate(context_layer_vals):
                            pred_inp = context_layer_val[:, -1, :].squeeze().float()
                            pred_inp = (pred_inp - pred_inp.min()) / (pred_inp.max() - pred_inp.min())
                            pred_inp = pred_inp if len(pred_inp.shape) == 2 else pred_inp.unsqueeze(0)
                            himp_v = self.headpredictor_dict[layer_idx](pred_inp)
                            _, head_indices = torch.sort(himp_v.view(-1))
                            head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                            local_headmask = torch.ones_like(self.head_mask[layer_idx])
                            local_headmask[head_indices] = 0.
                            self.newmask[layer_idx] = local_headmask
                            if self.fcmaskmethod == "fc":
                                fimp_v = self.fcpredictor_dict[layer_idx](pred_inp)
                                _, f_indices = torch.sort(fimp_v.view(-1))
                                f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                                local_fcmask = torch.ones_like(self.ffn_fc_mask[layer_idx])
                                local_fcmask[f_indices] = 0.
                                self.ffn_fc_mask[layer_idx] = local_fcmask
                    elif self.prune_style == "global":
                        self.newmask = self.newmask.view(-1)
                        inpprep = [context_layer_vals[layer_idx][:, -1, :].squeeze().float() for layer_idx in range(len(context_layer_vals))]
                        inpprep = [xz if len(xz.shape) == 2 else xz.unsqueeze(0) for xz in inpprep]
                        himp_vs = torch.cat([self.headpredictor_dict[layer_idx](inpprep[layer_idx]) for layer_idx in range(len(context_layer_vals))], dim = 0)
                        _, head_indices = torch.sort(himp_vs.view(-1))
                        head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                        self.newmask[head_indices] = 0.
                        self.newmask = self.newmask.view(old_head_shape)
                        if self.fcmaskmethod == "fc":
                            self.ffn_fc_mask = self.ffn_fc_mask.view(-1)
                            inpprep = [context_layer_vals[layer_idx][:, -1, :].squeeze().float() for layer_idx in range(len(context_layer_vals))]
                            inpprep = [xz if len(xz.shape) == 2 else xz.unsqueeze(0) for xz in inpprep]
                            fimp_vs = torch.cat([self.fcpredictor_dict[layer_idx](inpprep[layer_idx]) for layer_idx in range(len(context_layer_vals))], dim = 0)
                            _, f_indices = torch.sort(fimp_vs.view(-1))
                            f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                            self.ffn_fc_mask[f_indices] = 0.
                            self.ffn_fc_mask = self.ffn_fc_mask.view(old_fc_shape)
                    elif self.prune_style == "hybrid":
                        self.newmask = self.newmask.view(-1)
                        inpprep = [context_layer_vals[layer_idx][:, -1, :].squeeze().float() for layer_idx in range(len(context_layer_vals))]
                        inpprep = [xz if len(xz.shape) == 2 else xz.unsqueeze(0) for xz in inpprep]
                        himp_vs = torch.cat([self.headpredictor_dict[layer_idx](inpprep[layer_idx]) for layer_idx in range(len(context_layer_vals))], dim = 0)
                        _, head_indices = torch.sort(himp_vs.view(-1))
                        imp_head_indices = head_indices[int(self.head_percent_mask) * len(head_indices) // 100 : ]
                        head_indices = head_indices[: int(self.head_percent_mask) * len(head_indices) // 100]
                        self.newmask[head_indices] = 0.
                        self.newmask = self.newmask.view(old_head_shape)
                        # find index of newmask where .sum(dim=1) < self.num_min_heads code below
                        zeromask_ix = torch.where(self.newmask.sum(dim=1) < self.num_min_heads)[0]
                        num_heads_to_activate = len(zeromask_ix) * self.num_min_heads
                        # deactivate least important head indices from imp_head_indices  (its already sorted, use num_heads_to_activate)
                        self.newmask.view(-1)[imp_head_indices[:num_heads_to_activate]] = 0.
                        # Now, for each layer in zeromask_ix, activate self.num_min_heads
                        for layer_idx in zeromask_ix:
                            _, head_indices = torch.sort(himp_vs.view(old_head_shape)[layer_idx])
                            # take the indices of the MOST important heads
                            top_head_indices = head_indices[-self.num_min_heads:]
                            self.newmask[layer_idx][top_head_indices] = 1.
                        if self.fcmaskmethod == "fc":
                            self.ffn_fc_mask = self.ffn_fc_mask.view(-1)
                            inpprep = [context_layer_vals[layer_idx][:, -1, :].squeeze().float() for layer_idx in range(len(context_layer_vals))]
                            inpprep = [xz if len(xz.shape) == 2 else xz.unsqueeze(0) for xz in inpprep]
                            fimp_vs = torch.cat([self.fcpredictor_dict[layer_idx](inpprep[layer_idx]) for layer_idx in range(len(context_layer_vals))], dim = 0)
                            _, f_indices = torch.sort(fimp_vs.view(-1))
                            imp_ffn_indices = f_indices[int(self.ffn_percent_mask) * len(f_indices) // 100 : ]
                            f_indices = f_indices[: int(self.ffn_percent_mask) * len(f_indices) // 100]
                            self.ffn_fc_mask[f_indices] = 0.
                            self.ffn_fc_mask = self.ffn_fc_mask.view(old_fc_shape)
                            # find index of newmask where .sum(dim=1) < self.num_min_ffn_neurons code below
                            zeromask_ix = torch.where(self.ffn_fc_mask.sum(dim=1) < self.num_min_ffn_neurons)[0]
                            num_ffn_to_activate = len(zeromask_ix) * self.num_min_ffn_neurons
                            # deactivate least important ffn indices from imp_ffn_indices  (its already sorted, use num_ffn_to_activate)
                            self.ffn_fc_mask.view(-1)[imp_ffn_indices[:num_ffn_to_activate]] = 0.
                            # Now, for each layer in zeromask_ix, activate self.num_min_ffn_neurons
                            for layer_idx in zeromask_ix:
                                _, f_indices = torch.sort(fimp_vs.view(old_fc_shape)[layer_idx])
                                # take the indices of the MOST important ffn neurons
                                top_ffn_indices = f_indices[-self.num_min_ffn_neurons:]
                                self.ffn_fc_mask[layer_idx][top_ffn_indices] = 1.
                    else:
                        raise ValueError("Invalid prune_style")
                elif self.maskmethod == "kmeans":
                    raise NotImplementedError
                elif self.maskmethod == "none":
                    self.newmask = torch.ones_like(self.head_mask)
                    self.ffn_fc_mask = torch.ones_like(self.ffn_fc_mask)
                else:
                    raise ValueError("Invalid maskmethod")
                self.newmask = self.newmask.view(old_head_shape).half().contiguous().to(self._device)
                if self.fcmaskmethod == "fc":
                    self.ffn_fc_mask = self.ffn_fc_mask.view(old_fc_shape).half().contiguous().to(self._device)
                # make all first layer newmask stuff 1
                self.newmask[0, :] = 1.
                self.ffn_fc_mask[0, :] = 1.
                # for lix in range(self.newmask.shape[0]):
                #     if lix <= int(0.35 * self.newmask.shape[0]) or lix >= int(0.65 * self.newmask.shape[0]):
                #         self.newmask[lix, :] = 1.
                #         self.ffn_fc_mask[lix, :] = 1.
                # temporarily, fix sparsity for the first 35% and last 35% of the layer newask and ffn_fc_mask at 
                self._temp_tracker += 1
                resultant_act = self.opt(input_ids = inps, head_mask = self.newmask.half().to(self._device), fc_mask = self.ffn_fc_mask.to(self._device))[0][:, :, :50265]
                if resultant_act.isnan().any():
                    import pdb; pdb.set_trace()
                # return self.opt(input_ids = inps, head_mask = self.newmask.half().to(self._device), fc_mask = self.ffn_fc_mask.to(self._device))[0][:, :, :50265]
                return resultant_act
        else:
            return self.opt(input_ids = inps, attention_mask = attn_mask, labels = labels).loss
            
    def _model_generate(self, context, max_length, eos_token_id):
        return self.opt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
OPTLM = HFLM