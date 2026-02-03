from typing import List, Optional
import ctypes
import os
import torch
from transformers import AutoConfig

# 定义 C++ 结构体
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", ctypes.c_void_p),
        ("out_embed", ctypes.c_void_p),
        ("out_norm_w", ctypes.c_void_p),
        ("attn_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("attn_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_norm_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_gate_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_up_w", ctypes.POINTER(ctypes.c_void_p)),
        ("mlp_down_w", ctypes.POINTER(ctypes.c_void_p)),
    ]

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int), 
        ("nlayer", ctypes.c_size_t), ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t), ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t), ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t), ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float), ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

class Qwen2:
    def __init__(self, pkg_path: str, device: str = "cpu"):
        self.pkg_path = pkg_path
        self.config = AutoConfig.from_pretrained(pkg_path)
        
        # 1. 使用已加载的 Python 包装层共享库（避免重复加载导致符号混淆）
        try:
            from llaisys import libllaisys as _libmod
            self.lib = _libmod.LIB_LLAISYS
            print(f"Using shared library from llaisys package: {self.lib}")
        except Exception:
            # 备用：直接加载构建产物
            lib_path = os.path.abspath(os.path.join(os.getcwd(), "build/libllaisys.so"))
            if not os.path.exists(lib_path):
                lib_path = os.path.abspath(os.path.join(os.getcwd(), "build/linux/x86_64/release/libllaisys.so"))
            print(f"Loading library from: {lib_path}")
            self.lib = ctypes.CDLL(lib_path)
        
        # 2. 准备 Meta
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = 13     # FLOAT32
        self.meta.nlayer = self.config.num_hidden_layers
        self.meta.hs = self.config.hidden_size
        self.meta.nh = self.config.num_attention_heads
        self.meta.nkvh = self.config.num_key_value_heads
        self.meta.dh = self.config.hidden_size // self.config.num_attention_heads
        self.meta.di = self.config.intermediate_size
        self.meta.maxseq = self.config.max_position_embeddings
        self.meta.voc = self.config.vocab_size
        self.meta.epsilon = self.config.rms_norm_eps
        self.meta.theta = getattr(self.config, "rope_theta", 10000.0)
        self.meta.end_token = 151643 # EOS

        # 3. 创建模型
        self.lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p
        device_ids = (ctypes.c_int * 1)(0)
        dev_enum = 1 if "cuda" in str(device) else 0
        
        self.model = self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), dev_enum, device_ids, 1
        )
        
        self.lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)
        self.lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64
        
        # 修正：使用正确的函数名 tensorGetData
        try:
            self.lib.tensorGetData.restype = ctypes.c_void_p
        except:
            print("Error: tensorGetData symbol not found!")

    def load_weights(self):
        from safetensors.torch import load_file
        print(f"Loading weights from {self.pkg_path}...")
        
        w_struct = self.lib.llaisysQwen2ModelWeights(self.model).contents
        sf_path = os.path.join(self.pkg_path, "model.safetensors")
        state_dict = load_file(sf_path)
        
        def copy_tensor(c_ptr, torch_tensor):
            if not c_ptr: return
            
            # --- 关键修改：强制转为 Float32 ---
            # 原来的模型可能是 BFloat16，如果不转，C++ 后端可能无法识别或处理
            t = torch_tensor.to(torch.float32).contiguous()
            # --------------------------------
            
            # 使用修正后的函数名
            data_ptr = self.lib.tensorGetData(c_ptr)
            if data_ptr:
                ctypes.memmove(data_ptr, t.data_ptr(), t.numel() * 4) # float32 是 4 字节

        for name, tensor in state_dict.items():
            if "embed_tokens" in name: copy_tensor(w_struct.in_embed, tensor)
            elif "lm_head" in name: copy_tensor(w_struct.out_embed, tensor)
            elif "model.norm.weight" in name: copy_tensor(w_struct.out_norm_w, tensor)
            elif "layers" in name:
                parts = name.split(".")
                idx = int(parts[2])
                layer_w = tensor
                
                if "input_layernorm" in name: copy_tensor(w_struct.attn_norm_w[idx], layer_w)
                elif "self_attn.q_proj.weight" in name: copy_tensor(w_struct.attn_q_w[idx], layer_w)
                elif "self_attn.q_proj.bias" in name: copy_tensor(w_struct.attn_q_b[idx], layer_w)
                elif "self_attn.k_proj.weight" in name: copy_tensor(w_struct.attn_k_w[idx], layer_w)
                elif "self_attn.k_proj.bias" in name: copy_tensor(w_struct.attn_k_b[idx], layer_w)
                elif "self_attn.v_proj.weight" in name: copy_tensor(w_struct.attn_v_w[idx], layer_w)
                elif "self_attn.v_proj.bias" in name: copy_tensor(w_struct.attn_v_b[idx], layer_w)
                elif "self_attn.o_proj.weight" in name: copy_tensor(w_struct.attn_o_w[idx], layer_w)
                elif "post_attention_layernorm" in name: copy_tensor(w_struct.mlp_norm_w[idx], layer_w)
                elif "mlp.gate_proj.weight" in name: copy_tensor(w_struct.mlp_gate_w[idx], layer_w)
                elif "mlp.up_proj.weight" in name: copy_tensor(w_struct.mlp_up_w[idx], layer_w)
                elif "mlp.down_proj.weight" in name: copy_tensor(w_struct.mlp_down_w[idx], layer_w)

    # def generate(self, prompt_tokens: List[int], max_new_tokens: int = 100, **kwargs) -> List[int]:
    #     tokens = list(prompt_tokens)
    #     for _ in range(max_new_tokens):
    #         c_tokens = (ctypes.c_int64 * len(tokens))(*tokens)
    #         next_token = self.lib.llaisysQwen2ModelInfer(self.model, c_tokens, len(tokens))
            
    #         if next_token == self.meta.end_token or next_token == 151645: 
    #             break
    #         tokens.append(next_token)
    #     return tokens
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 128, **kwargs) -> List[int]:
        tokens = list(prompt_tokens)
        
        print(f"[Llaisys] Start generating (max {max_new_tokens} tokens)...")
        
        for i in range(max_new_tokens):
            # 准备数据传给 C++
            c_tokens = (ctypes.c_int64 * len(tokens))(*tokens)
            
            # 调用 C++ 推理
            next_token = self.lib.llaisysQwen2ModelInfer(self.model, c_tokens, len(tokens))
            
            # EOS (End of Sentence) 检测
            # DeepSeek/Qwen 的常见结束符: 
            # 151643: <|end_of_text|>
            # 151645: <|im_end|>
            if next_token == self.meta.end_token or next_token in [151643, 151645]: 
                break
            
            tokens.append(next_token)
            
            # (可选) 稍微打印一点进度条，避免看起来像卡死了
            # print(".", end="", flush=True) 

        print("\n[Llaisys] Generation done.")
        return tokens
            
        print("[DEBUG] Generation finished.")
        return tokens