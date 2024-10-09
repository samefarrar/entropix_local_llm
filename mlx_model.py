import mlx.nn as nn
import mlx.core as mx
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.models.llama import Attention, Model, LlamaModel, TransformerBlock, ModelArgs, create_attention_mask
from mlx_lm.utils import load_config, _get_classes
from mlx_lm import load
from typing import Tuple, Optional, Callable
from types import MethodType
from mlx_stats import AttnStats
import inspect
from pathlib import Path
import glob

class EntropyAttention(Attention):
    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)
        self.n_reps = self.n_heads // self.n_kv_heads

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, mx.array]:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3) # (B, n_heads, L, head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3) # (B, n_kv_heads, L, head_dim)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3) # (B, n_kv_heads, L, head_dim)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        shaped_keys = mx.repeat(keys, repeats=self.n_reps, axis = 1).transpose(0, 1, 3, 2) # (B, n_heads, L, head_dim)
        pre_scores = mx.matmul(queries, shaped_keys) * self.scale # (B, head_dim, L, L)
        # Waiting patiently for einsum to be supported in MLX
        # pre_scores = einsum(queries, shaped_keys, 'b h i d, b h j d -> b h i j', ) / mx.sqrt(self.n_heads)

        # You might think - why are we computing the matrix multiplication twice!
        # Once for pre_scores and once for output, turns out mx.fast.scaled_dot_product_attention
        # is SO fast that it's faster to compute the matrix multiplication twice than to implement it
        # ourselves
        # 52 tok/s with Peak memory: 4.806 GB vs 13 tok/s with Peak memory: 7.156 GB
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), pre_scores

class EntropyTransformerBlock(TransformerBlock):
    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)
        self.self_attn = EntropyAttention(model_args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, mx.array]:
        r, scores = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, scores

class EntropyLlamaModel(LlamaModel):
    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)
        self.layers = [
            EntropyTransformerBlock(model_args) for _ in range(model_args.num_hidden_layers)
        ]
        self.n_heads = model_args.num_attention_heads

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = create_attention_mask(h, cache)
        attention_stats = AttnStats.new(h.shape[0], n_layers = len(self.layers), n_heads = self.n_heads)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h, scores = layer(h, mask, cache=c)
            attention_stats = attention_stats.update(scores[:, :, -1, :], i)

        return self.norm(h), scores, attention_stats

class EntropixModel(Model):
    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)
        self.model = EntropyLlamaModel(model_args)
        print(f"I have a model: {self.model_type}")
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, scores, attention_stats = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out, scores, attention_stats

def load_entropix_model(model_path: Path, lazy = False):
    config = load_config(model_path)

    _, model_args_class = _get_classes(config=config)
    model_class = EntropixModel

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Print the attributes of model
    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}

    for wf in weight_files:
        weights.update(mx.load(wf))

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model

if __name__ == "__main__":
    path = Path("weights/Llama-3.2-1B-Instruct")
    entropix_model = load_entropix_model(path)
