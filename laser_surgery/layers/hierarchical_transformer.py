import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP, apply_rotary_pos_emb


class LoRALayer(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super(LoRALayer, self).__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.lora_a = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x, use_lora=True):
        if use_lora:
            return self.linear_layer(x) + self.lora_b(self.lora_a(x))
        else:
            return self.linear_layer(x)


class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx, window_size):
        super(HierarchicalTransformerBlock, self).__init__()
        self.window_size = window_size
        self.attention = MistralAttention(config, layer_idx=layer_idx)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp_block = MistralMLP(config)
        self.q_lora = LoRALayer(self.attention.q_proj)
        self.k_lora = LoRALayer(self.attention.k_proj)
        self.v_lora = LoRALayer(self.attention.v_proj)
        self.out_lora = LoRALayer(self.attention.o_proj)
        self.mlp_fc1_lora = LoRALayer(self.mlp_block.gate_proj)
        self.mlp_fc2_lora = LoRALayer(self.mlp_block.up_proj)
        self.mlp_fc3_lora = LoRALayer(self.mlp_block.down_proj)

    def bottom_up_aggregation(self, token_embeddings, pseudo_tokens=None):
        pseudo_tokens_levels = [token_embeddings]
        current_tokens = token_embeddings

        if pseudo_tokens is not None:
            pseudo_tokens_levels.extend(pseudo_tokens)
            current_tokens = pseudo_tokens[-1]
        else:
            while current_tokens.size(1) > self.window_size:
                num_pseudo_tokens = current_tokens.size(1) // self.window_size

                pooled_tokens = current_tokens[:, :num_pseudo_tokens * self.window_size, :].reshape(
                    current_tokens.size(0), num_pseudo_tokens, self.window_size, current_tokens.size(2)
                ).max(dim=2)[0]

                pseudo_tokens_levels.append(pooled_tokens)
                current_tokens = pooled_tokens

        return pseudo_tokens_levels

    def top_down_propagation(self, pseudo_tokens_levels):
        num_levels = len(pseudo_tokens_levels) - 1
        for level in range(num_levels, 0, -1):
            higher_level_tokens = pseudo_tokens_levels[level]
            lower_level_tokens = pseudo_tokens_levels[level - 1]

            batch_size, seq_len_lower, d_model = lower_level_tokens.shape
            _, seq_len_higher, _ = higher_level_tokens.shape

            updated_lower_level_tokens = torch.zeros_like(lower_level_tokens)

            for i in range(seq_len_higher):
                start_idx = i * self.window_size
                end_idx = start_idx + self.window_size

                window_lower = lower_level_tokens[:, start_idx:end_idx, :]

                Q_lower = self.q_lora(window_lower)
                K_higher = self.k_lora(higher_level_tokens[:, i:i + 1, :]).expand(-1, self.window_size, -1)
                V_higher = self.v_lora(higher_level_tokens[:, i:i + 1, :]).expand(-1, self.window_size, -1)

                attn_scores = torch.matmul(Q_lower, K_higher.transpose(-2, -1)) / torch.sqrt(
                    torch.tensor(d_model).float())
                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                attn_output = torch.matmul(attn_weights, V_higher)

                updated_lower_level_tokens[:, start_idx:end_idx, :] = window_lower + attn_output

            pseudo_tokens_levels[level - 1] = updated_lower_level_tokens

        return pseudo_tokens_levels

    def forward(self, token_embeddings, pseudo_tokens=None, use_lora=False):
        # Bottom-Up Aggregation
        pseudo_tokens_levels = self.bottom_up_aggregation(token_embeddings, pseudo_tokens)

        for level in range(len(pseudo_tokens_levels)):
            # Apply normalization and attention with residual connection
            x = pseudo_tokens_levels[level]
            x_norm = self.norm1(x)
            attn_output = self.attention(x_norm)
            if use_lora:
                attn_output = self.out_lora(attn_output, use_lora)
            x = x + attn_output  # Residual connection
            x_norm = self.norm2(x)
            mlp_output = self.mlp_block(x_norm)
            if use_lora:
                mlp_output = self.mlp_fc3_lora(self.mlp_fc2_lora(self.mlp_fc1_lora(mlp_output, use_lora), use_lora),
                                               use_lora)
            x = x + mlp_output  # Residual connection
            pseudo_tokens_levels[level] = x

        # Top-Down Propagation
        pseudo_tokens_levels = self.top_down_propagation(pseudo_tokens_levels)

        return pseudo_tokens_levels[0], pseudo_tokens_levels[1:] if len(pseudo_tokens_levels) > 1 else None


class HierarchicalTransformer(nn.Module):
    def __init__(self, pretrained_model, num_layers, d_model, n_heads, window_size):
        super(HierarchicalTransformer, self).__init__()
        self.config = MistralConfig(hidden_size=d_model, num_attention_heads=n_heads)
        self.pretrained_model = pretrained_model
        self.embedding = pretrained_model.transformer.wte
        self.hierarchical_blocks = nn.ModuleList([
            HierarchicalTransformerBlock(self.config, layer_idx=i, window_size=window_size) for i in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, pretrained_model.config.vocab_size, bias=False)
        self._initialize_weights_from_pretrained_model()

    def _initialize_weights_from_pretrained_model(self):
        for i, block in enumerate(self.hierarchical_blocks):
            pretrained_layer = self.pretrained_model.transformer.h[i]
            block.attention.q_proj.weight.data = pretrained_layer.attn.q_proj.weight.data.clone()
            block.attention.q_proj.bias.data = pretrained_layer.attn.q_proj.bias.data.clone()
            block.attention.k_proj.weight.data = pretrained_layer.attn.k_proj.weight.data.clone()
            block.attention.k_proj.bias.data = pretrained_layer.attn.k_proj.bias.data.clone()
            block.attention.v_proj.weight.data = pretrained_layer.attn.v_proj.weight.data.clone()
            block.attention.v_proj.bias.data = pretrained_layer.attn.v_proj.bias.data.clone()
            block.attention.o_proj.weight.data = pretrained_layer.attn.out_proj.weight.data.clone()
            block.attention.o_proj.bias.data = pretrained_layer.attn.out_proj.bias.data.clone()
            block.mlp_block.gate_proj.weight.data = pretrained_layer.mlp.fc1.weight.data.clone()
            block.mlp_block.gate_proj.bias.data = pretrained_layer.mlp.fc1.bias.data.clone()
            block.mlp_block.up_proj.weight.data = pretrained_layer.mlp.fc2.weight.data.clone()
            block.mlp_block.up_proj.bias.data = pretrained_layer.mlp.fc2.bias.data.clone()
            block.mlp_block.down_proj.weight.data = pretrained_layer.mlp.fc3.weight.data.clone()
            block.mlp_block.down_proj.bias.data = pretrained_layer.mlp.fc3.bias.data.clone()

    def forward(self, input_ids, attention_mask=None, pseudo_tokens=None, use_lora=False):
        # Embedding and RoPE
        token_embeddings = self.embedding(input_ids)
        seq_len = token_embeddings.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        token_embeddings = apply_rotary_pos_emb(token_embeddings, pos_ids)

        # Hierarchical Transformer Blocks
        for block in self.hierarchical_blocks:
            token_embeddings, pseudo_tokens = block(token_embeddings, pseudo_tokens, use_lora)

        # Output projection to vocab size logits
        logits = self.lm_head(token_embeddings)
        return logits


def try_it_out():
    # Load a pretrained Mistral-7B model
    pretrained_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Define the hierarchical transformer model
    num_layers = 32  # Number of layers in Mistral-7B
    d_model = 4096
    n_heads = 32
    window_size = 16
    hierarchical_transformer = HierarchicalTransformer(pretrained_model, num_layers, d_model, n_heads, window_size)

    # Example input
    input_ids = torch.randint(0, 1000, (1, 1024))  # Example input
    logits = hierarchical_transformer(input_ids, use_lora=True)
