from functools import lru_cache
from typing import List
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


def per_token_average_loss(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, texts: List[str],
                           max_length: int = 512,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu") -> float:
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loss_fct = CrossEntropyLoss(reduction='none')

    total_loss = 0
    total_tokens = 0

    for text in tqdm(texts, desc="Processing Texts"):
        # Tokenize input, we don't need labels here
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Shift the input and label so that each token is predicted from the tokens before it
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()  # Adjust attention mask to match shifted inputs

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Calculate per-token loss without reduction
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Mask for active tokens (excluding padding)
        active_loss = attention_mask.view(-1) == 1
        active_loss_values = loss[active_loss]

        # Accumulate the loss from active tokens and count them
        total_loss += active_loss_values.sum().item()
        total_tokens += active_loss.sum().item()

    # Calculate average loss across all active tokens
    average_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return average_loss


# dataset I've been testing with: c-s-ale/dolly-15k-instruction-alpaca-format
@lru_cache
def sample_data(dataset_name: str, split: str = "train",
                num_samples: int = 16, seed: int = 42) -> List[str]:
    data = load_dataset(dataset_name, split=split).shuffle(seed=seed).select(range(num_samples))
    data = data.map(lambda x: {'text': x['instruction'] + '\n\n' + x['output'] + '\n'})
    return data["text"]


# Note, from the paper, we will find that perplexity does not improve with the reduction of the model.
# "While there is an improvement in the task at hand, the model’s perplexity worsens slightly after
# applying LASER. We do not yet fully understand what the worsening in perplexity of the model corresponds
# to and leave this for future study"
# and
# "For layers corresponding to the MLP input matrices, the
# perplexity of the model increases from 4.8 to 5.0, showing that the language modeling objective is indeed
# slightly effected. For the MLP output layers, the perplexity of GPT-J on PILE increases from 4.8 to 4.9 with
# LASER. It may be possible to fix this small degradation by calibrating the temperature of the model."
# from https://arxiv.org/abs/2312.13558
def evaluate_loss(model, tokenizer, dataset_name, split, num_samples=10, max_length=512, seed=42):
    texts = sample_data(dataset_name, split, num_samples, seed)
    return per_token_average_loss(model=model, tokenizer=tokenizer, texts=texts, max_length=max_length)
