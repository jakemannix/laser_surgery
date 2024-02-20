# laser_surgery
Exploration with variations of LAyer SElective Rank reduction for LLMs

Main entrypoint: `optimization.laser_surgery` module, and the `scan_layers_and_report` function.

API of `scan_layers_and_report`:
```python
def scan_layers_and_report(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                           dataset_name: str, split: str, max_length: int,
                           layer_type: str, layer_number: int,
                           rank_override: int = None,
                           num_samples: int = 16, seed: int = 0):
```

Usage: 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimization.laser_surgery import scan_layers_and_report

model_name = "microsoft/phi-2"  # for use with the LASER paper: "EleutherAI/gpt-j-6b"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

reducer = scan_layers_and_report(model=model,
                                 tokenizer=tokenizer, 
                                 dataset_name="c-s-ale/dolly-15k-instruction-alpaca-format", 
                                 split="train", max_length=128, 
                                 layers_type="mlp.fc1", layer_number=31, 
                                 rank_override=64, num_samples=16, seed=137)
```

Note: the current evaluation method is based on raw language model loss, and from the paper,
the models don't show "improved" LM loss. However, the models do show improved performance 
on downstream tasks, so the evaluation method may need to be adjusted (see lm_eval_utils.py).
