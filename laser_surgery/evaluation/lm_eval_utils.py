from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval import evaluator


def evaluate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    hf_lm = HFLM(pretrained=model, tokenizer=tokenizer)

    # Choose tasks for evaluation TODO: pick from the laserRMT tasks
    tasks_to_evaluate = ["sst", "cola"]  # Replace with the tasks of your choice

    # Run the evaluation
    results = evaluator.evaluate(lm=hf_lm, tasks=tasks_to_evaluate, num_fewshot=0)
    return results

