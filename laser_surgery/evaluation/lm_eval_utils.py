from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedModel, PreTrainedTokenizer
import lm_eval
from lm_eval import evaluator


def evaluate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    hf_lm = HFLM(pretrained=model, tokenizer=tokenizer)  # batch_size = ...

    # make sure the right tasks are available
    # lm_eval.tasks.ALL_TASKS
    # {'scrolls_govreport', 'scrolls_summscreenfd', 'scrolls_qasper', 'scrolls_contractnli', 'scrolls_qmsum',
    # 'scrolls_narrativeqa', 'squadv2', 'scrolls_quality'}

    # Choose tasks for evaluation
    # (copied from https://github.com/cognitivecomputations/laserRMT/blob/main/script_lm_eval.sh )
    # fewshots:          5,      5,           25,               10,          5,       1
    # batch_sizes:       1,      4,           2,                2,           2,       4
    tasks_to_evaluate = {
        "mmlu": 5, "winogrande": 5, "arc_challenge": 25, "hellaswag": 10, "gsm8k": 5, "truthfulqa_mc2": 1
    }
    for task, fewshot in tasks_to_evaluate.items():
        # Run the evaluation
        results = evaluator.evaluate(lm=hf_lm, tasks=[task], num_fewshot=fewshot,
                                     limit=100, bootstrap_iters=100, write_out=True, log_samples=True)[task]
        accuracy = results['acc_[FIXME SOME KEY]']
        # do something?

    return results

