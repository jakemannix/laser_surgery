from dataclasses import dataclass
from typing import Callable, List, Set, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.logging import get_logger
import torch
from torch import nn, Tensor
from torch.nn.modules import Module
import numpy as np
from evaluation.loss_evaluator import evaluate_loss


class FactoredLinear(nn.Module):
    """
    Layer that represents the SVD factorization of a linear layer, U * S * V^T,
    where U has shape (out_features, small_rank), S has shape (small_rank, small_rank),
    V^T has shape (small_rank, in_features), and the bias has shape (out_features).
    """
    def __init__(self, U: Tensor, S: Tensor, Vt: Tensor, bias: Tensor = None,
                 device: torch.device = torch.device('cuda')):
        super(FactoredLinear, self).__init__()
        self.U = U.to(device).float()    # Shape: (out_features, small_rank)
        self.S = S.to(device).float()    # Shape: (small_rank, small_rank)
        self.Vt = Vt.to(device).float()  # Shape: (small_rank, in_features)
        self.bias = bias.to(device).float() if bias is not None else None
        self.device = device
        self.logger = get_logger()
        self.logger.debug(f"FactoredLinear initialized with U.shape={self.U.shape}, S.shape={self.S.shape}, "
                          f"Vt.shape={self.Vt.shape} and bias.shape="
                          f"{self.bias.shape if self.bias is not None else None}")

    def __repr__(self):
        return f"FactoredLinear(in_features={self.Vt.shape[1]}, out_features: {self.U.shape[0]}, " \
               f"bias={True if self.bias is not None else False}, reduced_rank: {self.S.shape[0]})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"FactoredLinear.forward: x.shape={x.shape}, U.shape={self.U.shape}, S.shape={self.S.shape}, " +
        #      f"Vt.shape={self.Vt.shape}, bias.shape={self.bias.shape if self.bias is not None else None}")

        # if x.shape is (1, seq_len, in_features), we need to remove the first dimension
        must_unsqueeze = False
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
            must_unsqueeze = True
        x = x.to(self.device)
        # TODO: handle batch sizes > 1 by looking for or writing a batched multi_dot implementation
        # NOTE: we can probably do this with sequential calls to torch.bmm to start out.
        # as the major gains should come from the reordering of the summation in the multi-matrix product.
        # also note: x.shape (seq_len, in_features), must be transposed and then undone on return
        result = torch.linalg.multi_dot([self.U, torch.diag(self.S), self.Vt, x.t()]).t()
        result = result + self.bias if self.bias is not None else result
        if must_unsqueeze:
            result = result.unsqueeze(0)
        return result

    @staticmethod
    def from_linear(linear: nn.Linear, small_rank: int, device: torch.device = torch.device('cuda')):
        weights: Tensor = linear.weight.double()
        bias: Tensor = linear.bias.double() if linear.bias is not None else None
        U, S, Vt = torch.linalg.svd(weights, full_matrices=False)
        U, S, Vt = U[:, :small_rank], S[:small_rank], Vt[:small_rank, :]
        return FactoredLinear(U, S, Vt, bias, device)


# TODO: this assumes x is a single input, not a batch
def compare_factorization(linear: nn.Linear, rank: int, x: Tensor):
    factored = FactoredLinear.from_linear(linear, rank)
    reconstructed = factored.U @ torch.diag(factored.S) @ factored.Vt
    bias = linear.bias if linear.bias is not None else 0
    with torch.no_grad():
        factored_output = factored(x)
        reconstructed_output = reconstructed(x) + bias
        return torch.allclose(factored_output, reconstructed_output, atol=1e-7)


def factorization_latency_perf_test(linear: nn.Linear, rank: int, x: Tensor, num_passes: int = 1000):
    # first perf-test how long it takes to run the forward pass with the linear layer, then with the factored layer
    import time
    start = time.perf_counter()
    for _ in range(num_passes):
        y = linear(x)
    end = time.perf_counter()
    linear_time = end - start
    print(f"Linear layer took {linear_time} seconds to run {num_passes} forward passes")
    factored_layer = FactoredLinear.from_linear(linear, rank)
    start = time.perf_counter()
    for _ in range(num_passes):
        y = factored_layer(x)
    end = time.perf_counter()
    factored_time = end - start
    print(f"Factored layer (rank: {rank}) took {factored_time} seconds to run {num_passes} forward passes")
    print(f"fractional time =  {factored_time/linear_time} * linear_time")
    return linear_time, factored_time


class Validator:
    def __init__(self,
                 metric_fn: Callable[[PreTrainedModel], float]):
        self.metric_fn = metric_fn
        self.logger = get_logger()

    def get_metric(self, model: PreTrainedModel):
        return self.metric_fn(model)

    def validate(self, model: PreTrainedModel, original_metric_value: float, max_metric_ratio_factor: float):
        metric_value = self.metric_fn(model)
        self.logger.info(f"relative change in perplexity: {1 - (metric_value / original_metric_value)}")
        if metric_value > original_metric_value * max_metric_ratio_factor:
            return False
        return True

    def to_evaluator(self,
                     original_value: float, max_metric_ratio_factor: float) -> Callable[[PreTrainedModel], float]:
        return lambda model: self.validate(model, original_value, max_metric_ratio_factor)

    @staticmethod
    def get_dataset_perplexity_validator(
            tokenizer: PreTrainedTokenizer, dataset_name: str, split: str,
            max_length: int, num_samples: int = 16, seed: int = 0):
        def perplexity_fn(model: PreTrainedModel):
            return evaluate_loss(
                model=model, tokenizer=tokenizer,
                dataset_name=dataset_name, split=split,
                max_length=max_length, num_samples=num_samples, seed=seed)

        return Validator(perplexity_fn)


@dataclass
class ModelModification:
    module_name: str
    layer_number: int
    original_shape: tuple[int, int]
    reduced_rank: int


class PrismaticLaserReducer:
    def __init__(self,
                 model: PreTrainedModel,
                 evaluator: Callable[[PreTrainedModel], float]):
        self.logger = get_logger()
        self.model: PreTrainedModel = model
        self.evaluator = evaluator
        print("Evaluating base model")
        self.base_evaluation = evaluator(model)
        print(f"Base evaluation: {self.base_evaluation}")
        self.layers_checked_for_modification: Set[str] = set()
        self.original_layers: Dict[str, (Module, Module)] = {}
        self.modifications: List[ModelModification] = []

    @staticmethod
    def for_model(model: PreTrainedModel, tokenizer,
                  dataset_name: str, split: str, max_length: int,
                  num_samples=16, seed=0) -> 'PrismaticLaserReducer':
        validator = Validator.get_dataset_perplexity_validator(
            tokenizer=tokenizer, dataset_name=dataset_name, split=split,
            max_length=max_length, num_samples=num_samples, seed=seed)
        return PrismaticLaserReducer(model, validator.metric_fn)

    def get_reduced_model(self) -> PreTrainedModel:
        return self.model

    def reduce_layers_matching(self, layer_type: str, layer_number: Optional[int], rank_override=None):
        """
        Strategy for updating the model: We will loop over all the named modules in the model (filtering by layer_type
        and layer_number), and replace nn.Linear layers with a reduced rank factorization which is API-compatible
        with nn.Linear - FactoredLinear, which replaces the single weight matrix with two smaller matrices of rank r.
        The choice of r will be determined by the Marchenko-Pastur threshold, which is a statistical measure of the
        signal-to-noise ratio of the singular values of the weight matrix. After computing the lower-rank
        representation, we'll compute the perplexity of the model using the replaced layer, and compare it to the
        original perplexity. If the perplexity is better than a threshold (e.g. 1% improvement), we'll keep the
        reduced-rank factorization. Alternatively, we can simply pick the Marchenko-Pastur threshold (optionally
        scaled by a factor) as the rank r to keep.
        :param layer_number:
        :param layer_type:
        :param rank_override:
        :return:
        """
        named_modules: list[tuple[str, Module]] = list(self.model.named_modules())
        # Iterate over all named modules
        for module_name, module in named_modules:
            if module_name in self.layers_checked_for_modification:
                print(f"Layer {module_name} has already been modified. Skipping.")
                continue
            self.layers_checked_for_modification.add(module_name)
            if (layer_type in module_name
                    and (layer_number is None or str(layer_number) in module_name)
                    and isinstance(module, nn.Linear)):
                *parent_path, child_name = module_name.split('.')
                parent = ".".join(parent_path)
                parent_module = PrismaticLaserReducer.get_module_by_name(self.model, parent)
                print(f"Reconstructing layer: {child_name} in {parent}")
                self.reduce_rank_module(child_name=child_name, parent=parent, layer_number=layer_number,
                                        module=module, parent_module=parent_module,
                                        rank_override=rank_override)

    # module_name might be model.layers.31.mlp.fc1
    def reduce_rank_module(self, child_name: str, parent: str, layer_number: Optional[int],
                           module: nn.Linear, parent_module: Module,
                           rank_override=None):
        module_name = f"{parent}.{child_name}"
        self.original_layers[module_name] = (parent_module, module)
        weights: Tensor = module.weight.double()
        bias: Tensor = module.bias.double() if module.bias is not None else None
        U, S, V = torch.linalg.svd(weights, full_matrices=False)
        reduced_rank = get_rank(S, weights.size(0), weights.size(1)) if rank_override is None else rank_override
        print(f"Reducing {module_name} from {S.shape} to {reduced_rank}")
        U, S, V = U[:, :reduced_rank], S[:reduced_rank], V[:reduced_rank, :]
        self.modifications.append(ModelModification(module_name=module_name, layer_number=layer_number,
                                                    original_shape=(weights.size(0), weights.size(1)),
                                                    reduced_rank=reduced_rank))
        factored_layer = FactoredLinear(U, S, V, bias)
        setattr(parent_module, child_name, factored_layer)

    @staticmethod
    def get_module_by_name(model: PreTrainedModel, module_path: str) -> Module:
        for module_name, model_module in model.named_modules():
            if module_name == module_path:
                return model_module

    @staticmethod
    def replace_module(model, module_path, new_module):
        *parent_path, child_name = module_path.split('.')
        parent = model
        for name in parent_path:
            parent = getattr(parent, name)
        setattr(parent, child_name, new_module)

    def get_loss(self) -> float:
        return self.evaluator(self.model)

    def revert(self):
        for name, (parent_module, original_layer) in self.original_layers.items():
            submodule_name = name.split('.')[-1]
            print(f"Restoring original weights for layer: {name} by resetting {submodule_name} to {original_layer}")
            setattr(parent_module, submodule_name, original_layer)
        self.original_layers.clear()
        self.layers_checked_for_modification.clear()
        self.modifications.clear()


#  Following three functions Borrowed from https://github.com/cognitivecomputations/laserRMT/blob/main/laserRMT.py
def get_rank(S: Tensor, n: int, m: int) -> int:
    estimated_sigma = estimate_sigma_with_full_iqr(S)
    mp_threshold = marchenko_pastur_threshold(estimated_sigma, n, m)
    k = (S > mp_threshold).sum().item()
    return k


def estimate_sigma_with_full_iqr(S):
    q75 = torch.quantile(S, 0.75)
    q25 = torch.quantile(S, 0.25)
    iqr = q75 - q25
    sigma_estimated = iqr / 1.349  # 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
    return sigma_estimated


def marchenko_pastur_threshold(sigma, n, m):
    beta = n / m if n < m else m / n
    threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
    return threshold


def scan_layers_and_report(model: PreTrainedModel, tokenizer, dataset_name: str,
                           split: str, max_length: int,
                           layer_type: str, layer_number: int,
                           rank_override: int = None,
                           num_samples: int = 16, seed: int = 0):
    reducer = PrismaticLaserReducer.for_model(
        model=model, tokenizer=tokenizer, dataset_name=dataset_name,
        split=split, max_length=max_length, num_samples=num_samples,
        seed=seed)
    reducer.reduce_layers_matching(layer_type=layer_type,
                                   layer_number=layer_number,
                                   rank_override=rank_override)
    loss = reducer.get_loss()
    print(f"Original loss:        {reducer.base_evaluation}")
    print(f"Loss after reduction: {loss}")
    return reducer
