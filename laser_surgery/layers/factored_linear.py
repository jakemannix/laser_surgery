from torch.nn.modules import Module
from transformers.utils.logging import get_logger
import torch
from torch import nn, Tensor


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
        if len(x.shape) == 3:
            if x.shape[0] == 1:
                x = x.squeeze(0)
                must_unsqueeze = True
        # TODO: check carefully if this actually adds the bias in the right way, and transposes correctly
        x = x.to(self.device)
        # batch case, with nontrivial batches
        if len(x.shape) == 3 and x.shape[0] > 1:
            x_transposed = x.transpose(1, 2)
            Vt = self.Vt.unsqueeze(0)
            VtX = torch.bmm(Vt.expand(x.shape[0], -1, -1), x_transposed)
            S = torch.diag_embed(self.S).unsqueeze(0)
            SVtX = torch.bmm(S.expand(x.shape[0], -1, -1), VtX)
            U = self.U.unsqueeze(0)
            result = torch.bmm(U.expand(x.shape[0], -1, -1), SVtX)
            result = result + self.bias.view(1, 1, -1) if self.bias is not None else result
            result.transpose(1, 2)
        else:
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


# TODO: this assumes x is a single input, not a batch.  Let's compare with batches too.
def compare_factorization(linear: nn.Linear, rank: int, x: Tensor):
    factored = FactoredLinear.from_linear(linear, rank)
    reconstructed = factored.U @ torch.diag(factored.S) @ factored.Vt
    bias = linear.bias if linear.bias is not None else 0
    with torch.no_grad():
        factored_output = factored(x)
        reconstructed_output = reconstructed(x) + bias
        return torch.allclose(factored_output, reconstructed_output, atol=1e-7)


# TODO: pass in a list of input Tensors, so nothing gets cleverly cached by the GPU
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

