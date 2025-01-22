"""

https://github.com/tinygrad/tinygrad/blob/49b914ee691a9f2ecdc6f0f852c4a5f4fe40c03b/tinygrad/tensor.py

"""

#   def cross_entropy(self, Y:Tensor, reduction:ReductionStr="mean", label_smoothing:float=0.0) -> Tensor:
#     """
#     Compute the cross entropy loss between input logits and target.

#     NOTE: `self` are logits and `Y` are the target labels or class probabilities.

#     See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

#     ```python exec="true" source="above" session="tensor" result="python"
#     t = Tensor([[-1, 2, -3], [1, -2, 3]])
#     Y = Tensor([1, 2])
#     print(t.cross_entropy(Y).item())
#     ```
#     ```python exec="true" source="above" session="tensor" result="python"
#     t = Tensor([[-1, 2, -3], [1, -2, 3]])
#     Y = Tensor([1, 2])
#     print(t.cross_entropy(Y, reduction='none').numpy())
#     ```
#     """
#     assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
#     Y = Y.one_hot(num_classes=cast(int, self.shape[1])) if Y.ndim < 2 else Y
#     Y = (1 - label_smoothing)*Y + label_smoothing / cast(int, Y.shape[1])
#     ret = -self.log_softmax(axis=1).mul(Y).sum(axis=1)
#     return ret._do_reduction(reduction)

#   def nll_loss(self, Y:Tensor, weight:Optional[Tensor]=None, ignore_index:Optional[int]=None, reduction:ReductionStr="mean") -> Tensor:
#     """
#     Compute the negative log likelihood loss between log-probabilities and target labels.

#     NOTE: `self` is log-probabilities and `Y` is the Y labels or class probabilities.

#     See: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html

#     ```python exec="true" source="above" session="tensor" result="python"
#     t = Tensor([[-1, 2, -3], [1, -2, 3]])
#     Y = Tensor([1, 2])
#     print(t.log_softmax().nll_loss(Y).item())
#     ```
#     ```python exec="true" source="above" session="tensor" result="python"
#     t = Tensor([[-1, 2, -3], [1, -2, 3]])
#     Y = Tensor([1, 2])
#     print(t.log_softmax().nll_loss(Y, reduction='none').numpy())
#     ```
#     """
#     weight = Tensor.ones_like(Y, requires_grad=False) if weight is None else weight[Y]
#     masked_weight = weight if ignore_index is None else weight * (Y != ignore_index)
#     nll = -self.gather(1, Y.unsqueeze(1)).squeeze(1) * masked_weight
#     return nll.sum() / masked_weight.sum() if reduction == "mean" else nll._do_reduction(reduction)

from tinygrad import Tensor


def kl_loss(p, q):
    return p.log_softmax().mul(q).sum(axis=1)

p = Tensor([9/25, 12/25, 4/25])
q = Tensor([1/3, 1/3, 1/3])

kl_loss(p, q)