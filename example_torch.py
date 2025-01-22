"""

https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#kl_div

>>> kl_loss = nn.KLDivLoss(reduction="batchmean")
>>> # input should be a distribution in the log space
>>> input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
>>> # Sample a batch of distributions. Usually this would come from the dataset
>>> target = F.softmax(torch.rand(3, 5), dim=1)
>>> output = kl_loss(input, target)

"""

import torch

def kl_divergence(p, q):
    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)
    return torch.sum(p * torch.log(p / q))

# Define the distributions
p = [9/25, 12/25, 4/25]  # Distribution P
q = [1/3, 1/3, 1/3]      # Distribution Q

# Calculate KL divergences
kl_pq = kl_divergence(p, q)  # D_KL(P || Q)
kl_qp = kl_divergence(q, p)  # D_KL(Q || P)

print(f"D_KL(P || Q) = {kl_pq.item():.6f}")
print(f"D_KL(Q || P) = {kl_qp.item():.6f}")
