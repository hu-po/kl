"""
Shulman approximation of KL divergence
http://joschu.net/blog/kl-approx.html    
"""
import torch.distributions as dis

# create two normal distributions
p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)

# sample from q
x = q.sample(sample_shape=(10_000_000,))

# calculate true KL divergence
truekl = dis.kl_divergence(p, q)
print("true", truekl)

# calculate KL divergence using Shulman approximation
logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr
for k in (k1, k2, k3):
    print((k.mean() - truekl) / truekl, k.std() / truekl)