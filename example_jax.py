"""

https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.special.rel_entr.html#jax.scipy.special.rel_entr

https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L616-L645

"""

import jax.numpy as jnp

def kl_divergence(p, q):
    p = jnp.array(p, dtype=jnp.float32)
    q = jnp.array(q, dtype=jnp.float32)
    return jnp.sum(p * jnp.log(p / q))

# Define the distributions
p = [9/25, 12/25, 4/25]  # Distribution P
q = [1/3, 1/3, 1/3]      # Distribution Q

# Calculate KL divergences
kl_pq = kl_divergence(p, q)  # D_KL(P || Q)
kl_qp = kl_divergence(q, p)  # D_KL(Q || P)

print(f"D_KL(P || Q) = {kl_pq:.6f}")
print(f"D_KL(Q || P) = {kl_qp:.6f}")
