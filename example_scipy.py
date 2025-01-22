"""

https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html


"""

import scipy.special as sp

def kl_divergence(p, q):
    return sp.kl_div(p, q)

p = [9/25, 12/25, 4/25]  # Distribution P
q = [1/3, 1/3, 1/3]      # Distribution Q

kl_pq = kl_divergence(p, q)  # D_KL(P || Q)
kl_qp = kl_divergence(q, p)  # D_KL(Q || P)

print(f"D_KL(P || Q) = {kl_pq:.6f}")
print(f"D_KL(Q || P) = {kl_qp:.6f}")
