import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)  # Raise errors if backward encounters NaN/Inf

import jax
import jax.numpy as jnp
from jax import grad

import numpy as np
from tinygrad.tensor import Tensor
import os
os.environ['TINYGRAD_DEVICE'] = 'cpu'


def run_pytorch_tests():
    print("--- PyTorch Tests ---")

    # FP32
    try:
        logits_fp32 = torch.tensor([50000.0, -50000.0], dtype=torch.float32, requires_grad=True)
        target_fp32 = torch.tensor([0.5, 0.5], dtype=torch.float32)

        log_probs = F.log_softmax(logits_fp32, dim=0)
        loss = F.kl_div(log_probs, target_fp32, reduction='sum')
        print("PyTorch FP32: Loss value:", loss.item())
        loss.backward()
        print("PyTorch FP32: Gradient on logits:", logits_fp32.grad)
    except Exception as e:
        print("PyTorch FP32 error:", e)

    # FP16
    try:
        # Re-create in case the previous block modifies state
        logits_fp16 = torch.tensor([50000.0, -50000.0], dtype=torch.float16, requires_grad=True)
        target_fp16 = torch.tensor([0.5, 0.5], dtype=torch.float16)

        log_probs = F.log_softmax(logits_fp16, dim=0)
        loss = F.kl_div(log_probs, target_fp16, reduction='sum')
        print("PyTorch FP16: Loss value:", loss.item())
        loss.backward()
        print("PyTorch FP16: Gradient on logits:", logits_fp16.grad)
    except Exception as e:
        print("PyTorch FP16 error:", e)


def run_jax_tests():
    print("\n--- JAX Tests ---")
    jax.config.update("jax_debug_nans", True)  # Raise error on NaNs/Infs in JAX ops

    def kl_div_jax(logits, target):
        # sum(target * (log(target) - log_probs))
        log_probs = jax.nn.log_softmax(logits, axis=0)
        return jnp.sum(target * (jnp.log(target) - log_probs))

    # FP32
    try:
        logits_fp32 = jnp.array([50000.0, -50000.0], dtype=jnp.float32)
        target_fp32 = jnp.array([0.5, 0.5], dtype=jnp.float32)

        def loss_fn_fp32(x):
            return kl_div_jax(x, target_fp32)

        loss_val = loss_fn_fp32(logits_fp32)
        print("JAX FP32: Loss:", loss_val)
        grads = grad(loss_fn_fp32)(logits_fp32)
        print("JAX FP32: Grad:", grads)
    except Exception as e:
        print("JAX FP32 error:", e)

    # FP16
    try:
        logits_fp16 = jnp.array([50000.0, -50000.0], dtype=jnp.float16)
        target_fp16 = jnp.array([0.5, 0.5], dtype=jnp.float16)

        def loss_fn_fp16(x):
            return kl_div_jax(x, target_fp16)

        loss_val = loss_fn_fp16(logits_fp16)
        print("JAX FP16: Loss:", loss_val)
        grads = grad(loss_fn_fp16)(logits_fp16)
        print("JAX FP16: Grad:", grads)
    except Exception as e:
        print("JAX FP16 error:", e)


def run_tinygrad_tests():
    print("\n--- TinyGrad Tests ---")

    def kl_div_tinygrad(logits: Tensor, target: Tensor):
        # KL = sum( p * [log(p) - log(q)] ), here p = target
        log_probs = logits.log_softmax()
        return (target * (target.log() - log_probs)).sum()

    # FP32
    try:
        logits_fp32 = Tensor(np.array([50000.0, -50000.0], dtype=np.float32), requires_grad=True)
        target_fp32 = Tensor(np.array([0.5, 0.5], dtype=np.float32))

        loss = kl_div_tinygrad(logits_fp32, target_fp32)
        print("TinyGrad FP32: Loss:", loss.numpy())
        loss.backward()
        print("TinyGrad FP32: Grad:", logits_fp32.grad.numpy())
    except Exception as e:
        print("TinyGrad FP32 error:", e)

    # FP16
    try:
        logits_fp16 = Tensor(np.array([50000.0, -50000.0], dtype=np.float16), requires_grad=True)
        target_fp16 = Tensor(np.array([0.5, 0.5], dtype=np.float16))

        loss = kl_div_tinygrad(logits_fp16, target_fp16)
        print("TinyGrad FP16: Loss:", loss.numpy())
        loss.backward()
        print("TinyGrad FP16: Grad:", logits_fp16.grad.numpy())
    except Exception as e:
        print("TinyGrad FP16 error:", e)


if __name__ == "__main__":
    run_pytorch_tests()
    run_jax_tests()
    # run_tinygrad_tests()
