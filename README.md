# kl

playing around with relative entropy (KL divergence) part of this [YouTube Livestream](https://youtube.com/live/LuF4NGezcxo)


install cpu versions only

```bash
conda create -n kl python=3.10
conda activate kl   
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
pip install jax[jaxlib]
pip install scipy
pip install tinygrad
```

run examples

```bash
python example_jax.py
```
