"""
Step 1: Confirm the container can actually use the GB10 GPU.

Run this first. If it fails, nothing else will work, and you have saved
yourself a lot of confusing debugging.
"""

import torch

print("PyTorch version :", torch.__version__)
print("CUDA version    :", torch.version.cuda)
print("GPU visible     :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "\nFAILED. PyTorch cannot see the GPU.\n"
        "Most likely cause: you forgot '--gpus all' on the docker run command,\n"
        "or the NVIDIA Container Toolkit is not installed on the host.\n"
    )

print("GPU name        :", torch.cuda.get_device_name(0))
print("GPU memory (GB) :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

# Do a tiny bit of real GPU maths, to prove kernels actually execute.
# On Blackwell (sm_121) some libraries build fine but crash the first time
# they run a real kernel, so a successful import is not sufficient proof.
x = torch.randn(1000, 1000, device="cuda")
y = (x @ x).sum().item()
print("GPU compute test: OK (result =", round(y, 2), ")")

print("\nAll good. Move on to step 2.")