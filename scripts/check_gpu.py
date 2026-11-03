"""GPU check (run once after installing torch; see README.md, Environment): CUDA build, sm_120 support,
and that a 4096x4096 fp32 matmul on the GPU (TF32 off) matches the CPU result.

Usage: python scripts/check_gpu.py"""

import json
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

info = {
    "torch_version": torch.__version__,
    "torch_version_cuda": torch.version.cuda,
    "cuda_is_available": torch.cuda.is_available(),
}
if not info["cuda_is_available"]:
    print(json.dumps(info, indent=2))
    sys.exit("FAIL: torch.cuda.is_available() is False")

info["device_name"] = torch.cuda.get_device_name(0)
info["capability"] = list(torch.cuda.get_device_capability(0))
info["arch_list"] = torch.cuda.get_arch_list()
info["sm_120_in_arch_list"] = "sm_120" in info["arch_list"]

gen = torch.Generator().manual_seed(0)
a = torch.randn(4096, 4096, generator=gen, dtype=torch.float32)
b = torch.randn(4096, 4096, generator=gen, dtype=torch.float32)
cpu = a @ b
gpu = (a.cuda() @ b.cuda()).cpu()
ref = (a.double() @ b.double())
info["matmul_max_abs_diff_gpu_vs_cpu"] = float((gpu - cpu).abs().max())
info["matmul_max_abs_err_gpu_vs_fp64"] = float((gpu.double() - ref).abs().max())
info["matmul_max_abs_err_cpu_vs_fp64"] = float((cpu.double() - ref).abs().max())
# Entries are sums of 4096 products of N(0,1) values (magnitude ~64); fp32 accumulation-order
# differences of ~1e-3 are expected. TF32 would give errors ~1e-1, so this separates the two.
info["matmul_allclose"] = bool(torch.allclose(gpu, cpu, rtol=1e-4, atol=1e-2))

print(json.dumps(info, indent=2))
ok = info["capability"] == [12, 0] and info["sm_120_in_arch_list"] and info["matmul_allclose"]
print("GPU VERIFICATION PASSED" if ok else "GPU VERIFICATION FAILED")
sys.exit(0 if ok else 1)
