import platform
import sys

import bitsandbytes as bnb
import datasets
import peft
import torch
import transformers
import trl


def report(name, value):
    print(f"{name}: {value}")


report("executable", sys.executable)
report("python", sys.version.split()[0])
report("platform", platform.platform())
report("torch", torch.__version__)
report("torch_cuda", torch.version.cuda)
report("cudnn", torch.backends.cudnn.version())
report("transformers", transformers.__version__)
report("datasets", datasets.__version__)
report("peft", peft.__version__)
report("trl", trl.__version__)
report("bitsandbytes", bnb.__version__)
report("cuda_available", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，显卡相关测试全部停下")

properties = torch.cuda.get_device_properties(0)
report("device", properties.name)
report("capability", f"sm_{properties.major}{properties.minor}")
report("vram_gb", round(properties.total_memory / 1024**3, 1))

# bf16 矩阵乘法：4070S 是 sm_89，bf16 原生支持
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
report("bf16_matmul", tuple((x @ x).shape))
del x
torch.cuda.empty_cache()

# 4bit 线性层：QLoRA 走这条路径
layer = bnb.nn.Linear4bit(128, 128, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
probe = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
report("linear4bit_out", tuple(layer(probe).shape))
del layer, probe
torch.cuda.empty_cache()

# 200 步 bf16 训练：验证反向传播与优化器
torch.manual_seed(0)
features = torch.randn(512, 64, device="cuda")
target = features @ torch.randn(64, 8, device="cuda")
model = torch.nn.Linear(64, 8).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(200):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.mse_loss(model(features), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:
        report("train_loss_first", round(loss.item(), 6))
report("train_loss_last", round(loss.item(), 6))
report("vram_peak_gb", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
