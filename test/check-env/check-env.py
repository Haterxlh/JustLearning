import importlib
import importlib.metadata
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROMPT = "用一句话解释 KV Cache 为什么能加速解码。"
LLAMA_DIR = Path("D:/Environment/llama")
MODEL_DIR = LLAMA_DIR / "models"
PACKAGES_DORM = ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "tokenizers")
PACKAGES_OFFICE = ("torch", "transformers", "tokenizers")


def report(name, value):
    print(f"{name}: {value}")


def run(command, encoding="gbk"):
    result = subprocess.run(command, capture_output=True, text=True, encoding=encoding, errors="replace")
    print(f"$ {' '.join(str(part) for part in command)}")
    print(result.stdout.rstrip())
    if result.stderr.rstrip():
        print(result.stderr.rstrip())
    report("returncode", result.returncode)
    return result


print("=== 机器层面 ===")
report("executable", sys.executable)
report("python", sys.version.split()[0])
report("platform", platform.platform())
report("cwd", Path.cwd())

gpu = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"])
gpu_name = gpu.stdout.strip().splitlines()[0].split(",")[0].strip()
report("gpu", gpu_name)
branch = "office" if "T400" in gpu_name else "dorm"
report("branch", branch)

driver = run(["nvidia-smi"])
match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", driver.stdout)
report("driver_cuda_max", match.group(1) if match else "未解析到")

for drive in ("C:/", "D:/"):
    if not Path(drive).exists():
        continue
    usage = shutil.disk_usage(drive)
    report(f"disk_{drive[0]}", f"剩余 {usage.free / 1024**3:.1f} GB / 共 {usage.total / 1024**3:.1f} GB")

llama_cli = shutil.which("llama-cli")
if llama_cli is None and LLAMA_DIR.exists():
    found = sorted(LLAMA_DIR.glob("**/llama-cli.exe"))
    if found:
        llama_cli = str(found[0])
report("llama_cli", llama_cli or f"{LLAMA_DIR} 下未找到")
if llama_cli:
    run([llama_cli, "--version"])

models = sorted(MODEL_DIR.glob("**/*.gguf")) if MODEL_DIR.exists() else []
report("gguf_models", [f"{path.name} ({path.stat().st_size / 1024**3:.2f} GB)" for path in models] or f"{MODEL_DIR} 下未找到")

if llama_cli and models:
    model = models[0]
    inference = run(
        [llama_cli, "-m", str(model), "-p", PROMPT, "-n", "128", "-ngl", "99", "-c", "2048", "--seed", "42", "-no-cnv"],
        encoding="utf-8",
    )
    text = inference.stdout + "\n" + inference.stderr
    prompt_eval = re.search(r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second", text)
    generate = re.search(r"(?<!prompt )eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs.*?([\d.]+) tokens per second", text)
    report("prompt_eval", f"{prompt_eval.group(1)} tokens, {prompt_eval.group(2)} tok/s" if prompt_eval else "未解析到")
    report("generate", f"{generate.group(1)} tokens, {generate.group(2)} tok/s" if generate else "未解析到")

if branch == "dorm":
    sleep = run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"])
    match = re.search(r"(?:当前|Current)[^\n]*?0x([0-9a-fA-F]+)", sleep.stdout)
    if match:
        report("sleep_timeout_ac_seconds", int(match.group(1), 16))
    else:
        report("sleep_timeout_ac_seconds", "未解析到")

print("=== Python 层面 ===")
required = PACKAGES_DORM if branch == "dorm" else PACKAGES_OFFICE
missing = [name for name in required if importlib.util.find_spec(name) is None]
report("missing_packages", missing or "无")
for name in required:
    if name not in missing:
        report(name, importlib.metadata.version(name))
if "torch" in missing:
    raise SystemExit("torch 未装好，本次只报告机器层面")

torch = importlib.import_module("torch")
report("torch_cuda", torch.version.cuda)
report("cudnn", torch.backends.cudnn.version())
report("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，显卡相关测试全部停下")

properties = torch.cuda.get_device_properties(0)
report("device", properties.name)
report("capability", f"sm_{properties.major}{properties.minor}")
report("vram_gb", round(properties.total_memory / 1024**3, 1))

# 4096 三次方矩阵乘法：两张卡用同一个测法，算力可直接比
for dtype in (torch.float32, torch.float16):
    left = torch.randn(4096, 4096, dtype=dtype, device="cuda")
    right = torch.randn(4096, 4096, dtype=dtype, device="cuda")
    for _ in range(3):
        left @ right
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(20):
        left @ right
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    label = str(dtype).split(".")[-1]
    report(f"{label}_matmul_ms", round(elapsed / 20 * 1000, 2))
    report(f"{label}_tflops", round(20 * 2 * 4096**3 / elapsed / 1e12, 1))
    del left, right
    torch.cuda.empty_cache()

if properties.major >= 8:
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    report("bf16_matmul", tuple((x @ x).shape))
    del x
    torch.cuda.empty_cache()
    if branch == "dorm":
        bnb = importlib.import_module("bitsandbytes")
        layer = bnb.nn.Linear4bit(128, 128, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
        probe = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        report("linear4bit_out", tuple(layer(probe).shape))
        del layer, probe
        torch.cuda.empty_cache()
else:
    report("bf16_matmul", f"跳过：sm_{properties.major}{properties.minor} 无 bf16 硬件支持")

# 200 步小模型训练：验证反向传播与优化器。sm_75 上 fp16 比 fp32 慢，所以只给 sm_80+ 开 bf16
torch.manual_seed(0)
features = torch.randn(512, 64, device="cuda")
target = features @ torch.randn(64, 8, device="cuda")
model = torch.nn.Linear(64, 8).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
enabled = properties.major >= 8
report("autocast_bf16", enabled)
for step in range(200):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        loss = torch.nn.functional.mse_loss(model(features), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:
        report("train_loss_first", round(loss.item(), 6))
report("train_loss_last", round(loss.item(), 6))
report("peak_vram_gb", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
