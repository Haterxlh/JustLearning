### 宿舍电脑的环境

使用 conda 中的 pytorch-01 环境

`test\check-env\check-env.py` 运行结果（2026-09-28 21:29，下面是关键行，完整输出见 `test\check-env\宿舍-check-res.txt`）：
```shell
(pytorch-01) PS D:\Desktop\JustLearning> python test\check-env\check-env.py
executable: D:\Environment\Anaconda\envs\pytorch-01\python.exe
python: 3.10.16
platform: Windows-10-10.0.22631-SP0
gpu: NVIDIA GeForce RTX 4070 SUPER
driver_cuda_max: 未解析到
disk_C: 剩余 31.4 GB / 共 199.2 GB
disk_D: 剩余 252.3 GB / 共 754.3 GB
llama_cli: D:\Environment\llama 下未找到
gguf_models: ['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf (3.15 GB)']
sleep_timeout_ac_seconds: 0
missing_packages: ['tokenizers']
torch: 2.5.1+cu124
transformers: 5.17.0
datasets: 5.0.1
peft: 0.21.0
trl: 1.14.0
bitsandbytes: 0.50.2
torch_cuda: 12.4
cudnn: 90100
cuda_available: True
device: NVIDIA GeForce RTX 4070 SUPER
capability: sm_89
vram_gb: 12.0
float32_matmul_ms: 5.7
float32_tflops: 24.1
float16_matmul_ms: 1.96
float16_tflops: 70.2
bf16_matmul: (4096, 4096)
linear4bit_out: (8, 128)
autocast_bf16: True
train_loss_first: 63.020027
train_loss_last: 44.985588
peak_vram_gb: 0.2
```

- 驱动 610.74，nvidia-smi 表头写的是 `CUDA UMD Version: 13.3`；脚本的正则只认 `CUDA Version:`，所以那一行报「未解析到」，已改。
- 7B 模型只下了第一个分片，而且不完整：该分片应为 3,993,201,344 B（3.72 GiB），实测 3.15 GiB；`qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`（689,872,288 B）还没有。
- `missing_packages: ['tokenizers']`：transformers 5.17.0 在，tokenizers 没有被识别到，需要确认宿舍机是否真的没装（assignment1 的 BPE notebook 要用 `tokenizers.BpeTrainer`）。
- 睡眠超时交流电 0 秒（永不睡眠）、直流电 600 秒；nvidia-smi 进程表里有 `GameViewerServer.exe` 与 `GameViewer.exe`（UU远程 的组件），火绒的 `HipsDaemon.exe` 也在。

### 工位电脑的环境

使用仓库内的 uv 虚拟环境 `.just-learning-venv`

`test\check-env\check-env.py` 运行结果（2026-09-28 21:26，下面是关键行，完整输出见 `test\check-env\工位-check-res.txt`）：
```shell
PS D:\Desktop\JustLearning> .\.just-learning-venv\Scripts\python.exe test\check-env\check-env.py
executable: D:\Desktop\JustLearning\.just-learning-venv\Scripts\python.exe
python: 3.12.14
platform: Windows-11-10.0.26200-SP0
gpu: NVIDIA T400 4GB
driver_cuda_max: 12.2
disk_C: 剩余 176.3 GB / 共 474.9 GB
disk_D: 剩余 400.6 GB / 共 931.5 GB
llama_cli: D:\Environment\llama 下未找到
gguf_models: ['qwen2.5-1.5b-instruct-q4_k_m.gguf (1.04 GB)']
missing_packages: 无
torch: 2.5.1+cu121
transformers: 5.17.0
tokenizers: 0.23.2
torch_cuda: 12.1
cudnn: 90100
cuda_available: True
device: NVIDIA T400 4GB
capability: sm_75
vram_gb: 4.0
float32_matmul_ms: 120.56
float32_tflops: 1.1
float16_matmul_ms: 910.07
float16_tflops: 0.2
bf16_matmul: 跳过：sm_75 无 bf16 硬件支持
autocast_bf16: False
train_loss_first: 73.422707
train_loss_last: 53.408237
peak_vram_gb: 0.2
```

> 同一个测法下两张卡的算力：fp32 是 24.1 对 1.1 TFLOPS（约 22 倍），fp16 是 70.2 对 0.2 TFLOPS（约 350 倍）。T400 是 TU117，没有 tensor core，fp16 比它自己的 fp32 还慢 7 倍，所以工位的代码走 fp32、不开 AMP fp16。
> 两台都还没装 llama.cpp：`D:\Environment\llama` 下没有 `llama-cli.exe`。
