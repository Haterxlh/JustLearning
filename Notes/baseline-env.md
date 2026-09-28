### 宿舍电脑的环境

使用 conda 中的 pytorch-01 环境

`test\check-env\check-env.py` 运行结果（2026-09-28 21:52，下面是关键行，完整输出见 `test\check-env\宿舍-check-res.txt`）：
```shell
(pytorch-01) PS D:\Desktop\JustLearning> python test\check-env\check-env.py
executable: D:\Environment\Anaconda\envs\pytorch-01\python.exe
python: 3.10.16
platform: Windows-10-10.0.22631-SP0
gpu: NVIDIA GeForce RTX 4070 SUPER
driver_cuda_max: 未解析到
disk_C: 剩余 31.4 GB / 共 199.2 GB
disk_D: 剩余 251.1 GB / 共 754.3 GB
llama_cli: D:\Environment\llama 下未找到
gguf_models: ['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf (3.72 GB)', 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf (0.64 GB)']
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
float16_matmul_ms: 1.84
float16_tflops: 74.7
bf16_matmul: (4096, 4096)
linear4bit_out: (8, 128)
autocast_bf16: True
train_loss_first: 63.020027
train_loss_last: 44.985588
peak_vram_gb: 0.2
```

- 驱动 610.74，nvidia-smi 表头写的是 `CUDA UMD Version: 13.3`；这一份是脚本改动之前跑的，所以 `driver_cuda_max` 报「未解析到」，正则已经改成 `CUDA (?:UMD )?Version`。
- 7B 两个分片都齐了：`-00001-of-00002.gguf` 3.72 GB（3,993,201,344 B）+ `-00002-of-00002.gguf` 0.64 GB（689,872,288 B）。
- `missing_packages: ['tokenizers']` 是脚本误报：宿舍机上 `python -c "import tokenizers; print(tokenizers.__version__)"` 输出 0.23.2。原因是原来的判断走 `packages_distributions()`，它依赖每个包元数据里记录的顶层模块名，conda 装的 tokenizers 没有这份记录；已改成按 import 名判断、版本仍从安装记录读。
- 睡眠超时交流电 0 秒（永不睡眠）、直流电 600 秒；nvidia-smi 进程表里有 `GameViewerServer.exe` 与 `GameViewer.exe`（UU远程 的组件），火绒的 `HipsDaemon.exe` 也在。
- C 盘只剩 31.4 GB，模型与数据集继续放 D 盘。

#### llama.cpp（2026-09-29）

程序本体：`D:\Environment\llama\llama-b10068`，版本 `b10068-571d0d540`，构建 `win-cuda-12.4-x64`。
CUDA 运行时库 `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` 与该目录下的可执行文件放在同一目录。
模型：`D:\Environment\LLMs\qwen2.5-7b-instruct-q4_k_m\`，两个分片合计 4.36 GiB，只指定第一个分片即可。
启动脚本：`D:\Environment\llama\start-qwen-server.cmd`（前台窗口）、`D:\Environment\llama\llama-server-daemon.cmd`（输出写入 `D:\Environment\llama\logs\server.log`）。手动双击启动，未注册计划任务。

`llama-bench.exe -m <第一个分片> -ngl 99` 结果：

```shell
| model                  |    size | params | backend | ngl |  test |             t/s |
| qwen2 7B Q4_K - Medium | 4.36 GiB | 7.62 B | CUDA    |  99 | pp512 | 5922.61 ± 74.46 |
| qwen2 7B Q4_K - Medium | 4.36 GiB | 7.62 B | CUDA    |  99 | tg128 |    96.24 ± 0.14 |
```

接口：`http://127.0.0.1:8080`，OpenAI 兼容路径 `/v1/chat/completions`，`model` 字段填 `qwen2.5-7b-instruct`，`api_key` 填任意字符串。`n_slots = 4`，单槽上下文 8192。
调用样例：`test\check-llama\check-llama.py`。

> Windows PowerShell 5.1 的 `Invoke-RestMethod` 直接传字符串 body 会把非 ASCII 字符发成 `?`，必须先用 `[System.Text.Encoding]::UTF8.GetBytes()` 转成字节数组再传。

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

> 同一个测法下两张卡的算力：fp32 是 24.1 对 1.1 TFLOPS（约 22 倍），fp16 是 74.7 对 0.2 TFLOPS（约 370 倍）。T400 是 TU117，没有 tensor core，fp16 比它自己的 fp32 还慢 7 倍，所以工位的代码走 fp32、不开 AMP fp16。
> 两台都还没装 llama.cpp：`D:\Environment\llama` 下没有 `llama-cli.exe`。
