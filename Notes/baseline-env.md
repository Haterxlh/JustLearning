### 宿舍电脑的环境

使用 conda 中的 pytorch-01 环境

tmp\test\check-env\check-env.py 运行结果：
```shell
(pytorch-01) PS D:\Desktop\JustLearning> python tmp\test\check-env\check-env.py
executable: D:\Environment\Anaconda\envs\pytorch-01\python.exe
python: 3.10.16
platform: Windows-10-10.0.22631-SP0
torch: 2.5.1+cu124
torch_cuda: 12.4
cudnn: 90100
transformers: 5.17.0
datasets: 5.0.1
peft: 0.21.0
trl: 1.14.0
bitsandbytes: 0.50.2
cuda_available: True
device: NVIDIA GeForce RTX 4070 SUPER
capability: sm_89
vram_gb: 12.0
bf16_matmul: (4096, 4096)
linear4bit_out: (8, 128)
train_loss_first: 63.020027
train_loss_last: 44.985588
vram_peak_gb: 0.07
```

### 工位电脑的环境

使用仓库内的 uv 虚拟环境 `.just-learning-venv`

`tmp\test\check-env\check-env.py` 运行结果（2026-09-28 21:07，完整输出见 `tmp\test\check-env\工位-check-res.txt`）：
```shell
PS D:\Desktop\JustLearning> .\.just-learning-venv\Scripts\python.exe tmp\test\check-env\check-env.py
executable: D:\Desktop\JustLearning\.just-learning-venv\Scripts\python.exe
python: 3.12.14
platform: Windows-11-10.0.26200-SP0
gpu: NVIDIA T400 4GB
driver: 538.33（最高 CUDA 12.2）
torch: 2.5.1+cu121
torch_cuda: 12.1
cudnn: 90100
transformers: 5.17.0
tokenizers: 0.23.2
cuda_available: True
device: NVIDIA T400 4GB
capability: sm_75
vram_gb: 4.0
float32_matmul_ms: 121.22
float32_tflops: 1.1
float16_matmul_ms: 885.48
float16_tflops: 0.2
bf16_matmul: 跳过：sm_75 无 bf16 硬件支持
autocast_bf16: False
train_loss_first: 73.422707
train_loss_last: 53.408237
peak_vram_gb: 0.2
disk_C: 剩余 176.4 GB / 共 474.9 GB
disk_D: 剩余 401.6 GB / 共 931.5 GB
llama_cli: D:\Environment\llama 下未找到
gguf_models: D:\Environment\llama\models 下未找到
```

> T400 上 fp16 比 fp32 慢 7 倍（4096 三次方矩阵乘法 885.48 ms 对 121.22 ms）：TU117 没有 tensor core，cuBLAS 的 fp16 路径在这张卡上反而吃亏。工位的代码走 fp32，不开 AMP fp16。
