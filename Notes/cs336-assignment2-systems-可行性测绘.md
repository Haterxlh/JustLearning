# CS336 作业2（assignment2-systems）测绘报告 —— 兼「RTX 4070 Super 12GB 单卡可行性」

> 数据来源：datawhalechina/diy-llm `coursework/assignment2-systems/` 的 raw 文件 + 官方 stanford-cs336/assignment2-systems README。
> 全文严格区分【原文抓取】与【推断/估计】。抓不到的一律写"抓取失败"，不补写。

---

## 0. 抓取清单与状态

| 目标 | 状态 |
|---|---|
| README.md | ✅ 完整 |
| 相关文档/【导读】…md | ✅ 完整 |
| tests/test_attention.py | ✅ 完整 |
| tests/test_ddp.py | ✅ 完整 |
| tests/test_ddp_individual_parameters.py | ✅ 完整 |
| tests/test_sharded_optimizer.py | ✅ 完整 |
| tests/adapters.py | ✅ 完整 |
| tests/conftest.py | ✅ 完整 |
| tests/common.py（额外抓，关键） | ✅ 完整 |
| test_and_make_submission.sh | ✅ 完整 |
| pyproject.toml / main.py | ✅ 完整 |
| requirements.txt | ✅ 抓到（含 triton 行；末尾 2359 字节被截断） |
| cs336_systems/*.py 六个实现文件 | ✅ 完整 |
| cs336_systems/作业2.ipynb | ⚠️ 部分（末尾 69156 字节被截断） |
| cs336_systems/作业1.ipynb | ⚠️ 部分（末尾 64211 字节被截断） |
| 官方 stanford-cs336.github.io/spring2025-assignment2-systems/ | ❌ 抓取失败 |
| 官方 PDF cs336_assignment2_systems.pdf | ❌ 抓取失败 |
| 相关文档/CHANGELOG.md（额外抓，关键） | ✅ 完整 |

---

## 1.【原文抓取】作业结构与题目清单（含分值）

以下题目名/分值来自两个 notebook 的 markdown 单元格与 `相关文档/CHANGELOG.md`，均为逐字抓取。

### 1.1 作业1.ipynb（覆盖"基准测试 / 混合精度 / 注意力"三块）

- `# 作业一`：端到端 benchmark 脚本
  - (a) 初始化模型 → 生成随机 batch → w 个 warmup → 计时 n 步 → 每步 `torch.cuda.synchronize()`
  - **交付物**：「一个脚本，能够根据给定超参数初始化基础 Transformer 模型，创建随机数据批次，并对前向与反向传播进行计时。」
  - (b) 对 §1.2 模型规模测前向/反向，5 个 warmup、10 次测量取均值与标准差 → 交付物：「1–2 句话给出你的计时结果。」
  - (c) 不做 warmup 重跑，再试 1–2 个 warmup
- `# 作业二` → `问题（混合精度累积）：1 分`（4 段 fp32/fp16 累加代码，交付物：2–3 句话）
- `# 作业三` → `问题（benchmarking_mixed_precision）：2 分`
  - (a) 给出 `ToyModel(fc1/LayerNorm/fc2/relu)`，问各组件在 autocast 下的 dtype
  - (b) LayerNorm 哪些部分对混合精度敏感
  - (c) 加 BF16 选项，对 §1.1.2 每种模型规模计时
- `# 作业四` → `Problem（memory profiling）：4 分`
  - 表 1（原文）：`small 768/3072/12/12`、`medium 1024/4096/24/16`、`large 1280/5120/36/20`、`xl 1600/6400/48/25`、`2.7B 2560/10240/32/32`
  - 「对 **表 1 中的 2.7B 模型**进行 profiling……上下文长度分别为 **128、256 和 512**」
  - (a) 两张 `pytorch.org/memory_viz` 的 Active Memory Timeline 截图（仅前向 / 完整训练步），2–3 句说明
  - (b) 一张表格，每个 context length 两个数值
  - (c) mixed-precision 下峰值内存，2–3 句
  - (d) 2.7B 残差流激活张量大小（MB），1–2 句
  - (e) 最大分配及调用栈，1–2 句
- `# 作业 五` → `Problem (pytorch_attention): 2 points`
  - batch=8、单头；d_model ∈ {16,32,64,128} × seqlen ∈ {256,1024,4096,8192,16384}；前向计时 100 次；测反向开始前显存并反向计时 100 次；要 warmup + synchronize
  - 「请报告你在这些配置下得到的 **时间结果（或 out-of-memory 错误）**。**在什么规模下会出现 out-of-memory？**」
  - **Deliverable**：时间表 + 内存推导 + 1–2 段文字分析
  - 同节含 `# Flash Attention`：「我们将实现一个遵循 **FlashAttention-2** 论文的注意力 kernel，它通过 **按 tile 方式计算注意力**，避免显式地物化 seqlen×seqlen 的注意力分数矩阵」
- `# 作业六` → `Problem (torch_compile): 2 points`
  - (a) 编译版 attention 与未编译版对比，Deliverable：一张表格（前向+反向耗时）
  - (b) 编译整个 Transformer，Deliverable：一张表格（vanilla vs compiled）

### 1.2 作业2.ipynb（覆盖"分布式数据并行"）

- `# 作业一` → `## 问题（distributed_communication_single_node）：5 分`
  - 后端+设备：`Gloo + CPU` / `NCCL + GPU`；dtype `float32`；张量 1MB / 10MB / 100MB / 1GB；进程数 2 / 4 / 6
  - 「**资源限制**：最多可使用 **6 张 GPU**；每一次基准测试运行时间 **不得超过 5 分钟**」
  - 交付内容：「提供 **图表（plot）和/或表格（table）**……并附上 **2–3 句话的分析说明**」
- `# 作业二` → `## 问题（naive_ddp）：5 分`
  - 反向传播后对各参数梯度单独 all-reduce；用随机数据训练小玩具模型，验证权重与单进程一致
- `## 问题（naive_ddp_benchmarking）：3 分`
  - 「收集在单节点设置（**1个节点 x 2个GPU**）下对XL模型的测量结果」；交付物：基准设置 + 每迭代时间 + 梯度通信时间
- `# 问题（minimal_ddp_flat_benchmarking）：2分`
  - 「（**1个节点 x 2个GPU**，XL模型大小）」把梯度扁平化成一个张量，与逐参数 all-reduce 对比
  - 交付物：「在单次批量all-reduce调用下……每个训练迭代的测量时间以及梯度通信所花费的时间。1-2句话比较批量与单独通信梯度的结果。」
- `# 作业三` → `## 问题 (ddp_overlap_individual_parameters): 5 分`
  - 建议公共接口（原文）：
    - `def __init__(self, module: torch.nn.Module):`
    - `def forward(self, *inputs, **kwargs):`
    - `def finish_gradient_synchronization(self):`
  - 「然后，执行测试，通过运行 `pytest tests/test_ddp_individual_parameters.py`。我们建议多次运行测试（例如，5 次）以确保其可靠通过。」
- `# 作业四` → `### 问题 (ddp_bucketed_benchmarking)：3分`
  - (a) 「使用与之前实验相同的配置（**1个节点，2个GPU，XL模型大小**）……变化最大桶大小（**1, 10, 100, 1000 MB**）」
  - 交付物：「各种桶大小下每次训练迭代的测量时间。对结果、你的预期以及任何不匹配的可能原因进行3-4句评论。」
  - (b) 给定 s / w / o / n_b，写 DDP 开销方程 + 最优桶大小方程

### 1.3 只在 CHANGELOG 出现、两个 notebook 未覆盖的题目（原文）

`相关文档/CHANGELOG.md` 逐字引用：

- 「讲义：将问题 `distributed_communication_multi_node` 明确为 2x1、2x2 和 2x3」
- 「讲义：澄清 `optimizer_state_sharding_accounting` (a) 中的分析说明」
- 「讲义：`memory_profiling` (b) 应当对上下文长度进行扫描，而不是模型大小」
- 「讲义：`ddp_bucketed_benchmarking` (b) 中的假设条件」
- 「讲义：修复 bucketed DDP 测试命令中的拼写错误，应为 `pytest tests/test_ddp.py`」
- 「讲义：修复 `ddp_overlap_individual_parameters_benchmarking` (a) 的交付要求，不再要求通信时间，只需端到端 step 时间」
- 「讲义：补充排行榜提交的相关细节」（v1.0.4）
- 「代码：为 flash forward 实现中的 logsumexp 添加测试」（v1.0.1）
- 「代码：测试 forward 和 backward 中的 `causal=True`」（v1.0.1）

**结论（事实）**：`distributed_communication_multi_node`、`optimizer_state_sharding_accounting`、`flash_attention` / FlashAttention Triton 这几个题目在 datawhale 两个 notebook 里**没有讲解单元格**，只在 CHANGELOG / 独立 .py / tests 中出现。

---

## 2.【原文抓取】要写哪些模块/函数（到函数名级别）

### 2.1 唯一被官方 pytest 检验的接入点：`tests/adapters.py`（8 个函数，全部 `raise NotImplementedError`）

| 函数签名（原文） | 作用 |
|---|---|
| `get_flashattention_autograd_function_pytorch() -> Type` | 返回 FlashAttention2 的 `torch.autograd.Function` 子类，**只用标准 PyTorch，不用 Triton** |
| `get_flashattention_autograd_function_triton() -> Type` | 同上，但前向与反向都调用自定义 Triton kernel |
| `get_ddp_individual_parameters(module) -> torch.nn.Module` | 逐参数异步通信、与反向重叠 |
| `ddp_individual_parameters_on_after_backward(ddp_model, optimizer)` | 反向之后、optimizer.step 之前执行 |
| `get_ddp_bucketed(module, bucket_size_mb) -> torch.nn.Module` | 按桶异步通信，与反向重叠 |
| `ddp_bucketed_on_after_backward(ddp_model, optimizer)` | 反向之后、step 之前 |
| `ddp_bucketed_on_train_batch_start(ddp_model, optimizer)` | **训练步最开始**执行 |
| `get_sharded_optimizer(params, optimizer_cls, **kwargs) -> torch.optim.Optimizer` | 优化器状态分片 |

### 2.2 仓库已给出的"参考实现文件"里的类/函数（这些**不是** adapters 的返回值）

**`cs336_systems/flash_attention.py`**
- `class FlashAttentionAutograd(torch.autograd.Function)`
  - `forward(ctx, Q, K, V)` ← **注意：没有 `is_causal` 参数**；内部 `Bq=Bk=64`，保存 `ctx.save_for_backward(Q, K, V, O, L)`，`ctx.scale/Bq/Bk`
  - `backward(ctx, dO)` → 返回 `dQ, dK, dV`

**`cs336_systems/flash_attention_triton.py`**
- `@triton.jit def flash_attention_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, stride_*, Nq, Nk, scale, D: tl.constexpr, Q_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr, is_causal: tl.constexpr)`
- `class FlashAttentionTriton(torch.autograd.Function)`
  - `forward(ctx, Q, K, V, is_causal=False)`；`ctx.save_for_backward(Q, K, V, L)`
  - **没有 `backward`**（文件到 forward 结束）

**`cs336_systems/sharded_optimizer.py`**
- `class ShardedOptimizer(Optimizer)`
  - `__init__(self, params, optimizer_cls, **kwargs)`（内部 `dist.is_initialized()` 检查、`self.rank/world_size/optimizer_cls/optimizer_kwargs`，`super().__init__(params, defaults={})`，`_build_local_optimizer()`）
  - `_build_local_optimizer(self)`
  - `_param_owner(self, param) -> int`（`index % world_size`）
  - `_global_param_index(self, param) -> int`
  - `step(self, closure=None, **kwargs)`（`@torch.no_grad()`）
  - `_sync_parameters(self)`（逐参数 `dist.broadcast(p.data, src=owner)`）
  - `add_param_group(self, param_group)`
- 注意：`_sync_parameters` 用 `self.param_groups`，而 `super().__init__(params, defaults={})` 不会设置 `lr` 等 → 推断会报错（见 §8 观察项）

**`cs336_systems/ddp_bucket.py`**
- `class TeachingDDPOverlapPerParam(torch.nn.Module)`：`__init__(self, model)`、`_create_grad_hook(self)`、`forward(*args, **kwargs)`、`finalize_gradients(self)`
  - 用 `param.register_post_accumulate_grad_hook(...)`；初始 broadcast 用 `self.model`
- `class GradientBucket`：`__init__(self, params, bucket_id)`、`reset(self)`
- `class TeachingDDPBucketed(torch.nn.Module)`：`__init__(self, model, bucket_size_mb)`、`_create_bucket(self, params)`、`_create_bucket_hook(self, param)`、`forward(*args, **kwargs)`、`finalize_gradients(self)`

**`cs336_systems/ddp_overlap_individual_parameters.py`**
- `class DDPOverlapBucketed(nn.Module)`：`__init__(self, module, bucket_size_mb)`、`_broadcast_parameters(self)`、`_create_bucket(self)`、`_register_hook(self)`、`_create_hook(self, grad, param, bucket_idx)`、`forward(self, x)`、`finish_gradient_synchronization(self)`
- `def run()`（示例：`dist.init_process_group(backend="nccl")`、`nn.Linear(10,5).cuda()`）

**`cs336_systems/my_profile.py`**
- `profile(description, run, num_warmups=1, with_stack=False)` → `torch.profiler.profile(activities=[CPU, CUDA], ...)`
- `mean(values)`
- `benchmark(description, run, num_warmups=1, num_trials=3)`

### 2.3 两个 notebook 实际写出的文件（`%%writefile`）

- 作业2.ipynb：`distributed_demo.py`、`distributed_communication_single_node_demo.py`、`ddp_model_demo.py`、`minimal_ddp_flat_benchmarking.py`、`ddp_overlap_individual_parameters.py`、`ddp_bucketed_benchmark_complete.py`
- 作业1.ipynb：**在已抓取部分中没有任何 `%%writefile`**

⚠️ 因此 README 的「其他文件都是这两个文件生成的」与事实不符：`flash_attention.py`、`flash_attention_triton.py`、`sharded_optimizer.py`、`ddp_bucket.py` **不是** notebook 生成的。（作业1.ipynb 末尾被截断，此结论以已抓取部分为限。）

---

## 3.【原文抓取】交付物汇总

| 类型 | 原文证据 |
|---|---|
| **代码脚本** | `distributed_communication_single_node`（5分）、`naive_ddp`（5分）、`minimal_ddp_flat_benchmarking`（2分）、`ddp_overlap_individual_parameters`（5分）、`ddp_bucketed`（3分）、作业一 benchmark 脚本 |
| **表格** | `memory_profiling (b)`「一张表格，每个 context length 对应两个数值」；`pytorch_attention`「一张包含时间测量结果的表格」；`torch_compile` 两张对比表 |
| **图/截图** | `distributed_communication_single_node`「提供图表（plot）和/或表格」；`memory_profiling (a)`「两张截图（Active Memory Timeline）」 |
| **文字分析** | 大量「2–3 句话」「3-4句评论」「1–2段文字分析」 |
| **公式推导** | `ddp_bucketed_benchmarking (b)`「建模DDP开销的方程，以及最优桶大小的方程」；`memory_profiling (d)`「给出你的推导过程」 |
| **CI 提交 / tar.gz** | `test_and_make_submission.sh` 生成的 zip（见 §4） |
| **排行榜分数** | **仓库 frågan 抓取内容中没有任何 `compute_leaderboard_score` / `modal` 字样**；只有 `导读` 与 `CHANGELOG` 提到「排行榜提交」 |

`导读` 原文：「FlashAttention-2的性能优于标准的Attention实现，可以在Assignment 2的leaderboard上进行比较。」
`CHANGELOG` 原文：「讲义：补充排行榜提交的相关细节」。
→ 可确认**官方讲义里有 leaderboard 环节**，但**提交方式/计分脚本在 datawhale 仓库中不存在**（抓取失败项，见 §9）。

---

## 4.【原文抓取】官方测试与评分方式

### 4.1 `test_and_make_submission.sh` 全文（逐字）

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

# Set the name of the output tar.gz file
output_file="cs336-spring2024-assignment-2-submission.zip"
rm "$output_file" || true

# Compress all files in the current directory into a single zip file
zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth'

echo "All files have been compressed into $output_file"
```

关键事实（全部可核对）：
- 测试命令 = `uv run pytest -v ./tests --junitxml=test_results.xml`
- 有 `|| true` → **测试失败不会中断打包**
- 没有 modal、没有 leaderboard、没有网络提交
- 打包名写的是 **`cs336-spring2024-...`**（2024，不是 2025 —— 复制粘贴遗留）
- `-x '*.pt'` 会排除 `tests/fixtures/ddp_test_data.pt`、`tests/fixtures/ddp_test_labels.pt`、`mnist_simple_ddp.pt`；`-x '*.txt'` 会排除 `requirements.txt`（**推断**：从解压包内直接跑 pytest 会缺 fixture）

官方 repo README 原文（对照）：「To submit, run `./test_and_make_submission.sh` . This script will install your code's dependencies, run tests, and create a gzipped tarball with the output.」

### 4.2 全部 test 函数逐个精读

**`tests/test_attention.py`（6 个用例）**

| test 函数 | 测什么 | GPU 需求（从代码判定） |
|---|---|---|
| `test_flash_forward_pass_pytorch` | 调 `_test_flash_forward_pass(get_flashattention_autograd_function_pytorch().apply)`，device 默认 `"cpu"`；比对 O 与 **logsumexp L**（rtol/atol=1e-2）；断言 saved_tensors 中**恰好一个** shape==(4,128) 的张量（即 L） | **0 张**（CPU） |
| `test_flash_forward_pass_triton[False]` / `[True]` | 同上但 `device="cuda"`，`is_causal` 参数化；`@pytest.mark.skipif(not torch.cuda.is_available())` | **≥1 张**（无 CUDA 则 skip） |
| `test_flash_backward_pytorch` | 与 `_attention_and_lse` 的 autograd 对比 dq/dk/dv，`is_causal=False` | **0 张**（CPU） |
| `test_flash_backward_triton[False]` / `[True]` | 同上，`device='cuda'`，参数化 causal | **≥1 张**（无 CUDA 则 skip） |

测试输入固定：`batch_size=4, n_queries=n_keys=128, D=64`。**注意测试以 4 个位置参数调用 `.apply(q,k,v,is_causal)`**。

**`tests/test_ddp.py`（6 个用例）**
- `test_DistributedDataParallelCPU(bucket_size_mb, model_class)`，参数化 `model_class ∈ {ToyModel, ToyModelWithTiedWeights}` × `bucket_size_mb ∈ {0.0016, 0.0001, 0.01}` = 6 用例
- 用 `mp.spawn(..., world_size=2, nprocs=2)`；`_setup_process_group(backend="gloo")`
- 测试 docstring 原文（说明桶数设计）：「bucket_size_mb 0.0016 旨在测试有2个桶的情况……0.0001 旨在测试有3个桶的情况（每个桶有1个参数张量）。0.01 旨在测试有1个桶的情况（包含3个参数张量）。」
- 覆盖：初始 broadcast（rank≠0 参数须已变）、`requires_grad=False` 参数（`no_grad_fixed_param`）不得改变、tied weights、5 步训练后与"所有 20 样本一次性训练"的非并行基线**逐参数 allclose**
- 流程钩子：`ddp_bucketed_on_train_batch_start(...)` → forward/backward → `ddp_bucketed_on_after_backward(...)` → `step()`
- 依赖 `tests/fixtures/ddp_test_data.pt`（20×10）、`ddp_test_labels.pt`（20×5）

**`tests/test_ddp_individual_parameters.py`（2 个用例）**
- `test_DistributedDataParallelIndividualParameters(model_class)`，2 个 model_class
- `world_size=2`，`mp.spawn`，`backend="gloo"`；流程钩子仅 `ddp_individual_parameters_on_after_backward(ddp_model, ddp_optimizer)`（没有 train_batch_start）
- 同样校验初始 broadcast、non-grad 参数、5 步后与非并行基线一致

**`tests/test_sharded_optimizer.py`（2 个用例）**
- `test_sharded_optimizer(model_class)`，2 个 model_class
- `world_size=2`，`mp.spawn`，`backend="gloo"`；`optimizer_cls = torch.optim.AdamW`，`lr=0.1, weight_decay=0.1, betas=(0.9,0.999), eps=1e-8`，跑 **10 步**，最后 `numpy.testing.assert_allclose` 比对分片与非分片模型参数

**合计 16 个测试用例。**

**`tests/common.py`（关键，决定 GPU 行为）** —— 逐字核心：

```python
def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device
```

且 `validate_ddp_net_equivalence(net)` 使用 **`net.module.state_dict()`** —— 即 DDP 容器**必须暴露 `.module` 属性**。

**`tests/conftest.py`**：提供 `snapshot` / `numpy_snapshot` fixture 与 `--snapshot-exact` 选项；**没有**被上面 4 个测试文件使用（它们都不请求 fixture）。

### 4.3 隐藏测试

**抓取内容中的事实**：仓库 `tests/` 只有 `test_attention.py / test_ddp.py / test_ddp_individual_parameters.py / test_sharded_optimizer.py` 4 个测试文件；`test_and_make_submission.sh` 只跑 `./tests`；仓库内未出现任何隐藏测试脚本、也未出现 modal / leaderboard 相关代码。
**推断（不在抓取内容中）**：原版 Stanford CS336 通常另有用于评分的隐藏测试与 leaderboard 提交流程；但我在本次抓取中**没有拿到任何官方评分页原文**，因此"是否有隐藏测试"**无法从抓取内容判定**。

---

## 5.【原文抓取】Triton 与 Windows/Linux 约束（单卡可行性关键）

**`requirements.txt` 逐字**（由 `uv export --format requirements-txt` 生成）：

```
triton==3.5.1 ; platform_machine == 'x86_64' and sys_platform == 'linux' \
    --hash=sha256:0b4d2c70127fca6a23e247f9348b8adde979d2e7a20391bfbabaac6aebc7e6a8 \
    ...
    # via torch
```

同时：`torch==2.9.1`（**Windows/Linux 通用**，requirements 中 torch 行无 sys_platform 限制）；`numpy==2.4.1`；`wandb==0.24.0`；`einops==0.8.1`；`einx==0.3.0`；`jaxtyping==0.3.5`。

**`pyproject.toml` 逐字**（依赖里**没有 triton**）：

```toml
[project]
name = "assignment2-system"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "einops>=0.8.1", "einx>=0.3.0", "ipykernel>=7.1.0",
    "jaxtyping>=0.3.5", "pytest>=9.0.2", "torch>=2.9.1",
    "torchvision>=0.24.1", "wandb>=0.24.0",
]
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[tool.setuptools]
package-dir = {"" = "src"}
[tool.setuptools.packages.find]
where = ["src"]
```

**作业1.ipynb 逐字**（仓库自己的 Windows 说明）：

> 「## 在windows上安装Ubuntu-22.04（使用triton需要使用在linux上运行）
> 由于triton在windows没有可用的版本，考虑到大家都是使用的windows系统的比较多，这里附一个安装教程，有linux系统或者不想运行triton模块的同学可以跳过……」
>
> 「整个作业使用**linux或者ubuntu**进行，读者可以使用**windows的wsl**来在windows中跑通……」

**Torch/CUDA 安装提示（作业1.ipynb 逐字）**：

> 「这边通过pyproject.toml下载的torch是cpu版本
> 如果要安装GPU版本：
> ```bash
> uv pip uninstall torch
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> cu121是我的cuda版本为12.1，可以改为自己的cuda版本」

**仓库作者实际跑的环境（作业1.ipynb 输出逐字）**：`Torch version: 2.9.1+cu128` / `CUDA available: True` / `CUDA device count: 1` / `Device name: NVIDIA GeForce RTX 4060 Laptop GPU`。→ **该 notebook 是在单张 8GB 笔记本显卡上跑通的。**

---

## 6.【原文抓取】仓库自己关于"单卡够不够"的声明

`作业1.ipynb` 逐字：

> 「本次作业是关于GPU和分布式训练的，**如果有多个GPU就可以跑通，只有一个GPU也可以，但是就没有分布式训练的速度优势，但是作为教学学习是足够的**。」

**这是本次抓取中唯一直接的"单卡可行性"原文表述**（来自共建方，非官方讲义）。

`作业2.ipynb` 逐字（兜底降规模）：

> 「# 3. 0.5B模型定义，可以自行修改参数配置，**这里是让笔记本也能跑起来**」
> `class XLModel(nn.Module): def __init__(self, hidden_size: int = 1024, num_layers: int = 24)`（`nn.Embedding(50000, 1024)` + 24×`TransformerEncoderLayer` + `nn.Linear(1024, 50000)`，注释「总参数量约 0.5B（5亿参数）」，input `(2, 512)`）

→ 官方题目要 **XL（表 1：d_model 1600 / 48 层）**，共建版**主动把 demo 缩到 0.5B** 才能在笔记本上跑。

另：`作业2.ipynb` 的 `distributed_communication_single_node_demo.py` 里 `main()` 把进程数写成 `for world_size in [1]`（原文），`--device` 默认 `"cuda"`，并有 `assert args.world_size <= torch.cuda.device_count(), "world_size 不能超过 GPU 数量"`。

---

## 7.【原文抓取 vs 推断】单卡 / 多卡分界

### 7.1 纯事实（来自抓取代码）

| 环节 | 抓取到的硬性要求 |
|---|---|
| `tests/test_attention.py` 的 pytorch 两例 | device 默认 `"cpu"`，**0 GPU** |
| `tests/test_attention.py` 的 triton 四例 | `device="cuda"`，**≥1 GPU**，无 CUDA 自动 skip |
| `tests/test_ddp.py` 6 例 | `mp.spawn(world_size=2, nprocs=2)` + `backend="gloo"`；device 由 `_setup_process_group` 决定 |
| `tests/test_ddp_individual_parameters.py` 2 例 | 同上（gloo，world_size=2） |
| `tests/test_sharded_optimizer.py` 2 例 | 同上（gloo，world_size=2） |
| benchmark 题目 | `naive_ddp_benchmarking` / `minimal_ddp_flat_benchmarking` / `ddp_bucketed_benchmarking (a)` 均写「**1个节点 x 2个GPU**」；`distributed_communication_single_node`「最多可使用 **6 张 GPU**」（2/4/6 进程）；CHANGELOG：`distributed_communication_multi_node` = **2x1、2x2、2x3** |
| `memory_profiling` | 要求对 **2.7B 模型**（表 1：d_model 2560 / 32 层）在 context 128/256/512 下做 CUDA memory profiling |

### 7.2 推断（明确标注：这是我的推断，不在抓取内容中）

1. **官方测试套件本身不要求 2 张物理 GPU。** DDP 系列测试全部用 `gloo`，且 `common.py` 在无 CUDA 时显式退回 `device="cpu"`。因此单卡机器可以用 `CUDA_VISIBLE_DEVICES=""` 让 `torch.cuda.is_available()` 为 False，从而让 world_size=2 的两个进程跑在 CPU+gloo 上——这正好是"`test_DistributedDataParallelCPU`"这个名字的原意。
2. **在真的装了 GPU 的单卡机器上，两个 rank 会同时落到 `cuda:0`**（`local_rank = rank % device_count = 0 % 1 = 0`）。这会用 gloo 在单卡上跑 CUDA 张量的集合通信。**推断**这一步能跑但不代表真实多卡；更稳的做法是显式 `CUDA_VISIBLE_DEVICES=""` 走 CPU 路径。
3. **"必须 2+ 张卡"的部分是 benchmark 题目，不是测试**：即 `naive_ddp_benchmarking`、`minimal_ddp_flat_benchmarking`、`ddp_bucketed_benchmarking (a)`（都要 1 节点×2 GPU 的 XL 测量）、`distributed_communication_multi_node`（2x1/2x2/2x3，本质需要 ≥2 台机器）。
4. **2.7B memory profiling 在 12GB 上不可行**：2.7B fp32 权重 ≈ 10.8GB，加梯度/优化器状态/激活远超 12GB。**推断**：单卡只能降级到 small/medium（甚至 large 也危险），并在报告中说明降级理由。
5. **XL（1.6B，d_model 1600/48层）训练步在 12GB 上不可行**：fp32 权重 ≈ 6.4GB，Adam 状态 ≈ 12.8GB。**推断**：必须换成 0.5B 或更小（共建版已经这么做），或租 2×A100。
6. **Triton 在 4070 Super（Ada, sm_89）上可用——前提是 Linux（含 WSL2）。** Triton 3.5.1 支持 sm_89；但在原生 Windows 上 `requirements.txt` 的 `sys_platform == 'linux'` 标记会让 triton 根本装不上（这与作业1.ipynb 的"triton 在 windows 没有可用的版本"一致）。WSL2 里 `platform_machine=='x86_64' and sys_platform=='linux'` 成立，且 WSL2 支持 CUDA passthrough，`torch.cuda.device_count()` 为 1。
7. **打包脚本 `-x '*.pt'` 会漏掉测试 fixture**，**推断**从解压包内直接跑 pytest 会缺 `tests/fixtures/*.pt`（评分方应该有原仓库，所以不一定影响评分）。

---

## 8.【原文抓取】仓库现状的接口不一致（这些会直接卡住跑测试）

全部为逐字对照得出的事实：

1. `tests/adapters.py` 8 个函数**全部 `raise NotImplementedError`** → 仓库开箱 `pytest tests/` 必然失败。
2. `flash_attention.py` 的 `forward(ctx, Q, K, V)` **没有 `is_causal`**，而 `test_attention.py` 以 `.apply(q, k, v, is_causal)` 调用 → 直接 TypeError（`test_flash_backward_pytorch` 传 `False`）。
3. `flash_attention_triton.py` **只有 forward，没有 backward**，而 `test_flash_backward_triton` 需要 backward。
4. 类名/接口不一致：`ddp_bucket.py` 提供的是 `TeachingDDPBucketed(model, bucket_size_mb)`（属性名 `self.model`，同步函数叫 `finalize_gradients`），而测试要的是 `get_ddp_bucketed(...)` + `ddp_bucketed_on_after_backward` + `ddp_bucketed_on_train_batch_start`。
5. `tests/common.py::validate_ddp_net_equivalence` 调 `net.module.state_dict()`：`DDPOverlapBucketed` 有 `self.module` ✅；`TeachingDDPBucketed` / `TeachingDDPOverlapPerParam` 用 `self.model` ❌。
6. `sharded_optimizer.py` 的 `super().__init__(params, defaults={})` 后 `_build_local_optimizer` 拿 `group["lr"]` 等键去建 `AdamW`；**推断** `defaults={}` 下这些键可能不存在（`add_param_group` 会把 `defaults` 合进 group），需要补 `defaults` 或显式 kwargs。
7. `pyproject.toml` 用 `package-dir = {"" = "src"}` + `packages.find where=["src"]`，而 `cs336_systems/` 不在 `src/` 下 → **推断** `uv sync` 后 `cs336_systems` 可能无法作为包导入（测试用的是相对导入 `from .adapters import`，所以不受影响）。

---

## 9.【推断/估计】结论：RTX 4070 Super 12GB 单卡能不能做？

### 结论：**能做完大部分（"只能部分"，但不是"不能"）。**

判定依据分三层：

**A. 官方 pytest（16 个用例）——单卡可全绿（推断）**
- 4 个 attention 用例走 CPU；4 个 triton 用例只要 1 张支持 Triton 的卡（4070 Super 在 WSL2 下满足）。
- 8 个 DDP/sharded 用例是 `world_size=2` + gloo。用 `CUDA_VISIBLE_DEVICES=""` 退回 CPU 双进程即可完成"功能正确性"验证。**这是本报告最重要的判断**：这些测试被测的是**梯度同步的数值正确性**，不是多卡带宽。
- 风险点：在装了 GPU 的机器上，两个 rank 都会落到 `cuda:0`（推断可用，但更建议显式屏蔽 CUDA）。

**B. 性能类题目——只能"降规模 + 说明"，拿不到官方参考量级**
- 官方题面写死 **1 节点 × 2 GPU + XL 模型**，且 `distributed_communication_multi_node` 要 2x1/2x2/2x3。单卡**做不了**这些数字。
- 2.7B memory profiling（作业四）在 12GB 上**做不了**，必须降到 small/medium。
- 作业五的 attention 扫描（seqlen 到 16384、d_model 128）在 12GB 上**会 OOM**——但**这恰好是题目要求你报告的**（「在什么规模下会出现 out-of-memory？」）。仓库作者在 8GB 的 4060 Laptop 上就是在 T=16384 全 OOM，说明这条路是通的。
- 想要真实双卡数据：租 2 卡（2×A100 40G 约 1–2 小时）或按小时租 2×RTX 4090。**估计**只跑 benchmark 脚本的话 2–4 小时云时长足够。

**C. 环境——必须在 WSL2/Linux 下做，不能在原生 Windows 下做**
- `requirements.txt` 里 `triton==3.5.1 ; platform_machine == 'x86_64' and sys_platform == 'linux'`。
- 共建方原文：「由于triton在windows没有可用的版本」。
- **所以：4070 Super 这台 Windows 机器 = 装 WSL2，在 WSL2 里跑本作业。** 注意 WSL2 内 `torch.cuda.device_count()` 仍是 1，显存上限仍是 12GB。

### 【估计】工时（这是主观估计，非抓取内容）

| 阶段 | 估计工时 |
|---|---|
| 环境（WSL2 + CUDA + uv + triton 自检 + 关掉 CPU-only torch） | 4–8 h |
| 作业一~三（benchmark 脚本 + 混合精度 + 两种规模扫描 + 图表） | 6–10 h |
| 作业四 memory profiling（降规模 + memory_viz 截图 + 推导） | 3–6 h |
| 作业五 a/b（naive attention 扫描 + 显存核算 + OOM 边界） | 5–8 h |
| 作业五 FlashAttention **PyTorch 版**（前向+反向，含 causal、L 的保存约束） | 6–10 h |
| 作业五 FlashAttention **Triton 版**（前向+反向 kernel，debug 数值） | 15–30 h |
| DDP：naive(5) + flat(2) + individual-params overlap(5) + bucketed(3) | 15–25 h |
| ShardedOptimizer（tests 有、notebook 无论述） | 6–12 h |
| 报告/图表/文字交付物 | 6–10 h |
| 合计 | **估计 66–119 h；单卡 + WSL2 环境下更可能落在 80–120 h** |

参考：共建方还额外做了 `ddp_bucketed_benchmark_complete.py`（17790 B）这类完整 benchmark 脚本，说明实际投入不小。

### 【区分】核心必做 vs 可跳过（逐项）

**必做（P0，不做就是没做完）**
1. `tests/adapters.py` 8 个函数 + 对应实现；`pytest tests/` 16 例全绿。
2. FlashAttention PyTorch 版（含 `is_causal`、恰好保存一个 (B,Nq) 的 L）。
3. FlashAttention Triton 版前向 + 反向。
4. 三个 DDP 变体：naive / flat / individual-parameters overlap；bucketed + 三个钩子函数。
5. `ShardedOptimizer`（AdamW 状态分片 + 参数广播同步）。
6. 作业一~六的 benchmark 脚本与交付物（表格/截图/文字）。

**可跳过或降级（P1）**
7. XL(1.6B) 在 2 GPU 上的真实通信数据 → 降级为 0.5B 单卡 + 明确写"降规模原因"。
8. 2.7B memory profiling → 降级到 small/medium/large，并写明 12GB 约束。
9. `distributed_communication_multi_node`（2x1/2x2/2x3）→ 直接跳过，或云租 2 机。
10. `distributed_communication_single_node` 的 4/6 进程 + 1GB + NCCL 全网格 → 只做 1–2 进程 + 到 100MB（12GB 装得下 1GB 张量，但 6 进程在 1 张卡上跑不出有意义的重叠数据）。
11. **leaderboard 提交** → 抓取内容里没有任何计分脚本，**无法判断官方是否强制**；若有，单卡排名一定垫底，建议只做"能跑通"不做"冲榜"。
12. `--snapshot-exact` / `conftest.py` 的 snapshot fixture → 现有 4 个测试文件都不用，可忽略。

---

## 10. 抓取失败项（全部，不省略）

| URL / 目标 | 失败原因 |
|---|---|
| `https://stanford-cs336.github.io/spring2025-assignment2-systems/` | web_fetch 拒绝跨源重定向：`cross-origin redirect to http://cs336.stanford.edu is not followed automatically` |
| `http://cs336.stanford.edu/spring2025-assignment2-systems/`（按上一条提示直接重试） | **HTTP 404**（返回 GitHub Pages 的 404 页面） |
| `https://raw.githubusercontent.com/stanford-cs336/assignment2-systems/main/cs336_assignment2_systems.pdf` | 抓取失败：`unsupported content type "application/octet-stream"`（PDF 二进制，web_fetch 无法解析） |
| `相关文档/cs336_spring2025_assignment2_systems.pdf` | 未抓取：同类 PDF；已由上一条证实 web_fetch 对 raw PDF 返回 `application/octet-stream`，无法解析（**未重试，属工具能力限制**） |
| `相关文档/【翻译】cs336_spring2025_assignment2_systems.pdf` | 未抓取：同上（PDF） |
| `相关文档/image.png`、`相关文档/【脑图】….jpeg` | 未抓取：图片，本次任务不需要（非失败，属未取） |
| `cs336_systems/作业2.ipynb` 完整内容 | **部分失败**：web_fetch 截断，末尾 **69156 字节未返回/未落盘**（spill 文件末尾即截断标记） |
| `cs336_systems/作业1.ipynb` 完整内容 | **部分失败**：web_fetch 截断，末尾 **64211 字节未返回/未落盘** |
| `requirements.txt` 完整内容 | 轻微：末尾 **2359 字节**被省略（triton / torch / numpy 等关键行已取到） |
| 本地 pwsh/curl 直连 raw.githubusercontent.com（用于 grep 大文件） | 失败：`Invoke-WebRequest` → `基础连接已经关闭: 接收时发生错误`；`curl.exe` → `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`。**改走 web_fetch + spill 文件 grep 解决。** |
| 官方评分页 / leaderboard 提交页 / `compute_leaderboard_score` 源码 | **未找到任何可抓取 URL**；仓库抓取内容中零次出现 `compute_leaderboard_score` 与 `modal` |
| 隐藏测试是否存在的官方说明 | **无法判定**：未抓到任何官方评分页原文 |
| `api.github.com/.../contents/...`（中文文件名兜底方案） | **未使用**：本次全部抓取都走 `raw.githubusercontent.com` 且中文路径按 `%E4%BD%9C%E4%B8%9A2.ipynb` 形式 URL 编码后成功，因此**没有因为上游 403 API rate limit 损失任何内容**（该 403 是本任务期间的上游环境状况，对本报告无影响） |

---

## 11. 一句话回答提问

**在单张 RTX 4070 Super 12GB 上"能做，但只能做部分"**：官方 16 个 pytest 用例（含 4 个 DDP/sharded 的双进程测试）可以在单卡上跑通（attention 走 CPU + 1 卡 Triton；DDP 系列走 gloo，必要时用 `CUDA_VISIBLE_DEVICES=""` 退回 CPU 双进程）；**做不了的是题目里写死"1 节点 × 2 GPU + XL 模型"的性能数字、2.7B 的显存 profiling、以及 2x1/2x2/2x3 的多节点测量**——这些只能降规模做或租云卡。**并且必须先在 Windows 上装 WSL2**，因为 `triton==3.5.1` 的安装标记是 `sys_platform == 'linux'`，共建方也明确写了「triton在windows没有可用的版本」。
