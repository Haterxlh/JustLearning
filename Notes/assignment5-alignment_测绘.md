# CS336 作业5（Alignment：SFT + GRPO）测绘报告
### 目标仓库：`datawhalechina/diy-llm` → `coursework/assignment5-alignment/`

> 本报告严格区分两类内容：
> - **【原文】** = 本次 web_fetch 实际抓到的字符，逐条可回溯。
> - **【估计】** = 我的推断/计算，**不在抓取内容中**，含官方原版经验外推。
>
> 测绘日期：本次会话。所有抓取链接见文末「抓取来源」。

---

## 0. 结论先行

| 问题 | 答案 | 依据 |
|---|---|---|
| 单张 4070 Super 12GB 能不能做完？ | **不能完整做完；只能做一个大幅缩水的子集（约 1/4 ~ 1/3 的题量）** | 【估计】 |
| 主路径基座模型 | **Qwen2.5-Math-1.5B**（1.5B） | 【原文】 |
| Llama-3.1-8B / Llama-3.3-70B | **只出现在「可选」的指令微调补充作业里，主作业不用** | 【原文】 |
| 是否必须 vLLM | 官方措辞是「推荐」，但参考实现的代码与依赖是**硬依赖** | 【原文】 |
| 是否必须多卡 | 参考实现**硬编码 2 张卡**（`cuda:0`/`cuda:1`），但 1.5B 本身不需要张量并行；多卡只为「训练与 vLLM 常驻共存」 | 【原文】+【估计】 |
| 有没有 pytest | 官方有（`tests/test_sft.py`、`tests/test_grpo.py`、`tests/adapters.py`）；**本仓库的这份拷贝被删掉了 tests/** | 【原文】 |
| 有没有本地测试脚本 / test_and_make_submission.sh | **本仓库没有**（GitHub API 目录清单里不存在） | 【原文】 |
| 有没有隐藏测试 | 有：官方 `tests/` 即评分用测试，官方 README 明说「Initially, all tests should fail with NotImplementedError」 | 【原文·外部】 |
| 全量官方预算 | **52 个 H100 小时**（仅统计带小时标注的题） | 【原文】 |

---

## 1. 仓库事实核查（GitHub API 实测，非猜测）

### 1.1 顶层目录 `coursework/assignment5-alignment/` 实际条目

GitHub Contents API 返回的**完整**条目（10 条）：

```
README.ipynb                                        (131,926 B)
README.md                                           (  9,832 B)
README.pdf                                          (181,475 B)
cs336_alignment/                                    (dir)
cs336_spring2025_assignment5_alignment.pdf          (281,460 B)
cs336_spring2025_assignment5_alignment_zh.md        ( 98,848 B)
cs336_spring2025_assignment5_supplement_safety_rlhf.pdf (148,716 B)
images/                                             (dir)
pyproject.toml                                      (  1,024 B)
uv.lock                                             (489,119 B)
```

### 1.2 `cs336_alignment/` 实际条目（完整）

```
__init__.py                 (0 B)
drgrpo_grader.py            (29,434 B)
evaluate_math.py            ( 6,303 B)
gpro_helper.py              (17,638 B)
grpo.py                     (17,158 B)
grpo_experiments.py         (33,146 B)
log.py                      ( 4,616 B)
prompts/                    (dir)
sft_helper.py               ( 5,768 B)
sft_math_reasoning.py       (13,971 B)
sft_math_reasoning_ei.py    (15,501 B)
```

### 1.3 对你「已知真实文件树」的核实结论

- 你给的文件树 **与 API 实测一致**（顶层 10 项 + `cs336_alignment/` 11 项 + `prompts/` 4 个模板）。
- **确认：该目录下没有 `tests/` 目录。**
- **确认：没有 `test_and_make_submission.sh`。**
- 此外 API 清单里**还不存在**这些通常会有的东西：`tests/adapters.py`、`tests/conftest.py`、`data/` 目录、任何 `run_*.sh`、`Makefile`。
- **注意一个反讽**：`pyproject.toml` 仍然声明了 `pytest>=8.3.5`，`README.ipynb` 里也写着「运行单元测试：`uv run pytest`」，而 `README.md` 的 FAQ 写着「**优先以官方讲义 + tests 为准**」——**但这个仓库里没有任何 tests**。在这份拷贝上跑 `uv run pytest` 会收集到 0 个测试。这是本次测绘最关键的可执行性事实之一。

---

## 2. 任务清单：要写哪些模块/函数（函数名级，全部【原文】）

### 2.1 官方作业说明（`cs336_spring2025_assignment5_alignment_zh.md`）点名的必做方法与 pytest 选择器

| 章节 | 问题 ID | 分数 | 要实现的函数（原文函数名） | 原文给出的测试命令 |
|---|---|---|---|---|
| §3.2 | `math_baseline` | 4 | `evaluate_vllm`（"建议包含"） | 无 pytest |
| §4.2 | `tokenize_prompt_and_output` | 2 | `tokenize_prompt_and_output` | `uv run pytest -k test_tokenize_prompt_and_output` |
| §4.2 | `compute_entropy` | 1 | `compute_entropy` | `uv run pytest -k test_compute_entropy` |
| §4.2 | `get_response_log_probs` | 2 | `get_response_log_probs` | `uv run pytest -k test_get_response_log_probs` |
| §4.2 | `masked_normalize` | 1 | `masked_normalize` | `uv run pytest -k test_masked_normalize` |
| §4.2 | `sft_microbatch_train_step` | 3 | `sft_microbatch_train_step` | `uv run pytest -k test_sft_microbatch_train_step` |
| §4.2 | `log_generations` | 1 | `log_generations` | 原文未给测试命令 |
| §4.3 | `sft_experiment` | 2 | 完整 SFT 流程 | 无 |
| §5 | `expert_iteration_experiment` | 2 | 完整 EI 流程 | 无 |
| §7.2 | `compute_group_normalized_rewards` | 2 | `compute_group_normalized_rewards` | `uv run pytest -k test_compute_group_normalized_rewards` |
| §7.2 | `compute_naive_policy_gradient_loss` | 1 | `compute_naive_policy_gradient_loss` | `uv run pytest -k test_compute_naive_policy_gradient_loss` |
| §7.2 | `compute_grpo_clip_loss` | 2 | `compute_grpo_clip_loss` | `uv run pytest -k test_compute_grpo_clip_loss` |
| §7.2 | `compute_policy_gradient_loss` | 1 | `compute_policy_gradient_loss` | 原文未给测试命令 |
| §7.2 | `masked_mean` | 1 | `masked_mean` | `uv run pytest -k test_masked_mean` |
| §7.2 | `grpo_microbatch_train_step` | 3 | `grpo_microbatch_train_step` | `uv run pytest -k test_grpo_microbatch_train_step` |
| §7.2 | `grpo_train_loop` | 5 | 完整 GRPO 训练循环 | 无 |
| §8 | `grpo_learning_rate` | 2 | 学习率扫描 | 无 |
| §8 | `grpo_baselines` | 2 | 损失类型对比 | 无 |
| §8 | `think_about_length_normalization` | 1 | 纯书面 | 无 |
| §8 | `grpo_length_normalization` | 2 | 归一化对比 | 无 |
| §8 | `grpo_group_standard_deviation` | 2 | std 归一化开关 | 无 |
| §8 | `grpo_off_policy` | — | 离策略 GRPO（原文未标分数） | 无 |
| §8 | `grpo_off_policy_sweep` | 4 | 超参扫描 | 无 |
| §8 | `grpo_off_policy_clip_ablation` | 2 | 新增损失类型 `"GRPO-No-Clip"` | 无 |
| §8 | `grpo_prompt_ablation` | 2 | 提示消融 | 无 |
| §9 | `leaderboard` | 16 | 2×H100 / 4h 内冲最高验证奖励 | 无 |

**合计约 56 分**（其中排行榜独占 16 分 ≈ 29%）。

原文对「能通过哪些测试」的界定：
> 「你只需要通过 `tests/test_sft.py` 和 `tests/test_grpo.py` 中的测试——其余测试是为作业的非强制部分准备的。」

### 2.2 本仓库参考实现里**已经写好**的函数签名（原文，供对照抄作业/自查）

> **重要事实**：本仓库的 9 个 `.py` 文件**全部是完整实现**，在我抓取到的内容中**没有出现任何 `NotImplementedError` 或 `TODO` 占位**。`README.md` 里说「`sft_helper.py`：你需要实现的核心函数」与实际代码不符。也就是说：**这份拷贝是一份「参考答案 + 实验脚手架」，不是空白作业模板。**

**`sft_helper.py`（原文）**
```python
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str],
                               tokenizer: PreTrainedTokenizerBase) -> Dict[str, torch.Tensor]
def compute_entropy(logits: torch.Tensor) -> torch.Tensor
def get_response_log_probs(model: torch.nn.Module, input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool = False) -> Dict[str, torch.Tensor]
def masked_normalize(tensor, mask, normalize_constant: float,
                     dim: int | None = None) -> torch.Tensor
def sft_microbatch_train_step(policy_log_probs, response_mask,
                              gradient_accumulation_steps: int,
                              normalize_constant: float = 1.0
                              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

**`gpro_helper.py`（原文）**
```python
def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths,
                                     group_size: int, advantage_eps: float,
                                     normalize_by_std: bool
                                     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]
def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs) -> torch.Tensor
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange: float,
                           loss_type: Literal["grpo_clip","grpo_no_clip"] = "grpo_clip"
                           ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
def compute_policy_gradient_loss(policy_log_probs,
                                 loss_type: Literal["no_baseline","reinforce_with_baseline",
                                                    "grpo_clip","grpo_no_clip"],
                                 raw_rewards=None, advantages=None,
                                 old_log_probs=None, cliprange=None
                                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
def masked_mean(tensor, mask, dim: int | None = None) -> torch.Tensor
def masked_normalize(tensor, mask, dim: int | None = None,
                     constant_normalizer: float = 1.0) -> torch.Tensor   # 注意与 sft_helper 同名不同签名
def grpo_microbatch_train_step(policy_log_probs, response_mask,
                               gradient_accumulation_steps: int, loss_type,
                               raw_rewards=None, advantages=None, old_log_probs=None,
                               cliprange=None,
                               length_norm: Literal["masked_mean","masked_normalize"] = "masked_mean"
                               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

**`drgrpo_grader.py`（原文）** —— 奖励函数，无需你实现，但决定评分口径
```python
def r1_zero_reward_fn(response, ground_truth, fast=True) -> dict   # 返回 reward / format_reward / answer_reward
def question_only_reward_fn(response, ground_truth, fast=True) -> dict
def grade(model_answer: str, gt_answer: str, fast: bool = True) -> bool
def extract_answer(passage: str) -> str
def is_latex_equal / is_value_equal / grade_answer_sympy / grade_answer_mathd ...
```
原文对格式奖励的判定条件（严格到反直觉）：
```python
if "</think> <answer>" in response and "</answer>" in response:
```
即 **`</think>` 与 `<answer>` 之间必须有恰好一个空格**。README.ipynb 自己实测发现模型常输出 `</think>\n<answer>`，因此大量样本拿不到 format_reward。

**`evaluate_math.py`（原文）**
```python
def evaluate_vllm(vllm_model: LLM, reward_fn: Callable[[str,str,bool],Dict[str,float]],
                  dataset_path: str, prompt_template: str,
                  eval_sampling_params: SamplingParams, output_filepath: str,
                  fast: bool = True) -> Dict[str, Any]
def load_r1_zero_prompt(prompt_file_path: str) -> str
def format_prompt(question: str, prompt_template: str) -> str
```

**`log.py`（原文）**：`def log_generations(prompts, responses, ground_truths, reward_infos=None, token_entropies=None, response_lengths=None, *, step=None, max_examples_to_print=3) -> Dict[str, Any]`

**`sft_math_reasoning.py`（原文）**：`load_prompt_template` / `format_prompt` / `init_vllm` / `load_policy_into_vllm_instance` / `log_generations_to_wandb` / `run_sft_experiment` / `main`

**`sft_math_reasoning_ei.py`（原文）**：`compute_mean_entropy` / `init_vllm` / `load_policy_into_vllm` / `log_generations_to_wandb` / `run_expert_iteration_experiment` / `main`

**`grpo.py`（原文）**：`@dataclass GRPOConfig` / `load_prompt_template` / `format_prompt` / `init_vllm` / `load_policy_into_vllm_instance` / `load_json_or_jsonl` / `extract_question` / `extract_ground_truth` / `main(cfg)`

**`grpo_experiments.py`（原文）** —— Typer CLI，子命令：`lr-sweep` / `baselines` / `length-norm` / `std-norm` / `off-policy-sweep` / `clip-ablation` / `prompt-ablation` / `leaderboard` / `all-experiments`；内部函数 `run_grpo_experiment(cfg)` + `run_lr_sweep_experiment` / `run_baseline_experiment` / `run_normalization_experiment` / `run_std_norm_experiment` / `run_off_policy_sweep_experiment` / `run_clip_ablation_experiment` / `run_prompt_ablation_experiment` / `run_leaderboard_experiment`

---

## 3. 交付物（原文）

### 3.1 官方提交物（`_zh.md` §1「如何提交」，原文）
> 「你将向 Gradescope 提交以下文件：
> - **writeup.pdf**：回答所有书面问题。请对你的回答进行排版。
> - **code.zip**：包含你编写的所有代码。」

### 3.2 本仓库 README.md §4「写作与提交」补充（原文）
> - `writeup.pdf`：包含所有书面问题的答案
> - `code.zip`：至少包括 `cs336_alignment/sft_helper.py`、`cs336_alignment/gpro_helper.py`、你修改或新增的实验脚本（如 `evaluate_math.py`、`sft_math_reasoning*.py`、`grpo_experiments.py`）

### 3.3 实操层面还会产出（原文，从脚本与 README.md 归纳）
- 代码：`cs336_alignment/*.py`
- **训练曲线图**：`images/*.png`（约 40 个，如 `sft_train_loss.png`、`sft_eval_acc.png`、`sft_eval_format.png`、`sft_volume.png`、`sft_256_raw_vs_filter.png`、`sft_ei_epoch_*.png`、`grpo_lr_eval_acc.png`、`grpo_baseline_{train_loss,train_entropy,eval_acc}.png`、`grpo_length_norm_{train_loss,train_entropy,train_grad_norm,eval_acc}.png`、`grpo_std_norm_{...}.png`、以及 `algorithm1_sft.png`）
- **结果 jsonl**：`results/base/zero_shot_math_evaluation.jsonl`、`results/sft_experiments_*/*/step_*/results.jsonl`、`results/ei_*`、`results/grpo_*`
- **wandb offline 记录**：`wandb/` 目录（脚本里 `wandb.init(..., mode="offline")`）
- **写作文档**：`README.ipynb` 本体就是这份拷贝的 writeup/讲解（原文：README.md「所有的内容讲解和结果分析都在 `README.ipynb`」）

### 3.4 用了什么模型（原文，明确回答）
- **主路径唯一基座：`Qwen2.5-Math-1.5B`**（原文 `_zh.md` §2.2：「我们将使用 **Qwen 2.5 Math 1.5B Base** 模型」；代码里 `MODEL_PATH` / `BASE_MODEL` / `GRPOConfig.base_model` 全指向它）。
- **Llama-3.1-8B Base 与 Llama-3.3-70B Instruct 只用于「可选的指令微调实验」**（原文 §3.1：「Llama 3.1 8B Base（用于可选的指令微调实验）」「Llama 3.3 70B Instruct（用于可选的指令微调实验）」）。
- 补充作业（safety / RLHF / instruction tuning）官方明说是 **"completely optional"**（原文 §1：「我们将在未来几天发布一个完全可选的部分」）。

---

## 4. 官方测试与评分方式（原文）

### 4.1 有 pytest 吗？
**有，但不在这个仓库里。**

- 官方 `_zh.md` §1「代码结构」原文：
  > 「3. `tests/.py`：这包含了你必须通过的所有测试。你只需要通过 `tests/test_sft.py` 和 `tests/test_grpo.py` 中的测试——其余测试是为作业的非强制部分准备的。这些测试会调用在 `tests/adapters.py` 中定义的钩子（hooks）。你需要实现这些适配器（adapters）以将你的代码连接到测试中。编写更多测试和/或修改测试代码对调试你的代码很有帮助，但你的实现需要通过原始提供的测试套件。」
- `README.ipynb` 原文：「运行单元测试：`uv run pytest`」
- **本仓库实测：无 `tests/`、无 `adapters.py`、无 `conftest.py`**（§1.1/§1.3 API 实测）。

### 4.2 有本地测试脚本吗？
- **没有。** 没有 `test_and_make_submission.sh`，没有任何 `run_*.sh`，没有 `Makefile`。实验靠直接 `uv run python cs336_alignment/<script>.py`（README.md §3 原文给的就是这种命令）。

### 4.3 有隐藏测试吗？
**有。** 官方仓库的 `tests/` 就是那份测试套件（Gradescope 侧另有一套）。外部来源实测（**注意这是 Spring 2026 版**）：

- 官方 `README.md`（`stanford-cs336/assignment5-alignment` main 分支）原文：
  > 「Run the required unit tests: `uv run pytest tests/test_grpo.py`」
  > 「Initially, all tests should fail with `NotImplementedError`s. To connect your implementation to the tests, complete the functions in `./tests/adapters.py`.」
- 官方 `tests/adapters.py`（2026 版）实测包含的钩子（全部 `raise NotImplementedError`）：
  `run_tokenize_prompt_and_output`、`run_get_response_log_probs`、`run_compute_rollout_rewards`、`run_compute_group_normalized_rewards`、`run_compute_policy_gradient_loss`、`run_aggregate_loss_across_microbatch`、`run_grpo_train_step`，另加可选部分 `get_packed_sft_dataset`、`run_iterate_batches`、`run_parse_mmlu_response`、`run_parse_gsm8k_response`、`run_compute_per_instance_dpo_loss`。

> ⚠️ **版本不匹配警告（本次测绘的重要发现）**：官方 main 分支已经变成 **Spring 2026**，测试钩子名与 2025 handout 明显不同（2026 用 `run_compute_rollout_rewards` / `run_aggregate_loss_across_microbatch` / `run_grpo_train_step`；2025 用 `compute_entropy` / `masked_normalize` / `masked_mean` / `sft_microbatch_train_step` / `grpo_microbatch_train_step`）。**本仓库镜像的是 Spring 2025 v1.0.2**。若你要取回官方 tests，必须 checkout 2025 对应的 tag/commit，否则 adapter 对不上。

### 4.4 如何判定「对齐成功」？
不是靠单一测试，而是 **「pytest 通过（实现正确性） + 书面问题（writeup.pdf） + 各问题明确写出的验证准确率门槛」**。原文门槛：

| 原文位置 | 原文门槛 |
|---|---|
| §4.3 `sft_experiment` | 「调整学习率和 Batch Size 以在使用全量数据时达到至少 **15%** 的验证准确率」 |
| §5 `expert_iteration_experiment` | 「一个在 MATH 上达到至少 **15%** 验证准确率的模型」 |
| §8 `grpo_learning_rate` | 「一个在 MATH 上达到至少 **25%** 验证准确率的模型」 |
| §9 `leaderboard` | 「在 2 个 H100 GPU 上 4 小时训练内获得的验证准确率」，4 条硬约束（见下） |

`leaderboard` 的 4 条硬约束（原文）：
1. 验证准确率必须是**整个** MATH 验证集（所有 5K 示例）的平均准确率。
2. 验证时必须使用 **R1-Zero** 提示。
3. 评估时 vLLM 必须用 `temperature 1.0`、`max tokens 1024`。
4. 必须用启动代码提供的 `r1_zero_reward_fn` 产生的答案奖励来算验证准确率。

---

## 5. 算力需求（全部【原文】）

### 5.1 基座模型参数量
- Qwen2.5-Math-1.5B（**1.5B**）。原文 §2.2：「即使是像 1.5B 参数这样小的模型，使用经验证奖励的纯强化学习也能提高推理性能。」

### 5.2 参考实现自己声明的 GPU 需求（两条硬引用）
- `README.md` §2.3 原文：
  > 「建议使用带有至少 **80GB 显存**的 GPU（运行完整 SFT / GRPO 实验时更推荐 **2\*80GB 级别**）。」
- `README.ipynb` §4.6 原文：
  > 「由于 RL 训练需要消耗大量资源，为了在准确度和资源消耗之间取得平衡，我们在下面的 GRPO 训练使用了两张 GPU：
  > - **GPU 0**：策略模型训练
  > - **GPU 1**：vLLM 推理和评估」

代码层面的印证（原文）：`GRPOConfig.device_policy: str = "cuda:1"`、`device_eval: str = "cuda:0"`；`sft_math_reasoning.py` 里 `device_policy = "cuda:0"`、`device_eval = "cuda:1"`；并且每步都调用 `load_policy_into_vllm_instance(policy, llm)` 做权重同步。

### 5.3 官方每题 H100 小时预算（原文，逐题）

| 问题 | 预算（原文标注） |
|---|---|
| `sft_experiment`（§4.3） | **2 H100 小时** |
| `expert_iteration_experiment`（§5） | **6 H100 小时** |
| `grpo_learning_rate` | **6 H100 小时** |
| `grpo_baselines` | **2 H100 小时** |
| `grpo_length_normalization` | **2 H100 小时** |
| `grpo_group_standard_deviation` | **2 H100 小时** |
| `grpo_off_policy_sweep` | **12 H100 小时** |
| `grpo_off_policy_clip_ablation` | **2 H100 小时** |
| `grpo_prompt_ablation` | **2 H100 小时** |
| `leaderboard`（§9） | **16 H100 小时** |
| **合计** | **52 H100 小时**（不含 `math_baseline`、`grpo_train_loop`、`grpo_off_policy` 等未标时长的题） |

`leaderboard` 原文还规定：「在 **2 个 H100 GPU** 上 **4 小时**训练时间内获得尽可能高的验证奖励。」

### 5.4 GRPO 的 rollout 数量（原文超参）
来自 `grpo_experiments.py::GRPOConfig` 与 `README.ipynb` §4.6/§5.1：
```
n_grpo_steps                = 200
rollout_batch_size          = 256      # 每次 rollout 的响应总数
group_size                  = 8        # 每个问题采样 8 条 → 32 个 prompt/步
train_batch_size            = 256
epochs_per_rollout_batch    = 1
advantage_eps               = 1e-6
cliprange                   = 0.2
use_std_normalization       = False
length_norm                 = "masked_mean"
loss_type                   = "reinforce_with_baseline"
lr                          = 1e-5
grad_accum_steps            = 64
gen_max_tokens              = 1024
eval_max_tokens             = 1024
eval_max_examples           = 256      # 注释：max=1319
vllm_gpu_memory_utilization = 0.7
device_policy / device_eval = "cuda:1" / "cuda:0"
优化器                       = bnb.optim.AdamW8bit
```
→ 一次标准 200 步 GRPO 运行 = **200 × 256 = 51,200 条 rollout**，每条最长 1024 token。
`grpo_experiments.py` 还把多数实验缩到 `n_grpo_steps=50`（原文注释 `# Shorter for sweep`），`leaderboard` 用 `n_grpo_steps=400`。

### 5.5 SFT / EI 的规模（原文超参）

**SFT（`sft_math_reasoning.py`）**
```
BATCH_SIZE = 4 ; GRAD_ACCUM = 8 ; LR = 5e-5 ; EPOCHS = 1
EVAL_EVERY_STEPS = 5 ; MAX_GRAD_NORM = 1.0 ; SEED = 2026
DATASET_SIZES = [128, 256, 512, 1024, 2048]
```
SFT 数据规模 sweep 原文（`_zh.md` §4.3）：「变化唯一示例数量 {128, 256, 512, 1024} 及全量数据集」。

**专家迭代 EI（`sft_math_reasoning_ei.py`）**
```
N_EI_STEPS            = 5
EXPERT_BATCH_SIZES    = [512, 1024, 2048]   # Db
ROLLOUTS_PER_QUESTION = [1, 4, 8, 16]       # G
SFT_EPOCHS_PER_STEP   = [1, 4, 8, 16]
BATCH_SIZE = 4 ; GRAD_ACCUM = 8 ; LR = 5e-5
```
→ `main()` 要跑 3（Db）+ 3（G）+ 2（epochs）= **8 组** EI 实验，每组 5 轮。

### 5.6 是否用 vLLM 生成
**是。** 原文证据：
- `_zh.md` §3.1 标题就是「使用 vLLM 进行离线语言模型推理」，正文：「在本次作业中，我们将**推荐使用 vLLM** 进行离线批处理推理」。
- `pyproject.toml` 硬钉：`"vllm==0.7.2"`。
- `evaluate_math.py` / `sft_math_reasoning.py` / `sft_math_reasoning_ei.py` / `grpo.py` / `grpo_experiments.py` **全部**有 `from vllm import LLM, SamplingParams`（原文）。
- 评估生成超参（原文 §3.2）：「temperature=1.0、top_p=1.0、最大生成长度 1024」，停止串 `["</answer>"]`，`include_stop_str_in_output=True`。

### 5.7 使用的数据集（原文）
- 官方用 **MATH 12K**（`/data/a5-alignment/MATH`，validation 5K），但**因版权不可公开**。
- 本仓库替代方案（`README.ipynb` 原文）：「**baseline model**: `Qwen2.5-Math-1.5B`；**benchmark**: `GSM8K`」；「我们这里使用了 GSM8K 数据集来替代 MATH 12K 用作评测」。
- 训练数据（`README.ipynb` 原文）：「我们使用来自 `garg-aayush` 用户复现该项目时制作的 `hiyouga/math12k` 数据集。该数据集使用 gpt-oss 蒸馏的带推理轨迹的数据」（HF: `garg-aayush/sft-cs336-assign5-datasets/sft-reason`）。
- 代码内路径（原文）：`data/gsm8k/test.jsonl`、`data/sft/sft_gpt-oss-120b.jsonl`、`data/sft/sft_gpt-oss-120b_filtered.jsonl`；`README.md` 里却写成 `data/math12k/sft_gpt-oss-120b.jsonl`（**文档与代码路径不一致**）。
- **这些 `data/` 文件都不在仓库里**（API 实测无 `data/` 目录）→ 必须自己去 HuggingFace 下载。

---

## 6. 本仓库参考实现的实测结果数字（【原文】，来自 `README.ipynb`）

> ⚠️ 全部是 **GSM8K test（1319 条）** 上的数字，**不是**官方要求的 MATH 5K。**不可与官方的 15% / 25% 门槛直接对比**（GSM8K 明显更简单，同一模型同一方法在 GSM8K 上的分数会显著更高）。

- **Zero-shot 基线**（`README.ipynb` §1.7 原文 JSON）：
  `n=1319, format_rate=0.5042, answer_accuracy=0.1759, reward_mean=0.1759`
  `counts: format=1 answer=1 → 232；format=1 answer=0 → 433；format=0 answer=0 → 654；format=0 answer=1 → 0`
  结论原文：「基准模型 `Qwen2.5-Math-1.5B` 的回答准确率是 `17.59%`，格式准确率为 `50.42%`……问题主要出在**解析器的格式要求过于严格**，而不是基础模型的输出质量问题」（因为 `</think>` 与 `<answer>` 之间少一个空格就拿不到 format_reward）。
- **SFT**：数据集越大训练损失越低；**512 条**最好，达 **76.20%**；256 条时过滤从 75.50% → 76.20%；相对基座 17.59% → 76.20%。
- **EI**：Db=1024 最佳 **67.17%**（Db=512 初期最好，Db=2048 最稳）；G=4 最佳 **67.17%**（G=16 为 65.96%，G 增大先升后降）；epochs=16 最高 **71.80%**（epochs=4 → 63.08%，epochs=8 → 63.46%）。
- **GRPO 学习率扫描**（原文）：`1e-6: 18.88% / 5e-6: 32.98% / 1e-5: 51.71% / 2e-5: 66.57% / 5e-5: 49.28% / 1e-4: 0.61%`，最优 **2e-5 → 66.57%**。
- **基线消融**：`no_baseline: 49.68%` vs `reinforce_with_baseline: 34.95%`（原文结论：在当前超参下 `no_baseline` 更优）。
- **长度归一化**：`masked_mean: 63.38%` vs `masked_normalize: 69.37%`。
- **标准差归一化**：`use_std=True: 63.46%` vs `use_std=False: 68.00%`（原文记录了 use_std=True 在 step≈100 发生梯度爆炸/训练崩溃）。

---

## 7. 【我的估计，不在抓取内容中】单张 RTX 4070 Super 12GB 可行性

### 7.1 显存账（我的计算，非原文）

Qwen2.5-Math-1.5B 按 bf16 计：

| 项目 | 估算占用 | 说明 |
|---|---|---|
| 权重 bf16 | ~3.1 GB | 1.54B × 2 B |
| 梯度 bf16 | ~3.1 GB | 全参微调 |
| AdamW 8-bit 优化器状态 | ~3.1 GB | 2 个状态 × 1 byte/param（代码用的是 `bnb.optim.AdamW8bit`） |
| **训练小计（不含激活）** | **~9.3 GB** | 12 GB 卡只剩 ~2.5 GB 给激活 |
| 激活（micro-batch=2 × 1024 token，无梯度检查点） | 数 GB | 很可能直接 OOM |
| vLLM 单跑（`gpu_memory_utilization=0.5`） | ~6 GB 预算，权重 3.1 GB + KV ~3 GB | 单独跑没问题 |

**关键矛盾**：参考实现要让 **策略模型（含优化器状态）+ vLLM 实例同时常驻**。9.3 GB + 6 GB ≈ 15.3 GB > 12 GB，**必然 OOM**。即使把 `vllm_gpu_memory_utilization` 压到 0.25（≈3 GB，只够放权重），合计 12.3 GB 仍然超。→ **照抄默认配置在 12GB 单卡上跑不起来。**

### 7.2 结论：**不能完整做完，只能做缩水子集**

判定：**只能部分完成（约 1/4 ~ 1/3 题量）**。理由分三层：

**① 是「双卡常驻」的架构问题，不是「模型太大」的问题。**
1.5B 在 bf16 下只有 3 GB，单卡 12 GB 装得下模型本身，也不需要张量并行（代码里 `tensor_parallel_size=1`）。卡住的是参考实现把「训练进程」和「vLLM 推理进程」硬编码到 `cuda:0` / `cuda:1` 并要求同时存活。
→ **单卡必须改成「分时复用」**：先起 vLLM 生成全部 rollout → 释放 vLLM / 让 vLLM sleep → 再做前向反向 → 再唤醒 vLLM。这需要改 `grpo.py` / `grpo_experiments.py`（当前代码是「边训边生成、每步 `load_weights` 全量同步」的紧耦合循环）。**这是本次测绘给出的最重要工程建议。**

**② GRPO 的主要成本是 rollout 生成，而 4070S 的生成吞吐是数量级短板。**
原文自己就点明：「在 LLM 训练中，生成数据（Rollout/Inference）的成本极其高昂……相比之下，反向传播（训练）的速度要快得多。」
- 一次 200 步 GRPO = 51,200 条 × 最长 1024 token ≈ **最多 5200 万 token 的生成量**。
- 4070 Super：504 GB/s 带宽，约为 H100 SXM（3.35 TB/s）的 **1/6.6**；bf16 张量算力约为 H100 SXM 的 **1/7**；再加上只有 **1 张卡**而参考实现用 2 张，且要分时复用（生成与训练无法重叠）。
→ 综合下来，**4070S 的有效吞吐大约是「2×H100 参考配置」的 1/15 ~ 1/25**。这是**估计**，没有实跑数据，误差可能有 2 倍。

**③ 平台问题（我的知识，非抓取内容，需你自行确认）：**
- **vLLM 官方不支持原生 Windows，需要 Linux 或 WSL2。** 你的 4070S 机器是 Windows（见 AGENTS.md 里的待办「宿舍 Windows 版本」）→ 必须上 WSL2 或双系统 Linux。
- `flash-attn==2.7.4.post1` 是**源码编译**（`pyproject.toml` 里 `no-build-isolation-package = ["flash-attn"]`），Windows 下编译极其麻烦；WSL2 下也需要 CUDA toolkit + 1~3 小时编译。
- 好消息：**4070 Super 是 Ada（sm_89），支持 bf16、支持 flash-attn 2.7.4**——比你的另一台 T400（Turing sm_75、4GB、无 bf16/FlashAttention）强得多。**vLLM 属于 4070S，不属于 T400**，这点与 AGENTS.md 的既有判断一致。
- `bitsandbytes` 在 Windows 上支持较差（WSL2 内没问题）。

### 7.3 具体建议：4070S 上的可行方案（估计）

**必须改的 4 处配置：**
1. `device_policy = device_eval = "cuda:0"`（同卡），并把 vLLM 与训练改成**分时复用**。
2. `rollout_batch_size` 从 256 降到 **32~64**（`n_prompts_per_rollout` 从 32 降到 4~8，`group_size` 保持 8）。
3. `train_batch_size` / `grad_accum_steps` 缩小到 micro-batch = 1（例如 `train_batch_size=8, grad_accum_steps=8`），并**加上 `gradient_checkpointing_enable()`**（参考实现没有加）。
4. `gen_max_tokens` / `eval_max_tokens` 从 1024 降到 **256~512**；`eval_max_examples` 从 256（或全量 1319）降到 **128~256**。
   另：`vllm_gpu_memory_utilization` 用 0.9（单跑时），或改用 vLLM 的 sleep/wake 机制（vLLM 0.7.x 起提供 `enable_sleep_mode`，**此点我未从抓取内容确认，需实测**）。

**注意：以上任何缩小都会让数字与参考实现不可比**（`rollout_batch_size`=256 是官方硬约束之一：`grpo_off_policy_sweep` 原文要求「固定 `rollout_batch_size = 256`」）。要做 `grpo_off_policy_sweep` 这道 4 分题，你必须保持 256，那在 12GB 单卡上基本不可行（→ **建议直接放弃这题**）。

---

## 8. 【我的估计】需要多少小时

> 以下全为**我的估算**，无实跑数据。基准：4070S 有效吞吐 ≈ 参考 2×H100 配置的 1/15 ~ 1/25，再按你「每天有效 6~8 小时 + 大量调试返工」折算。

### 8.1 环境与前置（必花，省不掉）

| 项目 | 估计 |
|---|---|
| WSL2/Linux + CUDA 12.4 + torch 2.5.1 + vLLM 0.7.2 + flash-attn 源码编译 + 依赖冲突排查 | **6 – 12 h**（其中 flash-attn 编译 1–3 h） |
| 从官方仓库取回 2025 版 `tests/` + `adapters.py` 并跑通 | **4 – 8 h** |
| 若你要自己实现 `sft_helper.py` / `gpro_helper.py`（而不是抄本仓库的现成实现） | **+8 – 15 h** |
| 模型 + 数据下载（含 HF 网络问题） | **1 – 3 h** |

### 8.2 实验本身（估计）

| 任务 | 单次/单组估计 | 全量题量 | 全量估计 |
|---|---|---|---|
| 零样本基线（GSM8K 1319 条，vLLM 单卡） | 0.5 – 1 h | 1 | **0.5 – 1 h** |
| SFT 单次（512~2048 条，+ 每 5 步全量评测） | 1 – 3 h | 6 规模 × 2 数据版本 = 12 | **15 – 35 h** |
| EI 单组（5 轮 rollout+过滤+训练+评测） | 4 – 8 h | 8 组 | **35 – 65 h** |
| GRPO 训练循环跑通（50 步） | 3 – 6 h | 1 | **3 – 6 h** |
| GRPO 实验矩阵（≈20 次 50 步 + 1 次 400 步排行榜） | 3 – 6 h/次 | ≈21 | **65 – 125 h** |
| writeup + 全部曲线图 | — | — | **8 – 15 h** |

**全量照搬合计：≈ 140 – 270 h**（≈ 把 4070S 连跑 6–11 天不关机；按每天有效 7 h 计 → **20 – 39 个工作日**）。

### 8.3 我推荐的裁剪版（估计 **30 – 50 h**）

只做这些，保留「可写进简历的完整数字链」：
1. 零样本基线（GSM8K/或自行下 MATH 12K 的 500 条子集）→ 0.5 h
2. SFT 3 档规模（128 / 512 / 2048），只看曲线趋势，评测子集 200 条 → 4 – 8 h
3. EI 1 组（Db=512, G=4, Ep=4, 5 轮）→ 4 – 8 h
4. GRPO 训练循环 1 组（50 步，缩小 batch）→ 3 – 6 h
5. GRPO 学习率 3 档（1e-5 / 2e-5 / 5e-5，各 30 步）→ 6 – 12 h
6. 提示消融 2 档（r1_zero vs question_only，各 30 步）→ 4 – 8 h
7. 曲线 + writeup → 6 – 10 h

**放弃**：`grpo_off_policy_sweep`（4 分，要固定 rollout=256）、`leaderboard`（16 分，需要 2×H100 / 4h 的硬件条件）、以及全部可选的 safety/RLHF 补充作业、Llama-3.1-8B / Llama-3.3-70B 部分。

---

## 9. 【我的估计】数据下载量与磁盘需求

> 除注明外均为**估算**，未经实测。

| 项目 | 来源（原文） | 估计大小 |
|---|---|---|
| `Qwen2.5-Math-1.5B` 权重（safetensors, bf16） | HF `Qwen/Qwen2.5-Math-1.5B`（原文给的路径为集群路径 `/data/a5-alignment/models/Qwen2.5-Math-1.5B`） | **≈ 3.1 GB** |
| GSM8K test + train（本仓库替代评测集） | HF `openai/gsm8k`；代码读 `data/gsm8k/test.jsonl`（原文实测加载 **1319** 条） | **≈ 5 – 10 MB** |
| 带推理轨迹的 SFT 训练数据 | HF `garg-aayush/sft-cs336-assign5-datasets`（`sft-reason`）/ `hiyouga/math12k`（原文） | **≈ 50 MB – 1 GB**（很依赖是否含完整 gpt-oss 推理轨迹，**不确定**） |
| MATH 12K（可选，官方原版数据集） | HF `qwedsacf/competition_math`（原文给的链接） | **≈ 5 – 30 MB** |
| Python 环境（torch 2.5.x + cu12x、vLLM 0.7.2 及其依赖、xformers、flash-attn 编译产物） | `uv sync` | **≈ 10 – 18 GB** |
| `uv.lock` / 仓库本体 | 仓库 | **< 1 MB**（uv.lock 489 KB） |
| 实验产物：`results/` jsonl + wandb 日志 | 脚本自动生成 | **≈ 1 – 5 GB** |
| Checkpoint（每个 GRPO 运行存 `results/grpo/latest` ≈ 3 GB；EI 每轮存 `.../latest` ≈ 3 GB） | 代码原文 `policy.save_pretrained(...)` | **≈ 40 – 60 GB**（EI 8 组 × 3 GB + GRPO 若干） |

**建议预留磁盘：**
- **最小可跑（裁剪版）：≥ 40 GB**
- **想跑较完整：≥ 120 GB**（环境 18 + 模型 3 + 数据 1 + checkpoint 60 + 结果 5 + 余量）

---

## 10. 核心必做 vs 可跳过（逐项）

判定依据：官方分值（原文）+ 是否带 H100 小时（原文）+ 在 12GB 单卡上的可行性（【估计】）。

### 10.1 必做（P0：不做等于没做作业）

| 项 | 分值 | 理由 |
|---|---|---|
| `math_baseline`（`evaluate_math.py` + `evaluate_vllm`） | 4 | 一切后续对比的基线；单卡完全可跑 |
| `tokenize_prompt_and_output` | 2 | 后续全部训练/评分的地基；`pytest -k` 必过 |
| `compute_entropy` | 1 | 必过测试 |
| `get_response_log_probs` | 2 | 必过测试；GRPO 的 old_log_probs 也靠它 |
| `masked_normalize` | 1 | 必过测试 |
| `sft_microbatch_train_step` | 3 | 必过测试 |
| `compute_group_normalized_rewards` | 2 | 必过测试 |
| `compute_naive_policy_gradient_loss` | 1 | 必过测试 |
| `compute_grpo_clip_loss` | 2 | 必过测试 |
| `compute_policy_gradient_loss` | 1 | 必过测试 |
| `masked_mean` | 1 | 必过测试 |
| `grpo_microbatch_train_step` | 3 | 必过测试 |
| `grpo_train_loop` | 5 | GRPO 全部实验的入口 |
| `sft_experiment` | 2 (2 H100·h) | 单卡可跑，缩规模 |
| `grpo_learning_rate` | 2 (6 H100·h) | 缩到 3 档 × 30 步可跑 |
| 书面问题（`think_about_length_normalization` 1 分等） | 1+ | 零算力，性价比最高；用参考实现在 GSM8K 上的数字也能写 |
| `writeup.pdf` + `code.zip` | — | 官方唯一提交物 |

→ **必做集合估计 ≈ 30–40 h（4070S）**

### 10.2 应做（P1：能做则做，做不了也不致命）

| 项 | 分值 | 单卡判断 |
|---|---|---|
| `log_generations` | 1 | 纯 CPU/日志，必做（算力 0） |
| `expert_iteration_experiment` | 2 (6 H100·h) | **只做 1 组**（Db=512, G=4, Ep=4）；8 组全跑 35–65 h，不值 |
| `grpo_baselines` | 2 (2 H100·h) | 2 次 50 步，可缩到 30 步，**可做** |
| `grpo_length_normalization` | 2 (2 H100·h) | 同上，**可做** |
| `grpo_group_standard_deviation` | 2 (2 H100·h) | 同上，**可做** |
| `grpo_prompt_ablation` | 2 (2 H100·h) | 同上，**可做**（这是最能讲出「现象」的一题） |
| `grpo_off_policy`（实现部分） | — | **代码改造可做，实跑可只跑极短验证** |
| `grpo_off_policy_clip_ablation` | 2 (2 H100·h) | 短跑可做 |

→ **应做集合估计 ≈ +30–60 h**

### 10.3 建议放弃（P2：单卡不现实或非强制）

| 项 | 分值 | 放弃理由 |
|---|---|---|
| `grpo_off_policy_sweep` | 4 (12 H100·h) | 原文硬性「固定 `rollout_batch_size = 256`」，12GB 单卡装不下这个 batch 的生成 + 训练共存；且要 wide sweep(<50步) + focused sweep(200步) 两轮 |
| `leaderboard` | **16** (16 H100·h) | 原文硬约束「**2 个 H100 GPU 上 4 小时**」+「**整个** MATH 验证集 5K 条」+ vLLM `temperature 1.0 / max_tokens 1024`。**这是 4070S 的物理不可能项**（2×H100 4h ≈ 8 H100·h；按 1/20 折算 ≈ 160 小时单卡） |
| 可选补充作业（safety / RLHF / instruction tuning，`cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`） | 非强制 | 官方原文「完全可选」；且本仓库两个相关 prompt（`alpaca_sft.prompt`、`zero_shot_system_prompt.prompt`）在**所有抓取到的 `.py` 中均未被引用**，配套代码与数据都没有 |
| Llama-3.1-8B Base / Llama-3.3-70B Instruct 相关实验 | 非强制 | 原文明确标注「用于**可选**的指令微调实验」；8B 全参 SFT 在 12GB 上不可能，70B 更不可能 |
| 全量 SFT sweep（12 次，含每 5 步全量 1319 条评测） | 2 | 15–35 h 换来一条曲线，性价比低；用 3 档代替 |
| 全量 EI 8 组 | 2 | 35–65 h；用 1 组代替 |

---

## 11. 额外发现（原文事实，值得注意）

1. **`bitsandbytes` 未声明但被 import**：`grpo.py` 与 `grpo_experiments.py` 都有 `import bitsandbytes as bnb` 并调用 `bnb.optim.AdamW8bit`，但 `pyproject.toml` 的 `dependencies` 列表里**没有 `bitsandbytes`**。照 `uv sync` 装完环境后跑 GRPO 脚本会 `ModuleNotFoundError`。（原文对比：`pyproject.toml` 依赖全文见下）
2. **模型路径硬编码私有路径**：所有脚本的 `MODEL_PATH` / `BASE_MODEL` / `base_model` 都是 `/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B`（作者自己的机器）。README.md 已提示需要改。
3. **数据路径文档/代码不一致**：`README.md` 写 `data/math12k/sft_gpt-oss-120b.jsonl` 与 `data/sft/sft_gpt-oss-120b_filtered.jsonl`；`sft_math_reasoning.py` 用的却是 `data/sft/sft_gpt-oss-120b.jsonl`。
4. **README.md 自述与代码不符**：README 说 `sft_helper.py` / `gpro_helper.py` 是「你需要实现的核心函数」，但这两个文件里全是完整实现。
5. **两个 prompt 模板孤立**：`alpaca_sft.prompt`、`zero_shot_system_prompt.prompt` 在我抓取的所有 `.py` 中都没有被引用。
6. **`sft_math_reasoning.py` 的最终模型保存被注释掉了**（`# model.save_pretrained(final_save_dir)`），SFT 产物不落盘，只留下 `results.jsonl` 与 wandb 曲线。
7. **`gpro_helper.py` 与 `sft_helper.py` 各有一个同名但签名不同的 `masked_normalize`**（前者 `constant_normalizer`，后者 `normalize_constant`）—— 抄实现时极易搞错。
8. **官方版本已迭代到 Spring 2026**，测试钩子名与 2025 handout 不同（见 §4.3）。要对齐 2025 必须 checkout 官方对应 tag/commit。
9. **`README.ipynb` 中 GRPO 的 `USE_STD_NORMALIZATION = True`（§4.6 代码）与 `GRPOConfig.use_std_normalization = False`（§5.1 / `grpo_experiments.py`）默认值相反**，两处原文不一致。

### `pyproject.toml` 依赖全文（原文）
```toml
[project]
name = "alignment"
version = "1.0.0"
description = "CS 336 Spring 2025 Assignment 5: Alignment"
requires-python = ">=3.11,<3.13"  # Python 3.13 not yet supported for some deps
dependencies = [
    "accelerate>=1.5.2",
    "alpaca-eval",
    "flash-attn==2.7.4.post1",
    "jupyter>=1.1.1",
    "math-verify[antlr4-13-2]>=0.7.0",
    "pylatexenc==2.10",
    "notebook>=7.4.2",
    "pytest>=8.3.5",
    "torch",
    "tqdm>=4.67.1",
    "transformers>=4.50.0",
    "typer>=0.15.4",
    "vllm==0.7.2",
    "wandb>=0.19.8",
    "xopen>=2.0.2",
]

[tool.setuptools.packages.find]
include = ["cs336_alignment"]

[tool.uv]
package = true
no-build-isolation-package = ["flash-attn"]

[tool.uv.sources]
alpaca-eval = { git = "https://github.com/nelson-liu/alpaca_eval.git", rev = "forward_kwargs_to_vllm" }

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "WARNING"

[[tool.uv.dependency-metadata]]
name = "flash-attn"
version = "2.7.4.post1"
requires-dist = ["torch", "einops", "setuptools"]
```
**要点（原文）**：`vllm==0.7.2` 硬钉、`flash-attn==2.7.4.post1` 硬钉且需源码编译、`torch` 未钉版本（由 vLLM 0.7.2 间接约束）、**无 `peft` / 无 `trl`**（→ 官方路线是 **1.5B 全参微调，不是 LoRA/QLoRA**）、**无 `bitsandbytes`**（但代码要用）。

---

## 12. 四个 prompt 模板（原文逐字）

**`cs336_alignment/prompts/r1_zero.prompt`**
```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

**`cs336_alignment/prompts/question_only.prompt`**
```
{question}
```

**`cs336_alignment/prompts/zero_shot_system_prompt.prompt`**
```
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```
（原文末尾确为一个未闭合的 ```` ``` ````）

**`cs336_alignment/prompts/alpaca_sft.prompt`**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

---

## 13. 抓取失败项（全部，不省略）

| # | URL | 失败原因 |
|---|---|---|
| 1 | `https://stanford-cs336.github.io/spring2025-assignment5-alignment/` | **跨域重定向未被跟随**：工具报 `cross-origin redirect to http://cs336.stanford.edu is not followed automatically`。→ 官方 2025 作业页**未抓到**。 |
| 2 | `https://cs336.stanford.edu/spring2025-assignment5-alignment/` | **HTTP 404**（`Page not found · GitHub Pages`）。作为 #1 的替代重试也失败。→ **原版 2025 作业页面彻底未抓到**；只能用官方 GitHub 仓库 README + `tests/adapters.py` 作为对照（且那是 **Spring 2026 版**）。 |
| 3 | `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment5-alignment/README.ipynb` | **被抓取工具 120 KB 输出上限截断**。抓到 120,198 字节 / 3,492 行，正文止于第 5.6 节代码块中段（`"# 3. 计算裁剪后的目标函数 (Surrogate 2)"`）。**第 5.7 离线策略扫描结果、5.8 裁剪消融结果、5.9 提示消融结果、5.10 排行榜结果与最终分数未抓到。** 已确认**不是仓库文件缺失**（文件真实大小 131,926 字节，API 可查），而是抓取截断。 |
| 4 | `https://cdn.jsdelivr.net/gh/datawhalechina/diy-llm@main/coursework/assignment5-alignment/README.ipynb` | 作为 #3 的换源重试，**同样被截断在同一位置**（120,192 字节 / 3,492 行）。换源未能绕过上限。 |
| 5 | `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment5-alignment/README.pdf` | 工具报 `unsupported content type "application/octet-stream"`。该 PDF（181,475 字节）**很可能是同一份 notebook 的完整导出**，本可补上 #3/#4 的缺失段落，但**无法解析，抓取失败**。 |
| 6 | `https://api.github.com/repos/datawhalechina/diy-llm/contents/coursework/assignment5-alignment/images` | **HTTP 403 `API rate limit exceeded`**（未认证额度用尽，IP 58.251.166.51）。→ **`images/` 下约 40 个 png 的具体文件名未抓到**（正文里只能引用 `README.ipynb` 内嵌 `<img src="images/...">` 里出现过的那批文件名）。 |
| 7 | 本地 `pwsh Invoke-WebRequest https://raw.githubusercontent.com/...`（用于绕过 #3/#4 的 120 KB 上限、分段解析 notebook 尾部） | **失败**：`基础连接已经关闭: 接收时发生错误。` 本会话沙箱下 pwsh 无网络访问，无法用本地脚本下载后分段读取。 |

**未尝试抓取（非失败，仅说明）**：`README.md` 里给出的 3 个 wandb 报告链接（`wandb.ai/xuhu0115-sju/cs336-a5-sft-v3` / `cs336-a5-sft-ei` / `cs336-a5-grpo`）未抓取——它们是前端渲染页面且需要 accessToken，不在你给出的抓取清单内。**`uv.lock`（489,119 字节）未抓取**，因此 `torch` 的具体解析版本未能从原文确认。

---

## 抓取来源（本次实际成功的 URL）

1. `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment5-alignment/README.md`
2. `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment5-alignment/cs336_spring2025_assignment5_alignment_zh.md`
3. `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment5-alignment/pyproject.toml`
4. `.../cs336_alignment/sft_helper.py`
5. `.../cs336_alignment/gpro_helper.py`
6. `.../cs336_alignment/sft_math_reasoning.py`
7. `.../cs336_alignment/sft_math_reasoning_ei.py`
8. `.../cs336_alignment/grpo.py`
9. `.../cs336_alignment/grpo_experiments.py`
10. `.../cs336_alignment/evaluate_math.py`
11. `.../cs336_alignment/drgrpo_grader.py`
12. `.../cs336_alignment/log.py`
13. `.../cs336_alignment/prompts/{r1_zero,question_only,zero_shot_system_prompt,alpaca_sft}.prompt`
14. `.../coursework/assignment5-alignment/README.ipynb`（**截断至 §5.6**）
15. `https://api.github.com/repos/datawhalechina/diy-llm/contents/coursework/assignment5-alignment`（目录清单）
16. `https://api.github.com/repos/datawhalechina/diy-llm/contents/coursework/assignment5-alignment/cs336_alignment`（目录清单）
17. `https://raw.githubusercontent.com/stanford-cs336/assignment5-alignment/main/README.md`（**Spring 2026** 外部对照）
18. `https://raw.githubusercontent.com/stanford-cs336/assignment5-alignment/main/tests/adapters.py`（**Spring 2026** 外部对照）
19. `https://kjore.github.io/2026/02/04/cs336作业五要求/`（第三方翻译，用于校对 H100 小时标注）
20. `https://blog.csdn.net/weixin_43807749/article/details/156726594`（第三方翻译+实现，用于校对 H100 小时标注与 `tests/conftest.py` 的存在）
