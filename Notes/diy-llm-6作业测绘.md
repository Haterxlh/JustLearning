# datawhalechina/diy-llm（CS336 中文共建版）6 个作业测绘

> 抓取日期：2026-09（本 session）
> 方法：GitHub API 文件树（recursive=1，commit `3afe8d3883167c1417d796f50c24e1f039f34a01`）+ raw.githubusercontent.com 原文抓取 + 在线阅读站。
> **标注约定**：
> - 【原文】= 直接从仓库抓到的字面内容
> - 【估计】= 我基于抓到的原文做的计算/推断，可能有误
> - 【抓取失败】= 明确没抓到，不猜

---

## 0. 先纠正任务给定路径：路径是真的

【原文】GitHub API 文件树确认，`coursework/` 下 6 个作业目录**英文命名，与任务给定完全一致**：

```
coursework/assignment1-basics
coursework/assignment2-systems
coursework/assignment3-scaling
coursework/assignment4-data
coursework/assignment5-alignment
coursework/assignment6-evaluation
```

共 270 条 `coursework/` 路径。**没有中文目录名**（中文只出现在子文件名里，如 `相关文档/【导读】...md`、`配置.text`、`作业1.ipynb`）。

### 0.1 一个重大结构性发现：这不是 Stanford 原版脚手架

【原文】原版 `stanford-cs336/assignment1-basics` 的 README 明确写：

```
### Run unit tests
uv run pytest
Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the functions in ./tests/adapters.py.
```

【原文】而 datawhale 版 **`coursework/assignment1-basics/` 下根本没有 `tests/` 目录，也没有 `test_and_make_submission.sh`**。
【原文】`model.py`（14,356 字节）与 `train.py`（14,501 字节）是**完整可运行的参考实现，全文件没有任何 `raise NotImplementedError`**。

→ 【估计】datawhale 版作业1 的定位是「**读得懂 + 跑得通 + 做消融**」的教学复现，不是原版那种「填空 + pytest + leaderboard」的硬作业。
→ 【估计】这意味着它的实际难度**远低于**原版 CS336 作业1。这一点直接决定了下文对「20 小时能否做完」的判断。

同理，`assignment2-systems` 与 `assignment4-data` **有** `tests/` + `test_and_make_submission.sh`（保留了原版骨架）；
`assignment3-scaling`、`assignment5-alignment`、`assignment6-evaluation` **没有** `tests/`。

---

## 1. 作业1：assignment1-basics（手搓大模型）—— 最关键

### 1.1 真实文件树（【原文】，GitHub API 核实）

```
Assignment1_Ablations/SiLU/PPL.md                      (272 B)
Assignment1_Ablations/SiLU_model.py                    (7448 B)
Assignment1_Ablations/SiLU_train.py                    (14077 B)
Assignment1_Ablations/data.in                          (13,766,960 B)
Assignment1_Ablations/no_RMSNorm/PPL.md                (140 B)
Assignment1_Ablations/not_RMSNorm_model.py             (7560 B)
Assignment1_Ablations/not_RMSNorm_train.py             (14317 B)
Assignment1_Ablations/original_model/PPL.md            (270 B)
Assignment1_Ablations/post_Norm/PPL.md                 (273 B)
Assignment1_Ablations/post_Norm_model.py               (7582 B)
Assignment1_Ablations/post_Norm_train.py               (14091 B)
Assignment1_Ablations/tokenizer.json                   (3,571,373 B)
Assignment1_Basics.pdf                                 (419,506 B)
CS336_Assignment1_BPE.ipynb                            (20,986 B)
CS336_Assignment1_Transformer.ipynb                    (93,686 B)
README.md                                              (5,461 B)
TinyStoriesV2-GPT4-valid.txt                           (22,502,601 B)   ← 仓库自带！
bpe_tokenizer/tokenizer.json                           (3,822,009 B)    ← 仓库自带！
ckpt/PPL.md                                            (270 B)
get_train_data.py                                      (2,601 B)
model.py                                               (14,356 B)
train.py                                               (14,501 B)
```
**注意**：仓库里**没有** `TinyStoriesV2-GPT4-train.txt`（README 说因文件大小限制需自行下载），也**没有** `data.bin`（需自己生成）。

### 1.2 要写哪些模块 / 函数

【原文】`model.py` 已包含的类/函数（全部实现完毕，非填空）：
- `class Linear(nn.Module)` / `class Embedding(nn.Module)` / `class RMSNorm(nn.Module)`
- `class SwiGLU(nn.Module)`（d_ff = round_up(8/3·d_model, 64)）
- `class RotaryPositionalEmbedding(nn.Module)`：`_build_cache`, `forward`
- `class MultiHeadAttention(nn.Module)`（优先用 `torch.nn.functional.scaled_dot_product_attention` + `is_causal=True`，否则手写因果 mask）
- `class TransformerBlock(nn.Module)`（pre-norm：`x + attn(norm1(x))`，`x + ffn(norm2(x))`）
- `class TransformerLM(nn.Module)`：`forward`, `generate`
- `class CustomAdamW(torch.optim.Optimizer)`：`step`（手写 bias correction + decoupled weight decay）
- `class SimpleTokenizer`
- `def generate_with_sampling(model, idx, max_new_tokens, temperature, top_k, top_p)`
- `def decode_generated_text(model, tokenizer, prompt, ...)`

【原文】`train.py` 已包含的类/函数：
- `class CustomAdamW`（与 model.py 重复定义）
- `def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c)`（warmup + 余弦退火）
- `def run_gradient_clipping(params, max_norm, eps=1e-6)`
- `class CausalMemmapDataset(Dataset)`：`__init__`, `__len__`, `__getitem__`（np.memmap + int32）
- `def save_ppl_curve(train_ppls, val_ppls, save_dir)`
- `def save_checkpoint(path, model, optimizer, iteration, epoch, config)`
- `def get_memory_usage(device)`
- `def main()`

【原文】`get_train_data.py` 里的函数：
- `def load_tokenizer(tokenizer_json_path)`
- `def text_to_token_ids(text, tokenizer, add_eos=True)`
- `def build_random_data_bin(input_txt, tokenizer_json, output_bin, target_samples=10000, dtype=np.int32)`
- `__main__` 调用：`target_samples=800000`，`input_txt="TinyStoriesV2-GPT4-train.txt"`

→ 【估计】真正需要「手搓」的部分在两个 notebook（`CS336_Assignment1_BPE.ipynb` 手写 BPE，`CS336_Assignment1_Transformer.ipynb` 手写 Transformer），加上 3 个消融实验。`model.py`/`train.py` 的直接作用是**对照答案 + 跑通**。

### 1.3 交付物

【原文・README 代码结构段】交付物 = 可训练代码 + checkpoint + 困惑度曲线图：
- `data.bin`（处理后的训练数据）
- `ckpt/epoch_{epoch}.pt`（每 epoch 自动保存，含 model/optimizer/iteration/epoch/config）
- `ckpt/train_ppl.png`、`ckpt/val_ppl.png`（每 epoch 重画）
- 3 个消融实验的 `PPL.md`

【重要・原文核实】**`PPL.md` 里只有图片链接，没有任何数字**。`ckpt/PPL.md`、`Assignment1_Ablations/SiLU/PPL.md`、`.../no_RMSNorm/PPL.md` 抓到的内容全部只有
`<img ... src="https://github.com/user-attachments/assets/...">`。
→ 【估计】仓库**没有公开可读的最终 PPL 数值**，只有曲线图。想拿到数字必须自己训练。

### 1.4 官方测试 / 评分方式

- 【原文】**没有 pytest**，没有 `tests/`，没有 `test_and_make_submission.sh`，没有 CI 提交脚本，没有 leaderboard。
- 【估计】验收方式是「脚本跑通 + 产出曲线图 + 消融对比」，即**自证式**，没有自动化评分。

### 1.5 数据需求

【原文・README】训练数据从 HuggingFace 下：`https://huggingface.co/datasets/roneneldan/TinyStories/tree/main` 里的 `TinyStoriesV2-GPT4-train.txt`。
【原文・官方 CS336 README】确切下载 URL：
```
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
【原文】仓库自带 `TinyStoriesV2-GPT4-valid.txt` = **22,502,601 字节（约 21.5 MiB）**。
【抓取失败】`TinyStoriesV2-GPT4-train.txt` 的**确切字节数没能从 HuggingFace 直接抓到**：`huggingface.co`、`api/datasets/.../tree/main`、`datasets-server.huggingface.co` 全部 fetch failed / 被解析到非公网 IP。
→ **但我从 `CS336_Assignment1_BPE.ipynb` 的运行输出里拿到了实测值（见 1.12）**：处理到 **1,239.9 MB** 时训练完成 → **train.txt ≈ 1.2 GB**。

### 1.12 【原文】BPE notebook 的实际内容 —— 决定性事实

我完整抓取了 `CS336_Assignment1_BPE.ipynb`（20,986 B，全文）。

**它是一份完整实现，没有任何填空、没有 `NotImplementedError`、没有 `TODO`。** 含以下完整函数：
- `find_chunk_boundaries(file, desired_num_chunks, split_special_token)`（4096 字节窗口向后搜分隔符）
- `iter_text_chunks_with_monitor(file_path, chunk_size=1_000_000, log_every=5)`
- `get_memory_mb()`、`log_status(prefix, bytes_processed, start_time)`
- `train_bpe_tokenizer(train_file, val_file=None, vocab_size=50257, num_chunks=8, output_dir="./bpe_tokenizer")`
- `analyze_tokenizer(tokenizer, texts)`、`load_stories(file_path, num_samples=None)`

**关键：BPE 用的是 Rust `tokenizers` 库的 `BpeTrainer`，不是纯 Python 手写 BPE。**
【原文】`from tokenizers import Tokenizer` / `from tokenizers.models import BPE` / `from tokenizers.trainers import BpeTrainer` / `ByteLevel` / `NFKC`；
`trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)`；
`special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]`；`tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)`。
→ 【估计】**这推翻了我先前"从零手写纯 Python BPE 会吃掉 5–15 小时"的担忧**：这份 notebook 根本不要求你手写 BPE 算法，它调用的是 Rust 实现，1.2 GB 语料上跑得很快。

**★ 从 notebook 的 stdout 拿到的实测数字（【原文】，这是全仓库最硬的一组数字）：**
- 语料处理量：日志从 `data=19.1 MB` 一路涨到 **`data=1239.9 MB`**，然后打印 `✅ BPE Tokenizer训练完成`
  → **`TinyStoriesV2-GPT4-train.txt` ≈ 1.2 GB**（此前我估的 1.9–2.1 GB **偏大**，现更正）
- 文本迭代速率：前期 217–230 MB/s，BPE 实际开始消费后降到 **约 19–32 MB/s**
- 训练/验证集故事长度实测：`训练集统计: {'avg_tokens': 197.0, 'max_tokens': 308}`；`验证集统计: {'avg_tokens': 208.2, 'max_tokens': 434}`
  → **平均约 197 token/故事**（我先前估 250–300 偏大）
- tokenizer 验证输出：`['ĠHello', ',', 'Ġworld', '!', 'Ġ', '<|endoftext|>']` → `[8431, 16, 1501, 5, 149, 0]`
- notebook 里 `vocab_size=50257`，**与 `train.py` 的 `--vocab_size 50257` 默认值一致**（解决了 A3 那份 `tokenizer.json` 是 32000 词表的混淆：A1 与 A3 用的是**两份不同的 tokenizer**）
- 【原文】notebook 里训练集路径拼写为 `./TinyStoriesV2-GPt4-train.txt`（原文笔误，`GPt4` 大小写不一致）
- 【原文】`train_bpe_tokenizer` 的 `num_chunks` 形参**未被实际使用**（定义后没传给 `iter_text_chunks_with_monitor`）

→ **对数据量的更正估算**：800,000 故事 × 197 token ≈ **1.58 亿 token**；int32 存储 → `data.bin` ≈ **630 MB**（此前估 0.8–1.0 GB）。
→ **对训练时间的更正估算**：FLOPs = 6 × 30.5e6 × 1.58e8 ≈ **2.9e16 = 29 PFLOPs/epoch**；@20 TFLOPS ≈ **24 分钟/epoch**，5 epoch ≈ **约 2 小时**（@10 TFLOPS 则约 4 小时）。
→ 【估计】**磁盘总占用约 1.9 GB**（train.txt 1.2 GB + data.bin 0.63 GB + 仓库自带资产 0.05 GB）。

【原文】`get_train_data.py` 行为：**两遍扫描** train.txt（第一遍 `sum(1 for _ in f)` 数总行数，第二遍随机抽 800,000 行），把抽中的行 tokenize 后拼成一条长 id 流，以 **int32** 存成 `data.bin`。
→ 【估计】800k 条 TinyStories 故事 × 约 250–300 token/条 ≈ **2.0–2.4 亿 token**；int32 存储 → `data.bin` ≈ **0.8–1.0 GB**。
→ 【估计】磁盘总占用：train.txt ~2 GB + data.bin ~0.9 GB + 仓库自带资产 ~0.05 GB ≈ **3 GB 上下**。

### 1.6 训练规模（【原文】train.py argparse 默认值）

| 参数 | 默认值 |
|---|---|
| epochs | 5 |
| batch_size | 16 |
| context_length | 128 |
| d_model | 256 |
| num_heads | 8 |
| num_layers | 6 |
| vocab_size | 50257 |
| lr / min_lr | 3e-4 / 3e-5 |
| data_path | data.bin |

【原文】其他关键实现细节：
- 优化器 `CustomAdamW(lr=args.lr, weight_decay=0.1)`；损失 `nn.CrossEntropyLoss()`
- `warmup_steps = int(0.1 * total_steps)`；梯度裁剪 `max_norm=1.0`
- 训练/验证切分 `split = 0.8`（即 4:1，与 README 一致）
- **全 fp32 训练，没有 AMP / autocast**
- `os.environ["OMP_NUM_THREADS"] = "1"`、`MKL_NUM_THREADS = "1"`（【估计】这会限制 DataLoader 的 CPU 吞吐，可能成为小显存机型上的实际瓶颈）
- **若 `data.bin` 不存在会自动造 mock 数据**：`dummy_len = context_length * batch_size * 20 = 40960` 个随机 int32 → 脚本开箱即可跑通（但学不到东西）
- checkpoint 写到**硬编码** `ckpt/epoch_{epoch}.pt`（没用 `args.checkpoint_dir`）

### 1.7 参数量与显存估算（【估计】）

按 d_model=256 / layers=6 / heads=8 / d_ff=704（`int(8/3*256)=682` → 补齐到 704）/ vocab=50257、**lm_head 与 embedding 不共享**：

- 每层：SwiGLU `3×256×704 = 540,672`；Attention `4×256×256 = 262,144` → `802,816`
- 6 层 ≈ `4.82 M`
- token embedding `50257×256 ≈ 12.87 M`；lm_head `256×50257 ≈ 12.87 M`
- **合计 ≈ 30.5 M 参数**

fp32 训练显存拆解（batch 16 / ctx 128）：
- 参数 122 MB + 梯度 122 MB + AdamW 两个动量 244 MB ≈ **0.5 GB**
- **真正的大头是 logits**：`16×128×50257 = 1.029 亿` 个 float32 = **412 MB**，加上 CE 反向的中间量与 lm_head 梯度
- 【估计】峰值显存约 **2–5 GB**

→ **结论【估计】：12GB 完全够用，余量很大。** 可以放心把 batch_size 提到 32–64，或 context_length 提到 256–512。

### 1.8 时间估算（【估计】）

- 每 step FLOPs ≈ `6 × 30.5e6 × (16×128) = 3.75e11` = 375 GFLOPs
- 每 epoch：train blocks ≈ `2.2e8 × 0.8 / 128 ≈ 137.5 万` blocks ÷ 16 ≈ **8.6 万 step** → ≈ **3.2e16 FLOPs/epoch ≈ 32 PFLOPs/epoch**
- 4070 Super FP32 峰值约 35 TFLOPS；无 AMP 的实际利用率保守取 10–20 TFLOPS
  - @20 TFLOPS → 约 **27 分钟/epoch** → 5 epoch ≈ **2.3 h**（+20% 验证 ≈ 2.7 h）
  - @10 TFLOPS → 约 **53 分钟/epoch** → 5 epoch ≈ **4.4 h**（+验证 ≈ 5.3 h）
- 数据准备附加：扫+tokenize 1.9GB / 800k 行 → 【估计】15–40 分钟
- **不需要**训练 BPE：`bpe_tokenizer/tokenizer.json` 仓库自带

→ **总估算【估计】：3–7 小时**（不含从零手写 BPE/Transformer 的学习时间）。

### 1.9 【核心结论】assignment1 能否在 12GB 单卡 20 小时内做完？

**能，而且时间相当宽裕。【估计】**

三条理由（全部基于上面抓到的原文，不是泛泛而谈）：
1. **模型极小**：默认配置 ≈ 30.5M 参数、context 128、batch 16。峰值显存【估计】2–5 GB，12GB 有 2–5 倍余量。这不是「勉强能跑」，是「随便跑」。
2. **代码已经写好了**：`model.py` / `train.py` 是完整实现，**零 `NotImplementedError`**，且 `data.bin` 缺失时会自动造 mock 数据开箱即跑。作业1 在这个仓库里的形态是「读懂 + 复跑 + 做 3 个消融」，不是「填空 + 过隐藏测试」。
3. **tokenizer 也自带了**：`bpe_tokenizer/tokenizer.json`（3.82 MB）已在仓库里，可以完全跳过 BPE 训练。

**唯一的真实风险**（【估计】）：如果坚持「从零手写纯 Python BPE 并在 1.9GB 全量语料上训练 tokenizer」，那是最耗时的一环，可能单独吃掉 5–15 小时。**跳过它（用自带 tokenizer.json）或只在 20MB 的 valid.txt 上训 BPE，是把这个作业压进 20 小时的关键决定。**

### 1.10 「最小可完成版本」（只做哪几块也算跑通）

【估计】按性价比排序：

**核心必做（【估计】约 8–12 h）**
1. 环境跑通：`pip install torch numpy transformers tokenizers tqdm psutil matplotlib`
2. 拿到 `bpe_tokenizer/tokenizer.json`（仓库自带，**跳过 BPE 训练**）
3. 造数据：用**仓库自带的 `TinyStoriesV2-GPT4-valid.txt`（22.5 MB）**当语料，改 `get_train_data.py` 的 `input_txt=` 指向它、`target_samples` 调小（如 5 万）→ 生成一个小 `data.bin`。**这样完全不需要下 1.9GB 的 train.txt。**
4. 读懂并跑通 `model.py` + `train.py`（默认配置，2–5 epoch）
5. 产出 `ckpt/epoch_*.pt` + `train_ppl.png` / `val_ppl.png`（**这就是作业要求的交付物**）
6. `python model.py` 生成一段文本（证明语言模型真的训出来了）

**强烈建议做（【估计】额外 3–5 h，面试可讲）**
7. 手写版对照：照着 notebook 自己写一遍 `RMSNorm` / `RoPE` / `MultiHeadAttention` / `SwiGLU` / `CustomAdamW`，与 `model.py` 对照
8. 3 个消融里挑 **1 个**（推荐 `no_RMSNorm`，改动最小：把 `norm1/norm2` 去掉）→ 出一张 PPL 对比图

**可跳过**
- ✂️ 从零训练 BPE（自带 tokenizer.json，跳过）
- ✂️ 下载 1.9GB `TinyStoriesV2-GPT4-train.txt`（用自带 valid.txt 代替）
- ✂️ 跑满 800k 样本（`target_samples` 降到 5 万–10 万）
- ✂️ 全部 3 个消融（挑 1 个）
- ✂️ `Assignment1_Basics.pdf` 逐字精读（选读）

### 1.11 作业1 的文档不一致（【原文】核实到的事实）

- README「代码结构」写的是 `CS336_Assignment1_Ablations/`，**实际目录名是 `Assignment1_Ablations/`**
- README 写 `get train data.py`（带空格），**实际文件名是 `get_train_data.py`**（带下划线）
- README 说 `TinyStoriesV2-GPT4-train.txt` 在目录树里，**实际仓库里只有 valid**
- 【原文】README 称 `not_RMSNorm_model.py`「**完全**移除归一化层」，实际 `TransformerLM.norm_final = RMSNorm(d_model)` **仍保留**（只注释掉了 block 内的 `norm1/norm2`）
- 【原文】根 README 称第2章配套作业是「手写 tokenizer 训练代码」，**实际是调用 HuggingFace `tokenizers` 库**
- 【原文】`find_chunk_boundaries` 定义后从未被调用；`train_bpe_tokenizer(..., num_chunks=8)` 的 `num_chunks` 形参未使用而 `__main__` 传了 16
- 【原文】`class CustomAdamW` 在 `model.py` 与 `train.py` 中**被重复定义了两遍**（复制粘贴）

### 1.13 【原文】Transformer notebook 核实 —— 同样没有填空（独立复核）

subagent 对 `CS336_Assignment1_Transformer.ipynb` 完整内容做了穷尽 grep：
**`NotImplementedError` = 0 处；`TODO` / `FIXME` / `YOUR CODE HERE` / 占位 `pass` = 0 处。**
命中的 `test_*` 字样全部是 notebook **自写的测试函数**（`test_weights`、`test_input`、`test_implementation`、`test_gradient_clipping_logic`），不是填空标记。
旁证：code cell 带真实 `execution_count`（1,2,3,…,457–475）与真实输出（`输出形状: torch.Size([1, 2048, 8])`、`掩码功能测试通过`、`验证结果是否一致：True`）→ **被执行过、有真实输出 = 成品，不是骨架。**

【原文】Transformer notebook 的 6 个 markdown cell：架构总览表 + `一、基本板块的构建` + `二、 支持Transformer训练的板块构建` + 3 个概念 Q&A（反向传播 / RMSNorm / RoPE theta）。**无任何「练习/填空/实现要求/你的任务」字样。**
⚠️【原文】其中 RoPE 那一格（「theta是控制旋转频率分布的超参数：theta越大，整体旋转频率」之后）**正文被大段无意义字符污染**（形如 `V69ehW7r3379tqIESNU3WQyaREREdq0...` 的 base64 噪声），**不可作为可信内容引用**。

【原文】明面需自写的组件 **15 个**：
- 第一节（9 个）：`class Linear`(无偏置) / `class Embedding` / `class RMSNorm` / `class SwiGLU` / `class RotaryPositionalEmbedding` / `def Softmax(x, dim)` / `def scaled_dot_product_attention(q,k,v,mask=None)`（**返回 `output, attn_weights`**）/ `class MultiHeadAttention` / `class TransformerBlock`
- 第二节（6+ 个）：`def cross_entropy` / `def calculate_perplexity` / `class AdamW(Optimizer)` / `def get_lr_cosine_schedule` / `def run_gradient_clipping` / `def get_batch` / `def save_checkpoint` / `def load_checkpoint`
- 另加 notebook **没有**、只有 `model.py` 才有的 `class TransformerLM`

### 1.14 ★★★ 三个必须知道的操作性事实（【原文】核实）

**(1) `ckpt/` 目录里没有任何 `.pt` 权重文件。**
【原文】API 实测 `ckpt/` 下**只有 `PPL.md`（270 B）**，无任何 `.pt`。
→ **【估计】直接跑 `python model.py` 只会用随机初始化权重生成文本。** README 说「默认会加载 `ckpt/epoch_5.pt`」，但那个文件**不在仓库里**，必须自己训练产出。

**(2) 存在完全离线的跑通路径**（仓库自带 `Assignment1_Ablations/data.in` 13,766,960 B + `Assignment1_Ablations/tokenizer.json` 3,571,373 B + `TinyStoriesV2-GPT4-valid.txt`）。
【估计】`data.in` = 3,441,740 个 int32 token = 26,888 块（训练 21,510 / 验证 5,378）。
⚠️ **但有一个真实风险**：【原文】`data.in` 是用 `Assignment1_Ablations/tokenizer.json`（3,571,373 B）生成的，而 `train.py` **硬编码**加载 `bpe_tokenizer/tokenizer.json`（3,822,009 B）——两份 tokenizer 大小不同（词表也可能不同）。若不同则 token id 语义错位，**能跑但结果无意义**。

**(3) 三条路径（【估计】）**
| 路径 | 做法 | 估计耗时 | 说明 |
|---|---|---|---|
| **A** | `python train.py --data_path Assignment1_Ablations/data.in --epochs 1`（`--data_path` 可覆盖；tokenizer 已存在） | **20–60 min**，可纯 CPU | 最快拿到第一条 PPL 曲线；注意上面的词表风险 |
| **B** | 改 `get_train_data.py` 的 `input_txt="TinyStoriesV2-GPT4-valid.txt"` + 小 `target_samples` → train.py → model.py | **1–3 h** | **无需下载 train.txt**，推荐 |
| **C** | HF 下 train.txt(1.2GB) → BPE notebook → 80 万样本 → train.py 5 epochs → model.py →（可选）3 组消融 | **6–15 h** | 完整版 |

【估计】单个消融（用 `data.in`，10 epochs，2.753e7 token，token 数确定）：**15–30 min**；3 组约 **0.75–1.5 h**。
【估计】磁盘：`data.bin` 约 630 MB + `epoch_N.pt` 约 366 MB/个 ×5 ≈ 1.83 GB + valid 22.5 MB + tokenizer 3.8 MB ≈ **约 2.5 GB（不含 train.txt）**。

### 1.15 【原文】外部对照：Stanford 原版的 tests 清单（证明确实是两套东西）
Stanford `tests/` 实测：`__init__.py`、`_snapshots/`、`adapters.py`(25,581 B)、`common.py`、`conftest.py`(9,164 B)、`fixtures/`、`test_data.py`、`test_model.py`、`test_nn_utils.py`、`test_optimizer.py`、`test_serialization.py`、`test_tokenizer.py`(16,447 B)、**`test_train_bpe.py`(3,246 B)**
→ **`test_train_bpe.py` 的存在证明 Stanford 要求可测试的从零手写 BPE**，而 diy-llm 删掉了整套 tests。
→ 【原文】Stanford 用 Gradescope 提交、有 late days、允许问 LLM 概念题但禁止直接解题。自学报价（2026-03-28，单张 B200）：Modal $6.25/h、RunPod $4.99/h、Nebius $5.50/h（抢占 $3.05/h）；建议**先在 CPU 调正确性再上 GPU**。
→ 【原文】Stanford 主页**未给 A1 的具体显存/卡数**（数字在未抓到的 PDF 里）。

---

## 2. 作业2：assignment2-systems（系统优化）

### 2.1 【原文】README 核心信息
- 完整描述在 `相关文档/cs336_spring2025_assignment2_systems.pdf`（494,365 B）+ 中文翻译 PDF（35,065,997 B）+ 导读 md + 脑图 jpeg
- `cs336_systems/` 是你要实现的模块（README 说「此文件夹基本上是空的！」，**但实际仓库里已经塞满了 20 个左右实现文件**——又一个文档与实际不一致）
- `作业的全部流程都在 cs336_systems/作业1.ipynb 和 作业2.ipynb`；「只需运行这两个 notebook 就可以跑通流程，其他文件都是这两个文件生成的，不需要理会」

### 2.2 【原文】真实文件树要点
`cs336_systems/`：`flash_attention.py`(6,040 B)、`flash_attention_triton.py`(6,474 B)、`sharded_optimizer.py`(6,220 B)、`ddp_bucket.py`(9,753 B)、`ddp_overlap_individual_parameters.py`(6,559 B)、`ddp_bucketed_benchmark_complete.py`(17,790 B)、`minimal_ddp_flat_benchmarking.py`(8,404 B)、`my_profile.py`(4,605 B)、`distributed_demo.py`、`distributed_communication_single_node_demo.py`、`ddp_model_demo.py`、`test_sharded_optimizer.py`、`memory_snapshot.pickle` 等
`tests/`：`test_attention.py`(3,989 B)、`test_ddp.py`(7,325 B)、`test_ddp_individual_parameters.py`(6,775 B)、`test_sharded_optimizer.py`(2,761 B)、`adapters.py`、`common.py`、`conftest.py`、`fixtures/ddp_test_data.pt`、`fixtures/ddp_test_labels.pt`
另有 `test_and_make_submission.sh`(760 B)、`pyproject.toml`(523 B)、`requirements.txt`(51,928 B)

→ 【原文】**有 pytest**（4 个 test 文件）+ **有 `test_and_make_submission.sh`**，所以有官方提交/评分脚本。

### 2.3 深度核实补充（subagent 报告，【原文】/【估计】已分别标注）

#### 2.3.1 ★ 核心更正：测试**不要求 2 张物理 GPU**
【原文】`tests/common.py` 的 `_setup_process_group`：
```python
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    local_rank = rank % device_count
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
else:
    device = "cpu"
```
- `test_DistributedDataParallelCPU`（6 例）、`test_DistributedDataParallelIndividualParameters`（2 例）、`test_sharded_optimizer`（2 例）全部 `mp.spawn(world_size=2, nprocs=2)` + **`backend="gloo"`**
- `test_attention.py`（6 例）：2 例 CPU，4 例 `skipif(not torch.cuda.is_available())`
- **合计 16 个用例**
→ 【原文】共建方自己的 notebook 就写着：「**如果有多个GPU就可以跑通，只有一个GPU也可以，但是就没有分布式训练的速度优势，但是作为教学学习是足够的**。」
→ 【原文】作者实际环境输出：`Torch version: 2.9.1+cu128` / `CUDA device count: 1` / `Device name: NVIDIA GeForce RTX 4060 Laptop GPU`（8GB）
→ 【估计】单卡上两个 rank 都落到 `cuda:0`；更稳的做法是 `CUDA_VISIBLE_DEVICES=""` 让它退回 CPU+gloo 双进程（正是 `test_DistributedDataParallelCPU` 命名的原意）。**这测的是梯度同步的数值正确性，不是多卡带宽。**

#### 2.3.2 ★ 决定性环境约束：Triton 只有 Linux 版
【原文】`requirements.txt`（`uv export` 生成）逐字：
```
triton==3.5.1 ; platform_machine == 'x86_64' and sys_platform == 'linux'   # via torch
torch==2.9.1   （无平台限制）
```
【原文】`pyproject.toml` 的 dependencies **完全没有 triton**：`einops, einx, ipykernel, jaxtyping, pytest, torch, torchvision, wandb`
【原文】`作业1.ipynb` 逐字：「**由于triton在windows没有可用的版本**……」「整个作业使用**linux或者ubuntu**进行，读者可以使用**windows的wsl**来在windows中跑通」
→ 【原文+原文】**4070 Super 在原生 Windows 上装不上 triton，必须 WSL2。**

#### 2.3.3 要写的函数（【原文】）
唯一被 pytest 检验的接入点：`tests/adapters.py` 的 **8 个函数，全部 `raise NotImplementedError`**：
`get_flashattention_autograd_function_pytorch` / `get_flashattention_autograd_function_triton` / `get_ddp_individual_parameters(module)` / `ddp_individual_parameters_on_after_backward(ddp_model, optimizer)` / `get_ddp_bucketed(module, bucket_size_mb)` / `ddp_bucketed_on_after_backward(ddp_model, optimizer)` / `ddp_bucketed_on_train_batch_start(ddp_model, optimizer)` / `get_sharded_optimizer(params, optimizer_cls, **kwargs)`

仓库预置参考实现（**不是 adapters 的返回值，开箱跑不通测试**）：
- `flash_attention.py`：`FlashAttentionAutograd.forward(ctx, Q, K, V)` / `.backward(ctx, dO)` —— **没有 `is_causal` 参数**，而测试以 `.apply(q,k,v,is_causal)` 调用 → **TypeError**
- `flash_attention_triton.py`：`flash_attention_fwd_kernel(...)` + `FlashAttentionTriton.forward(ctx, Q, K, V, is_causal=False)` —— **只有 forward，没有 backward**，而 `test_flash_backward_triton` 需要 backward
- `sharded_optimizer.py`：`ShardedOptimizer.__init__/_build_local_optimizer/_param_owner/_global_param_index/step/_sync_parameters/add_param_group`
- `ddp_bucket.py`：`TeachingDDPOverlapPerParam`、`GradientBucket`、`TeachingDDPBucketed`（`_create_bucket`/`_create_bucket_hook`/`finalize_gradients`）—— 属性名 `self.model`，而 `tests/common.py::validate_ddp_net_equivalence` 调的是 **`net.module.state_dict()`** → **该文件会挂**
- `ddp_overlap_individual_parameters.py`：`DDPOverlapBucketed`（有 `self.module` ✅）+ `run()`
- `my_profile.py`：`profile()`、`mean()`、`benchmark()`

#### 2.3.4 官方测试与评分方式（【原文】`test_and_make_submission.sh` 全文）
```bash
#!/usr/bin/env bash
set -euo pipefail
uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"
output_file="cs336-spring2024-assignment-2-submission.zip"
rm "$output_file" || true
zip -r "$output_file" . -x '*egg-info*' ... -x '*.txt' -x '*.json' -x '*.bin' -x '*.pt' ...
```
- 测试命令 = `uv run pytest -v ./tests --junitxml=test_results.xml`；**`|| true` 意味着测试失败也不中断打包**
- 打包名写成 **`spring2024`**（复制粘贴遗留）
- 【原文】**没有 modal、没有 leaderboard、没有网络提交**；`compute_leaderboard_score` 与 `modal` 在全部抓取内容中出现 **0 次**
- 【原文】`-x '*.pt'` 会排除 `tests/fixtures/ddp_test_data.pt` / `ddp_test_labels.pt`

#### 2.3.5 题目与分值（【原文】notebook）
`distributed_communication_single_node`(5)、`naive_ddp`(5)、`naive_ddp_benchmarking`(3)、`minimal_ddp_flat_benchmarking`(2)、`ddp_overlap_individual_parameters`(5)、`ddp_bucketed_benchmarking`(3)；作业一 benchmark 脚本、作业二 混合精度累积(1)、作业三 benchmarking_mixed_precision(2)、作业四 **memory_profiling(4)**、作业五 pytorch_attention(2) + FlashAttention、作业六 torch_compile(2)。
**只在 CHANGELOG 出现、notebook 无论述**：`distributed_communication_multi_node`（2x1/2x2/2x3）、`optimizer_state_sharding_accounting`。

#### 2.3.6 模型规模表（【原文】表1）
small 768/3072/12/12｜medium 1024/4096/24/16｜large 1280/5120/36/20｜**xl 1600/6400/48/25**｜**2.7B 2560/10240/32/32**

#### 2.3.7 单卡做不了的部分（【原文】+【估计】）
- 【原文】`naive_ddp_benchmarking` / `minimal_ddp_flat_benchmarking` / `ddp_bucketed_benchmarking(a)` 均写死「**1个节点 x 2个GPU**」+ XL 模型
- 【原文】`distributed_communication_single_node`：「最多可使用 **6 张 GPU**；每一次基准测试运行时间**不得超过 5 分钟**」
- 【原文】`memory_profiling` 要 **2.7B**（d_model 2560/32层）
- 【估计】2.7B fp32 权重 ≈ 10.8GB，加梯度/优化器/激活**远超 12GB** → profiling 必须降级到 small/medium
- 【估计】XL(1.6B) fp32 权重 ≈ 6.4GB + Adam 状态 ≈ 12.8GB → 12GB 上跑不了训练步
- 【原文】`作业2.ipynb` 主动降规模：「# 3. 0.5B模型定义，可以自行修改参数配置，**这里是让笔记本也能跑起来**」（`XLModel(hidden_size=1024, num_layers=24)`，input `(2,512)`）—— 官方要 XL，共建版缩到 0.5B

#### 2.3.8 工时（【估计】，subagent）
环境 4–8h｜作业一~三 6–10h｜作业四 3–6h｜作业五 a/b 5–8h｜FA2 PyTorch 6–10h｜**FA2 Triton 前向+反向 15–30h**｜DDP 四题 15–25h｜ShardedOptimizer 6–12h｜报告 6–10h
**合计 66–119h；单卡 + WSL2 现实区间 80–120h。**

#### 2.3.9 核心必做 vs 可跳过（【估计】）
- **必做**：adapters 8 函数 + 实现 → `pytest tests/` 16 例全绿；FA2 PyTorch 版（含 causal、恰好保存一个 (B,Nq) 的 L）；FA2 Triton 版前向+反向；三个 DDP 变体 + 三个钩子；`ShardedOptimizer`；作业一~六交付物
- **可跳过/降级**：XL 双卡真实通信数据 → 降级 0.5B 单卡并写明原因；2.7B profiling → small/medium；`distributed_communication_multi_node` → 跳过或租云；6 进程/1GB NCCL 全网格 → 只做 1–2 进程到 100MB；leaderboard 冲榜
→ **⚠️ 与 AGENTS.md 的 P2 边界直接冲突**：分布式（DP/TP/PP/ZeRO）与 CUDA kernel 被明确列为 **P2「延后到 offer 之后」**，而作业2 正好是这两块。**建议只做 P0 部分（FA2 PyTorch 版 + profiling/显存核算方法论 + benchmark 报告写法），不要投入 Triton backward 那 15–30h。**

---

## 3. 作业3：assignment3-scaling（扩展定律）

### 3.1 【原文】README 核心信息
目标：从零构建 Transformer，研究 **非嵌入参数量 N / 数据量 D → loss** 的缩放律，并用 Chinchilla 公式 `L(N,D) = E + A/N^α + B/D^β` 预测不同规模模型的最终 loss。

【原文】预设三种规模：

| 规模 | 参数量 | d_model | num_layers | num_heads |
|---|---|---|---|---|
| 小型 | ~70M | 384 | 24 | 6 |
| 中型 | ~130M | 512 | 36 | 8 |
| 大型 | ~300M | 768 | 36 | 12 |

【原文】`model.py` 示例 config：`vocab_size=32000, context_length=2048, d_model=768, num_layers=12, num_heads=12, d_ff=3072`
【原文】训练命令：`python train.py --epochs 8 --batch_size 16 --d_model 384 --num_layers 24`（70M）
【原文】**数据是 int32 二进制**，`data.bin` 需由作业1的 `get_train_data.py` 生成；「建议每一个参数至少分配到 20 个 token 训练」
【原文】仓库里实际只做了**第一种配置（70M）**：`ckpt/scaling_records_70M.npy`、`ckpt/optimal_curve_70M.png`、`ckpt/fix_N_70M.png`

【原文】真实文件：`fit_scaling.py`(583 B)、`scaling.py`(3,270 B)、`predict loss.py`(2,742 B)、`test1.py`(2,235 B)、`model.py`(8,568 B)、`train.py`(15,669 B)、`isoflops_curves.json`(7,484 B)、`tokenizer.json`(3,571,373 B)、`配置.text`(383 B)、`Scaling Law.png`、`CS336_Assignment3_Scaling low.pdf`
**没有 tests/，没有 test_and_make_submission.sh。**

### 3.2 【估计】可行性与成本
- 70M 模型 @ context 2048 / batch 16：显存【估计】约 6–11 GB（fp32 + logits 2048×16×32000 ≈ 4.2 GB），**12GB 属于「勉强能跑、要调参」区间**；130M / 300M 配置【估计】在 12GB 上单卡 fp32 **跑不动**（除非大幅降 batch + 梯度累积）。
- 数据量：按 README 的「每参数 ≥20 token」，70M × 20 = **14 亿 token**；int32 → `data.bin` ≈ **5.6 GB**。
- 算力：`6 × 70e6 × 1.4e9 = 5.9e17` FLOPs ≈ **588 PFLOPs**，@15 TFLOPS ≈ **11 小时/次训练**。而拟合 scaling law 需要**多次不同 (N,D) 配置** → 【估计】**几十到几百 GPU·小时**。
- → 【估计】**结论：单张 4070S 只能做「70M 单配置 + 用仓库自带 `isoflops_curves.json` / `scaling_records_70M.npy` 做拟合分析」这一条路。完整 scaling law 复现做不了。**

### 3.3 【估计】核心必做 vs 可跳过
- 核心必做：读懂 `scaling.py` + `fit_scaling.py` + `predict loss.py`；**直接用仓库自带的 `scaling_records_70M.npy` / `isoflops_curves.json` 跑拟合**，产出一条 loss-FLOPs 曲线 + 预测点。
- 可跳过：130M/300M 配置；重新跑多组 IsoFLOPs 实验。

---

## 4. 作业4：assignment4-data（数据处理）

### 4.1 【原文】README 核心信息
- 「`cs336_data` 该文件夹基本为空！这是你将要实现数据过滤和处理代码的模块」（实际已含实现文件）
- **「你不应修改训练逻辑，因为排行榜提交必须完全使用该实现」** → **【原文】确认有 leaderboard 排行榜提交**
- 流程在 `cs336_data/作业一.ipynb` 与 `作业二.ipynb`

### 4.2 【原文】测试文件（6 个）
`tests/test_extract.py`、`test_deduplication.py`、`test_langid.py`、`test_pii.py`、`test_quality.py`、`test_toxicity.py`，另有 `adapters.py`、`common.py`、`fixtures/moby.html`
→ **有 pytest，有 `test_and_make_submission.sh`。**

### 4.3 【原文】实现文件
`cs336_data/`：`extracted_data.py`、`filter.py`、`cc2train.py`、`train_fasttext.py`、`run_identify_language.py`、`lid.176.ftz`（fastText 语言识别模型，**已自带**）
根目录：`get_assets.sh`、`test_and_make_submission.sh`、`pyproject.toml`
`cs336-basics/`：`cs336_basics/{data,ddp_utils,model,optimizer,train_config}.py` + `configs/config.yaml` + `configs/experiment/your_data.yaml`

### 4.4 【估计】可行性与成本
- 抽取/去重/语言识别/质量过滤/毒性过滤/PII 基本都是 **CPU 任务** → 4070S 的 12GB 显存不是瓶颈，但**磁盘与下载量是瓶颈**。
- （`get_assets.sh` 内容见 4.5。）

### 4.5 深度核实补充（subagent 报告，【原文】/【估计】已分别标注）

#### 4.5.1 GPU 需求：**GPU 完全不是瓶颈，磁盘才是**
【原文】作业一~作业九 **100% 纯 CPU**（正则 / resiliparse / langdetect / fastText 推理 / Gopher 规则 / exact + minhash 去重）。
【原文】唯一 GPU 环节是最后的 LM 训练验证，且仓库作者在 `cs336-basics/cs336_basics/train_config.py` 里亲手把模型减半并注释「**笔记本8G显存可以跑**」。
→ 【估计】**12GB 必然够**。真正瓶颈是 **375 GB WET 数据 / 约 1.2 TB 磁盘 / 数十小时下载**，与显卡无关。

#### 4.5.2 八个必须动手前知道的坑（【原文】核实）
1. **`tests/fixtures/` 里只有 `moby.html` 一个文件**（1,256 B）。测试需要的 `moby_extracted.txt`、`documents_with_line_duplicates/`、`documents_line_deduplicated/`、`documents_with_fuzzy_duplicates/`、`low_quality_cc.txt`、`high_quality_wiki_reference.txt` **全部不存在** → 【估计】**21 个 test 里约 15 个必挂**。
2. 【原文】`tests/adapters.py`（官方评分入口）有 **6 个函数是 `raise NotImplementedError`**：`run_classify_nsfw`、`run_classify_toxic_speech`、`run_classify_quality`、`run_gopher_quality_filter`、`run_exact_line_deduplication`、`run_minhash_deduplication`。实现散落在 `cs336_data/filter.py` 和两个 notebook 里，**没接线**。
3. 【原文】`adapters.py::run_extract_text_from_html_bytes` **`return 1`**（bug）。
4. 【原文】`run_identify_language.py` 硬编码 `/mnt/d/项目/.../lid.176.bin`，与仓库文件名 `lid.176.ftz` 不符；`adapters.py` 同功能却用 `langdetect`（两套实现）。
5. 【原文】`cs336-basics` **没有 `train.py`**，`pyproject.toml` 无 `[project.scripts]`。想跑训练验证必须自己补入口。
6. 【原文】两个 notebook 到「作业十的 WET 提取」就结束，**没有 tokenize / 训练 / 排行榜提交代码**。
7. 【原文】README 写「运行 `cs336_systems/作业1.ipynb`」，实际路径是 `cs336_data/`（笔误）。我自己也核到 README 写的 `Assignment2_System/相关文档` 路径不存在。
8. 【原文】`test_and_make_submission.sh` 用 `|| true` 吞掉测试失败，且 zip **排除 `*.txt` / `*.bin`**。
   → 【估计，subagent 已标注为推断】如果书面回答存成 .txt、模型存成 .bin，等于没交。

#### 4.5.3 题目与分值（【原文】来自 notebook 内的中文翻译）
作业一 look_at_cc 4分｜作业二 提取文本 3分｜作业三 语言识别 6分｜作业四 屏蔽PII 3分｜作业五 有害内容 6分｜作业六 gopher_quality_filters 3分｜**作业七 quality_classifier 15分**｜作业八 exact_deduplication 3分｜作业九 MinHash去重 8分｜作业十 filter_data 6分。
→ 【估计】合计 **57 分**（原文无总分）。【原文】抓取内容中**未出现"隐藏测试"字样**。

#### 4.5.4 适配器函数签名（【原文】）
`run_extract_text_from_html_bytes(html_bytes: bytes) -> str|None` / `run_identify_language(text) -> tuple[Any,float]` / `run_mask_emails` / `run_mask_phone_numbers` / `run_mask_ips`（均 `-> tuple[str,int]`）/ `run_classify_nsfw` / `run_classify_toxic_speech` / `run_classify_quality`（`-> tuple[Any,float]`）/ `run_gopher_quality_filter(text) -> bool` / `run_exact_line_deduplication(input_files, output_directory)` / `run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)`

#### 4.5.5 数据量（【原文】实测 wget Length）
- 【原文】handout：「**5000 个 WET 文件**……约 **375GB** 的压缩文本」；整个 Common Crawl = **100,000 个 WET**
- 【原文】CC-MAIN-2025-18 WARC 1.04 GB / WET 77 MB；CC-MAIN-2025-51 WARC 872/838/865 MB，WET ×6 ≈ 452 MB
- 【原文】`lid.176.bin` **125 MB**；`lid.176.ftz` **916 KB**（仓库已提交，938,013 B）
- 【原文】Jigsaw NSFW / hatespeech fastText：各 **946 MB**
- 【原文】`enwiki-20240420-extracted_urls.txt.gz` **1007 MB**；enwiki 全量 dump **23 GB**
- 【原文】**`get_assets.sh` 只下 2 个 HuggingFace 的 Dolma Jigsaw 模型，脚本里不写任何大小**。**但已用 `hf-mirror.com` 镜像 API 拿到字节级确切值**：
  | 脚本里的文件名 | 下载 URL（`get_assets.sh` 原文） | 精确大小 |
  |---|---|---|
  | `dolma_fasttext_nsfw_jigsaw_model.bin` | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin` | **991,753,734 字节** |
  | `dolma_fasttext_hatespeech_jigsaw_model.bin` | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin` | **992,245,824 字节** |
  ※ sha256 分别为 `dad02e9a612b85643d8f9acfcf810dcc26abb3df111b3ccb51619e5d3626ac27` 与 `3b4b994163f5ec6955a722c0c93cbadd6828a023249b1fdffeaf681659607452`
  → **合计 = 1,983,999,558 字节 = 1.848 GiB ≈ 1.98 GB**（相加是推断）
  ※ **交叉验证**：`作业一.ipynb` 从另一来源 `dolma-artifacts.org` 下载的同族模型，wget 实测 `Length:` 也是 991,753,734 与 992,245,824，**逐字节完全一致**
  → **【原文结论】`get_assets.sh` = 2 个文件 / 1.98 GB，完全不含 Common Crawl 数据**，也不含 `lid.176.ftz`（后者已直接提交在仓库，938,013 字节）
- 【估计】最小跑通 **约 3 GB 下载 / 8–10 GB 磁盘**；完整作业十 **375 GB 下载 / 1.2–1.5 TB 磁盘**；整个 CC 约 7.5 TB

#### 4.5.6 训练验证配置（【原文】）
`train_steps=100_000`、`train_batch_size=128`、`context_length=128`、`gradient_accumulation_steps=1`；`device="cuda"`, `dtype="bfloat16"`, `compile=True`。
减半后的形状：`d_model=384 / 6层 / 6头 / d_ff=1024`；注释掉的原始形状 `d_model=768 / 12层 / 12头`。
【原文】CHANGELOG：`[1.0.4] - 2025-05-19: Halve training tokens for the leaderboard run`。
→ 【估计】100k × 128 × 128 = **1.64e9 tokens**。

#### 4.5.7 工时（【估计】，subagent）
- **最小可交付版 40–70 h**
- **完整作业十（含 375 GB）100–200 h**，其中约 70% 卡在下载
- 【原文】作者实测：单进程过滤 **27.69 s/WET 文件**（约 900–1000 rec/s）；fastText 处理 780,806 篇用 **22 分 42 秒**（产出 247,635 条正样本）；5000 WET 单进程外推约 **38 h**

#### 4.5.8 核心必做 vs 可跳过（【估计】）
- **必做**：作业二/三/四(3个PII)/五(2个分类器)/六/七(质量分类器，15分最重)/八/九 的适配器实现 + 作业十(a)小规模过滤脚本 + (a)的丢弃比例书面说明 + (b)时间估算（只需外推）
- **可跳过**：作业一(a)(b)(c)读书报告、作业四5/作业五4 的 20 样本分析、**真下 5000 WET（375 GB）**、排行榜提交（需 Stanford 集群复现）
- **半必做**：作业一(d) 25 文档注释（可缩量）；补 `train.py` 跑训练验证

→ **【估计】结论：这是 6 个作业里性价比最差的一个，建议在 67 天计划中降级或跳过。**

---

## 5. 作业5：assignment5-alignment（对齐：SFT + GRPO）

### 5.1 【原文】README 关键事实（这份 README 信息量最大）
- **基座模型：`Qwen2.5-Math-1.5B`**
- 依赖：`torch`、`transformers`、**`vllm`**、**`flash-attn`**、`accelerate`、`math-verify`、`wandb`、`pytest`；Python `>=3.11,<3.13`；用 `uv` 管理
- 安装分两步：`uv sync --no-install-package flash-attn` 然后 `uv sync`
- 【原文】**「建议使用带有至少 80GB 显存的 GPU（运行完整 SFT / GRPO 实验时更推荐 2*80GB 级别）」**
- 需要实现的文件（【原文】明确点名）：
  - `cs336_alignment/sft_helper.py`：「你需要实现的 SFT & 评分相关核心函数（`tokenize_prompt_and_output` 等）」
  - `cs336_alignment/gpro_helper.py`：「你需要实现的 GRPO/策略梯度相关核心函数（`compute_group_normalized_rewards` 等）」
  - `cs336_alignment/drgrpo_grader.py`：`r1_zero_reward_fn`、`question_only_reward_fn`
  - `cs336_alignment/log.py`：`log_generations`
- 【原文】交付物（课程要求）：**`writeup.pdf`**（所有书面问题答案）+ **`code.zip`**（至少含 `sft_helper.py`、`gpro_helper.py` 及改动的实验脚本）
- 【原文】SFT 数据规模 sweep：`{128, 256, 512, 1024, full}`
- 【原文】GRPO CLI 子命令：`lr_sweep`、`baselines`、`length_norm`、`std_norm`、`off_policy` / `off_policy_sweep`、`clip_ablation`、`prompt_ablation`、`leaderboard`
- 【原文・救命条款】**「如果你没有那么多资源从头复现实验，我们也提供了实验记录可直接访问」** → 3 条 wandb 报告链接（SFT / SFT_EI / GRPO）
- 【原文】「如果作业讲义或本地实现有出入，以哪个为准？**优先以官方讲义 + tests 为准。**」→ 说明**官方版有 tests，但这个仓库副本里没有**
- 补充可选作业：`cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`

### 5.2 【原文】真实文件树要点
`cs336_alignment/`：`sft_math_reasoning.py`、`sft_math_reasoning_ei.py`、`grpo.py`、`grpo_experiments.py`、`gpro_helper.py`、`sft_helper.py`、`evaluate_math.py`、`drgrpo_grader.py`、`log.py`
`prompts/`：`alpaca_sft.prompt`、`question_only.prompt`、`r1_zero.prompt`、`zero_shot_system_prompt.prompt`
`images/`：约 40 张实验曲线 png（`sft_eval_acc.png`、`grpo_clip_train_entropy.png`、`computational_cost_comparison.png` …）
**没有 `tests/`，没有 `test_and_make_submission.sh`，也没有 `data/` 目录！**

→ 【原文+推断】README 第 1 节点名的 `data/gsm8k/test.jsonl`、`data/math12k/sft_gpt-oss-120b.jsonl`、`data/sft/sft_gpt-oss-120b_filtered.jsonl` **在仓库树里都不存在**，必须从官方 CS336 作业5 脚手架另行获取。

### 5.3 【估计】可行性
- 【原文】README 自己建议 80GB / 2×80GB。**12GB 单卡与要求差一个数量级。**
- 但基座只有 **1.5B**：bf16 权重约 3 GB。理论上可在 12GB 上做短序列 SFT。
- 真正的瓶颈是 **GRPO 的 rollout 采样**：需要 vLLM（或 HF generate）大批量生成，且论文级实验要跑 lr sweep / 8 组消融 → 【估计】单卡跑「一个」GRPO 配置都紧张，跑完整个消融矩阵**不可能**。
- → 【估计】**结论：只能部分做。**可行路径 = 实现 `sft_helper.py` + `gpro_helper.py` 核心函数 + 用仓库自带 `images/` 的曲线和 README 提供的 wandb 报告写 `writeup.pdf`；**不做（或只做极小规模）实际 GRPO 训练**。

### 5.4 深度核实补充（subagent 报告，【原文】/【估计】已分别标注）

#### 5.4.1 卡点的真正原因（【估计】，最重要的工程结论）
**不是"模型太大"，而是"双卡常驻"的架构问题。**
Qwen2.5-Math-1.5B bf16 权重仅约 3.1GB，`tensor_parallel_size=1`，不需要张量并行。
【原文】README.ipynb §4.6：「**GRPO 训练使用了两张 GPU：GPU 0 策略模型训练；GPU 1 vLLM 推理和评估**」（代码印证：`device_policy="cuda:1"` / `device_eval="cuda:0"`，每步 `load_policy_into_vllm_instance` 同步权重）。
【估计】单卡显存账：训练侧 ≈9.3GB（权重 3.1 + 梯度 3.1 + 8bit AdamW 3.1）+ vLLM 6GB ≈ **15.3GB > 12GB → 必然 OOM**（即使把 `vllm_gpu_memory_utilization` 压到 0.25 仍超）。
→ **必须改成"分时复用"**（先 vLLM 生成完 → 释放/sleep → 再训练），需改 `grpo.py` / `grpo_experiments.py` 的紧耦合循环。
→ 【估计】吞吐差距：4070S 带宽 504GB/s ≈ H100 SXM 的 1/6.6、bf16 算力 ≈1/7，且只有 1 卡还要分时复用（生成与训练无法重叠）→ 有效吞吐约为参考配置的 **1/15~1/25**。

#### 5.4.2 分值（【原文】，合计约 56 分）
§3 `math_baseline`(4) → `evaluate_vllm`；§4 `tokenize_prompt_and_output`(2)、`compute_entropy`(1)、`get_response_log_probs`(2)、`masked_normalize`(1)、`sft_microbatch_train_step`(3)、`log_generations`(1)、`sft_experiment`(2)；§5 `expert_iteration_experiment`(2)；§7 `compute_group_normalized_rewards`(2)、`compute_naive_policy_gradient_loss`(1)、`compute_grpo_clip_loss`(2)、`compute_policy_gradient_loss`(1)、`masked_mean`(1)、`grpo_microbatch_train_step`(3)、`grpo_train_loop`(5)；§8 `grpo_learning_rate`(2)、`grpo_baselines`(2)、`think_about_length_normalization`(1)、`grpo_length_normalization`(2)、`grpo_group_standard_deviation`(2)、`grpo_off_policy`、`grpo_off_policy_sweep`(4)、`grpo_off_policy_clip_ablation`(2)、`grpo_prompt_ablation`(2)；§9 `leaderboard`(**16**)。
→ **`leaderboard` 独占 16 分（约 29%）**，官方硬约束「**2 个 H100 GPU 上 4 小时**」+ 整个 MATH 5K 验证集 → 【估计】单卡约 160h，**物理上不可能**。
【原文】准确率门槛：SFT 全量数据 ≥15%、EI ≥15%、GRPO 学习率 ≥25%（均为 MATH）。
【原文】原文点名的 pytest 选择器：`test_tokenize_prompt_and_output` / `test_compute_entropy` / `test_get_response_log_probs` / `test_masked_normalize` / `test_sft_microbatch_train_step` / `test_compute_group_normalized_rewards` / `test_compute_naive_policy_gradient_loss` / `test_compute_grpo_clip_loss` / `test_masked_mean` / `test_grpo_microbatch_train_step`（命令 `uv run pytest -k <name>`）。

#### 5.4.3 官方算力（【原文】，硬数字）
每题 H100 小时：sft 2 / EI 6 / lr 6 / baselines 2 / length_norm 2 / std_norm 2 / off_policy_sweep **12** / clip_ablation 2 / prompt_ablation 2 / **leaderboard 16** → **合计 52 H100 小时**。
GRPO 超参（【原文】）：`n_grpo_steps=200, rollout_batch_size=256, group_size=8, train_batch_size=256, epochs_per_rollout_batch=1, grad_accum_steps=64, cliprange=0.2, lr=1e-5`, gen/eval `max_tokens=1024`, `bnb.optim.AdamW8bit` → 一次 200 步 = **51,200 条 rollout**。
SFT（【原文】）：`BATCH_SIZE=4, GRAD_ACCUM=8, LR=5e-5, EPOCHS=1, DATASET_SIZES=[128,256,512,1024,2048]`。
EI（【原文】）：`N_EI_STEPS=5, Db=[512,1024,2048], G=[1,4,8,16], epochs=[1,4,8,16]` → main() 共 8 组。

#### 5.4.4 交付物与测试（【原文】）
- 提交物：**`writeup.pdf` + `code.zip` → Gradescope**；本仓库补充要求 `code.zip` 至少含 `sft_helper.py`、`gpro_helper.py`、你改的实验脚本
- **官方有 pytest**（`tests/test_sft.py`、`tests/test_grpo.py`，通过 `tests/adapters.py` 钩子连接）**但这份拷贝被删掉了 `tests/`**
- **有隐藏测试**（官方 tests/ + Gradescope）——【原文】但本仓库副本中不存在
- 【原文】反讽点：`pyproject.toml` 仍声明 `pytest>=8.3.5`，`README.ipynb` 写着「运行单元测试 `uv run pytest`」，`README.md` FAQ 写着「优先以官方讲义 + tests 为准」——但**跑 `uv run pytest` 会收集到 0 个测试**
- 【原文】本仓库 9 个 `.py` **全部是完整实现**，抓取内容里**没有任何 `NotImplementedError` / `TODO`**

#### 5.4.5 ⚠️ 关键坑（【原文】）
1. **`bitsandbytes` 被 import 但没写进 `pyproject.toml`**（`grpo.py`/`grpo_experiments.py` 用 `bnb.optim.AdamW8bit`）→ `uv sync` 后跑 GRPO 会 **ModuleNotFoundError**
2. **无 `peft`、无 `trl`** → 官方路线是 **1.5B 全参微调，不是 LoRA/QLoRA**
3. **`data/` 目录不在仓库里**（API 实测）→ README 点名的 `data/gsm8k/test.jsonl`、`data/math12k/sft_gpt-oss-120b.jsonl`、`data/sft/..._filtered.jsonl` 都要自己从 HF 下
4. `pyproject.toml` 硬钉 `vllm==0.7.2`，5 个脚本全部 `from vllm import LLM, SamplingParams` → **vLLM 是硬依赖**（官方措辞只是"推荐"，代码层面是必需）；而 **vLLM 不支持原生 Windows → 又必须 WSL2**
5. 【原文】`README.ipynb` 中 `r1_zero_reward_fn` 的格式判定是 `if "</think> <answer>" in response` —— **`</think>` 与 `<answer>` 之间必须恰好一个空格**，作者实测发现这是 format_reward 只有 50.42% 的主因（解析器过严而非模型差）
6. 【原文】`sft_helper.py` 与 `gpro_helper.py` 各有一个**同名不同签名**的 `masked_normalize`（`normalize_constant` vs `constant_normalizer`），极易抄错
7. 【原文】**版本警告**：官方 main 分支已是 **Spring 2026**，测试钩子名与 2025 handout 不同（2026 用 `run_compute_rollout_rewards`/`run_aggregate_loss_across_microbatch`/`run_grpo_train_step`）。本仓库镜像的是 **Spring 2025 v1.0.2**。要取回官方 tests 必须 checkout 2025 对应 tag/commit。
8. 【原文】模型路径硬编码作者私有路径 `/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B`

#### 5.4.6 工时与磁盘（【估计】）
- 环境（WSL2 + CUDA + torch2.5.1 + vllm0.7.2 + flash-attn 编译 + 取回官方 tests）**6~20h**；若自己实现 helper 再 +8~15h
- 全量照搬：**约 140~270 h**；**推荐裁剪版约 30~50 h**
- 磁盘：Qwen2.5-Math-1.5B ≈3.1GB；Python 环境 ≈10~18GB；checkpoint ≈40~60GB → **最小可跑 ≥40GB；较完整 ≥120GB**

#### 5.4.7 核心必做 vs 可跳过（【估计】）
- **必做（约 30~40h）**：12 个带 pytest 的 helper 函数 + `grpo_train_loop` + `math_baseline` + `sft_experiment`(缩规模) + `grpo_learning_rate`(缩到 3档×30步) + **全部书面问题（零算力，性价比最高）** + writeup/code.zip
- **应做（+30~60h）**：`log_generations`、EI **只做 1 组**、`baselines`/`length_normalization`/`group_standard_deviation`/`prompt_ablation`（各 2 次 50 步可缩 30 步）、`grpo_off_policy` 代码改造 + `clip_ablation` 短跑
- **建议放弃**：`grpo_off_policy_sweep`(4分，原文硬性固定 rollout=256)、**`leaderboard`(16分)**、全部可选 safety/RLHF 补充作业、Llama-8B/70B 相关、全量 SFT 12 次 sweep、全量 EI 8 组
- 【原文】⚠️ 写报告的坑：参考实现的实测数字**全部是 GSM8K（1319 条）上的**，不是官方要求的 MATH 5K，所以 76.20% 这类数字**不能**与官方 15%/25% 门槛对比

---

## 6. 作业6：assignment6-evaluation（模型评估）

### 6.1 【原文】README 核心信息
- 目标是**介绍**评测框架（定位是「介绍与演示」，不是重实现）：lm-evaluation-harness（EleutherAI）、evalscope（ModelScope）、Evalchemy（ML Foundations）、lighteval（HuggingFace）
- 重点演示 **lm-evaluation-harness** 和 **evalscope**
- 【原文】lm-evaluation-harness 覆盖：零样本 `arc_easy, piqa, lambada, triviaqa`；少样本 `humaneval, mbpp, gsm8k, minerva_math`
- 【原文】安装：`conda create -n eval_env python=3.10`；`cd lm-evaluation-harness; pip install -e .; pip install -e .[math]`；`pip install evalscope`；`pip install 'evalscope[app]' -U`
- 【原文】运行：`python lm_eval_demo.py`、`python evalscope_demo.py`

### 6.2 【原文】真实文件树 + 两处文档不一致
实际只有：`README.ipynb`、`README.md`、`lm_eval_demo.py`、`evalscope_demo.py`、`data/index_testset.jsonl`、`images/evalscope_panel.png`、`outputs/20260119_232050/...`、`outputs/20260120_000654/...`
- README 目录结构写有 **`demo.ipynb`**，**实际文件名是 `README.ipynb`**
- README 写有 **`lm-evaluation-harness/` 框架源码目录**，**实际仓库树里没有** → 需自己 git clone
**没有 tests/，没有 test_and_make_submission.sh。**

【原文】`outputs/` 实际评测对象是 **`gpt2`**，评测集出现在文件名里：
`aime25_AIME2025-I`、`aime25_AIME2025-II`、`arc_ARC-Challenge`、`arc_ARC-Easy`、`ceval_logic`、`gsm8k_main`，另有 `data_collection_default`、`index_testset`

### 6.3 【估计】可行性
- 被评模型是 **gpt2**（124M）→ 12GB 上跑 lm-eval / evalscope **毫无压力**。
- 【估计】主要成本是**下载**：评测框架依赖 + 各 benchmark 数据集 + 模型权重；evalscope 的 Web UI 也吃一点资源。
- 【估计】**结论：12GB 单卡完全能做完**，是 6 个作业里**最轻**的一个。也可部分改用 API 调云端模型。
- 【估计】唯一注意：`humaneval` / `mbpp` 需要代码执行沙箱；`aime25` / `minerva_math` 是数学评测，gpt2 分数会接近随机（预期如此）。

### 6.4 深度核实补充（subagent 报告 + 我自己的交叉核对，【原文】）

**官方 demo 本身就是 CPU 跑的**：`lm_eval_demo.py` 里 `HFLM(..., device="cpu", ...)`；evalscope 侧 `# device='cuda'` 被注释。
→ 【估计】这是作业6 最轻的直接证据：官方不要求 GPU。

**没有作业题干**：【原文】README.md 与 README.ipynb **完全没有"作业要求 / 提交格式 / 评分标准 / 思考题"段落**。实际产物只有 2 个脚本 + `data/index_testset.jsonl`（38,762 字节，**只有 10 行**）+ `reports/*.json` + `reviews/*.jsonl` + 1 张 dashboard 截图。
**没有排行榜对比表**：5 个 config 的 `model:` 与 3 个 report 的 `model_name` **全部是 gpt2**——只评了一个模型。

**所有分数全是 0.0**（gpt2 太弱；唯一非 0 的是 `arc_easy acc_norm=1.0`、`piqa acc=1.0`，因 `limit=1` 样本量无意义）。
→ 【估计】**照抄 gpt2 全 0 分写进简历没有意义**。

**规模数字（【原文】）**：`count=10` 条样本；lm-eval `limit=1` 每任务；`batch_size=32`；`max_gen_toks=512`；`max_length=1024`。
**真实耗时（【原文】ipynb 日志）**：evalscope 10 条样本 = **26 分 46 秒**；lm-eval 未裁剪时 ETA **2 小时 50 分**（`576/6477 [16:36<2:50:13, 1.73s/it]`）；hellaswag limit10 cuda「不到一分钟」。
**数据集源规模（【原文】）**：gsm8k test 1319/train 7473；AIME2025 I=15、II=15；ARC-Easy test 2376；ARC-Challenge test 1172；ceval logic val 22/dev 5。

**下载量与磁盘（【估计】）**：评测数据集合计 5–10 MB；gpt2 权重约 0.5 GB；**真正的磁盘大头是 pip 依赖**（torch CUDA wheel 单包 2–3 GB + evalscope/transformers/datasets + `evalscope[app]`）→ **建议预留 20–30 GB**。

**能否纯 API 完成？【原文】能，且是官方预置路线**：`evalscope_demo.py` 里官方写好并**注释掉**的
`api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'` + `api_key=os.getenv('DASHSCOPE_API_KEY')` + `eval_type='openai_api'`，注释原文写着「可以是云上的API」，还有 `eval_batch_size=5 # 根据你的 API 并发限额调整`。
但【估计】lm-eval 那一半**没有 API 模板**（用本地 `HFLM`），且 `arc_easy/piqa/lambada` 是 loglikelihood 型任务、多数云 API 不给 logprobs 会受限。
→ 【估计】**建议：lm-eval 侧保留 gpt2 本地小跑证明框架跑通，把"真实多模型对比"放在 evalscope 侧用 API 做。**

**第10 / 第12 章正文都不含作业要求**：【原文】第12章完整 md 全文无"作业6"、无 lm-evaluation-harness、无 evalscope、无任何作业要求段落（纯综述）；第10章 grep `作业|评测框架|lm-eval|evalscope|Evaluation` → No matches found。
→ 作业6 的要求**只存在于** `coursework/assignment6-evaluation/README.md` 与 `README.ipynb`，而这两份也只到"跟着跑两个 demo"的粒度。

### 6.5 【原文】仓库级 CI 核实：没有任何 CI 判分（我自己抓的）

我抓取了 `.github/workflows/`：**整个仓库只有 1 个 workflow：`deploy.yml`**（1,790 字节）。
其内容【原文】是纯粹的 VitePress 文档站点构建 + GitHub Pages 部署：
`name: Deploy ViteNotes site to Pages` → `pnpm install --frozen-lockfile` → `pnpm run build` → `actions/upload-pages-artifact@v3` → `actions/deploy-pages@v4`。

→ **【原文结论】该仓库没有任何测试型 CI、没有自动判分、没有 leaderboard 后端。** 这关闭了「是否存在隐藏测试/CI 评分」这个缺口：**不存在**。
→ 【估计】因此本仓库 6 个作业的"完成"完全靠**自证**：跑通 + 产出文件/曲线。

（`.github` 目录 REST 列举在本次抓取时返回 **HTTP 403 API rate limit exceeded**；但 workflow 文件本身已通过 raw 逐个抓到，其中 `deploy.yml` 是文件树中唯一列出的 workflow，故结论成立。）

---

## 7. 六个作业横向对比（【估计】为主，已并入 4 个 subagent 的深度核实）

| 作业 | 有 pytest? | 有提交脚本? | 交付物 | 12GB 单卡 | 小时【估计】 | 下载/磁盘【估计】 |
|---|---|---|---|---|---|---|
| 1 basics | ❌ | ❌ | 代码 + ckpt + PPL 曲线 | ✅ **完全能**（峰值显存仅 2–5GB） | **3–7 h** | train.txt 1.2GB(实测) + data.bin 0.63GB ≈ **1.9GB 磁盘**；用自带 valid.txt 则 **0** |
| 2 systems | ✅ 4 个文件 / **16 例** | ✅ | 代码 + 表格 + 截图 + zip | ⚠️ **只能部分**（16 例可全绿，但需 **WSL2**；XL 双卡/2.7B profiling 做不了） | **66–119 h**（单卡现实 **80–120 h**） | ~几 GB（依赖极大） |
| 3 scaling | ❌ | ❌ | 拟合曲线 + 预测 + 图 | ⚠️ **只能部分**（70M 单配置勉强；130M/300M 跑不动） | 10–25 h（全量约 11h/次训练 × 多组） | data.bin ≈ **5.6 GB**（int32、14 亿 token） |
| 4 data | ✅ 6 个文件 | ✅ | 过滤后数据集 + leaderboard | ⚠️ CPU 为主，**GPU 不是瓶颈，磁盘是** | **最小版 40–70 h**；完整 **100–200 h** | `get_assets.sh` = **1.98 GB**（2 文件）；完整作业十 **375 GB / 1.2–1.5 TB** |
| 5 alignment | ❌（官方有，被删） | ❌ | `writeup.pdf` + `code.zip` | ⚠️ **只能部分**（约 1/4–1/3 题量；需 WSL2 + vLLM） | 裁剪版 **30–50 h**；全量 **140–270 h** | **最小 ≥40 GB；较完整 ≥120 GB** |
| 6 evaluation | ❌ | ❌ | 评测 report json + dashboard 截图 | ✅ **完全能**（官方 demo 就是 CPU 跑的） | **4–8 h** | 数据 5–10 MB + gpt2 0.5GB；**预留 20–30 GB**（大头是 pip 依赖） |

**共同环境前提**：作业2 与作业5 都需要 **WSL2**（`triton==3.5.1` 标记 `sys_platform=='linux'`；vLLM 不支持原生 Windows）。

---

## 8. 抓取失败项（明确不猜）

1. 【抓取失败】`TinyStoriesV2-GPT4-train.txt` 确切字节数 —— `huggingface.co`、`api/datasets/.../tree/main`、`datasets-server.huggingface.co` 全部 fetch failed（后者报 "resolves to a non-public IP address"）。
2. 【抓取失败】`Assignment1_Basics.pdf` / `cs336_spring2025_assignment2_systems.pdf` 等 PDF 正文未解析（未尝试深解析；PDF 文本提取在本环境未经验证）。
3. 【抓取失败】`data/index_testset.jsonl`（作业6）与 `isoflops_curves.json`（作业3）的**内容**未逐字读取（仅从文件树拿到字节数）。
4. 【未取得】`get_assets.sh`（作业4）内容 —— 决定作业4 下载量的关键文件，见 subagent 报告。
5. 【原文事实】所有 `PPL.md` 只有图片链接、**没有数字**，因此**无法从仓库得到任何 PPL 数值**。

---

## 9. 对 67 天求职冲刺的直接建议（【估计】，供 parent 决策）

1. **作业1 应该做，而且优先级高**：它是唯一一个「12GB 单卡能在 20h 内拿完整交付物」的作业，且直接命中 AGENTS.md 里 P0 的「Transformer 能手写」+「优化器」+「训练流程」。**把 BPE 训练和 1.9GB 下载砍掉**是关键。
2. **作业6 应该做**：最轻，且「评测」正好命中 AGENTS.md 里 P1 的「Agent 工程 → 评测」短板，出报告成本低。
3. **作业2 只做单卡部分**：FlashAttention / Triton / profiling 单卡可做（命中「LLM 推理工程」P0）；DDP / sharded optimizer 需 ≥2 卡，**标注为「读代码 + 讲原理，不实跑」**。
4. **作业3 只做分析不做训练**：直接用仓库自带 `.npy` / `.json` 跑拟合，产出一条曲线即可交差。
5. **作业4 建议只做 CPU 部分且跳过 Common Crawl 全量**：磁盘是硬约束，需先确认 `get_assets.sh` 到底要下多少。
6. **作业5 建议「实现核心函数 + 读 wandb 报告写 writeup」**，不实跑 GRPO；其目标岗位相关性对「AI Agent / 大模型应用」而言低于作业1/2/6。
7. **一句话**：6 个作业按「性价比」排序应为 **1 > 6 > 2(单卡部分) > 3(分析部分) > 5(实现+报告) > 4(受磁盘限制)**。
