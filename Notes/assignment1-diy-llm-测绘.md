# datawhalechina/diy-llm —— 作业1（assignment1-basics）测绘报告

> 测绘时间：2026 年（当前会话）
> 测绘方式：`web_fetch` 抓 raw.githubusercontent.com / api.github.com / 在线阅读站 / Stanford 官网
> **本文严格区分【原文事实】（抓到什么写什么）与【我的估计】（公式+假设+标注）。凡未抓到的一律写「抓取失败」，不做任何猜测填充。**

---

# 第一部分：目录结构核实（原文事实）

## 1.1 GitHub Contents API 实测结果

`GET https://api.github.com/repos/datawhalechina/diy-llm/contents/coursework/assignment1-basics` 返回 **12 个条目**，与你给的 ground truth 完全一致：

| # | name | type | size (bytes) |
|---|------|------|--------------|
| 1 | `Assignment1_Ablations` | dir | 0 |
| 2 | `Assignment1_Basics.pdf` | file | 419506 |
| 3 | `CS336_Assignment1_BPE.ipynb` | file | 20986 |
| 4 | `CS336_Assignment1_Transformer.ipynb` | file | 93686 |
| 5 | `README.md` | file | 5461 |
| 6 | `TinyStoriesV2-GPT4-valid.txt` | file | 22502601 |
| 7 | `bpe_tokenizer` | dir | 0 |
| 8 | `ckpt` | dir | 0 |
| 9 | `get_train_data.py` | file | 2601 |
| 10 | `model.py` | file | 14356 |
| 11 | `train.py` | file | 14501 |
| 12 | `Assignment1_Ablations`（子目录内容另查） | — | — |

## 1.2 子目录实测（原文事实）

**`ckpt/` 只有 1 个文件：**
```
ckpt/PPL.md    size=270
```
→ **仓库里没有任何 `.pt` 权重文件。** 这意味着直接 `python model.py` 会走到 `else` 分支打印
`[WARNING] 权重文件不存在，使用随机初始化模型`（原文），生成的是随机权重输出。

**`bpe_tokenizer/` 只有 1 个文件：**
```
bpe_tokenizer/tokenizer.json    size=3822009
```

**`Assignment1_Ablations/` 共 13 个条目：**
```
SiLU/                       (dir)
SiLU_model.py               7448
SiLU_train.py               14077
data.in                     13766960
no_RMSNorm/                 (dir)
not_RMSNorm_model.py        7560
not_RMSNorm_train.py        14317
original_model/             (dir)
post_Norm/                  (dir)
post_Norm_model.py          7582
post_Norm_train.py          14091
tokenizer.json              3571373
```
- `no_RMSNorm/` 内只有 `PPL.md`（140 B）
- `post_Norm/` 内只有 `PPL.md`（273 B）
- `original_model/` 内只有 `PPL.md`

## 1.3 「没有 tests/ 目录」核实结论（原文事实 + 我的说明）

**核实通过。** `assignment1-basics` 下 12 个条目中 **没有 `tests/`，没有 `test_and_make_submission.sh`，也没有 `pyproject.toml` / `requirements.txt` / `.gitignore` / `uv.lock` / 源码包目录**。

**对照核实**：`GET .../contents/coursework/assignment2-systems` 返回 11 个条目，其中**确实包含**：
```
tests/                             (dir)
test_and_make_submission.sh        760 B
pyproject.toml                     523 B
requirements.txt                   51928 B
uv.lock                            221997 B
.gitignore                         3188 B
src/                               (dir)
cs336_systems/                     (dir)
main.py / README.md / 相关文档/
```

**这个差异的含义（我的分析，基于上述抓取到的文件清单）：**

1. **assignment1 是「从零重写版」，不是 Stanford 骨架的移植版。** 它的 API 形状（`Linear/Embedding/RMSNorm/SwiGLU/RotaryPositionalEmbedding/MultiHeadAttention/TransformerBlock/TransformerLM`）与 Stanford 的 `tests/adapters.py` 接口是两套东西。
2. **Stanford 的 pytest 套件无法直接跑在 diy-llm 的代码上**——因为 diy-llm 侧完全没有 `tests/adapters.py` 这个「适配层」。
3. **assignment1 的「验收」不是测试驱动，而是「跑出产物 + 看图」。** 交付物是训练日志、`ckpt/epoch_N.pt`、`train_ppl.png` / `val_ppl.png`。这一点在下面的「测试与评分」一节展开。
4. assignment2 保留了 Stanford 的测试基础设施，说明 duy-llm 作者对 a1 做了完全的「教程化重写」（这是一个**观察**，作者动机未在任何抓取内容中说明）。

---

# 第二部分：要写哪些模块 / 函数（原文，函数名级别）

## 2.1 关键结论：**全仓库 a1 相关文件中，没有一处 `raise NotImplementedError`**

我做了全文正则检索（`NotImplementedError|测试框架|pytest|test_|TODO|提交|评分`），在两个 notebook 的完整/补读内容中：
- `CS336_Assignment1_BPE.ipynb`：**0 处 NotImplementedError**
- `CS336_Assignment1_Transformer.ipynb`：**0 处 NotImplementedError**
- `get_train_data.py` / `model.py` / `train.py` / `SiLU_model.py` / `SiLU_train.py` / `not_RMSNorm_model.py`：**0 处 NotImplementedError**

**所以「填空式骨架」在 diy-llm 的 a1 里根本不存在——所有代码都是已实现的成品。** 这一点与 Stanford 原版（README 原文：*"Initially, all tests should fail with `NotImplementedError`s. To complete your implementation, complete the functions in ./tests/adapters.py"*）**根本不同**。

## 2.2 `CS336_Assignment1_Transformer.ipynb` 的组件清单（原文，按 notebook 的顺序与标题）

**「一、基本板块的构建」**

| 序 | 原文标题注释 | 原文类名/函数名与签名 |
|---|---|---|
| 1 | `#  1. 自定义线性层的实现（没有偏置项）` | `class Linear(nn.Module)` / `__init__(self, in_features, out_features, device=None, dtype=None)` / `reset_parameters(self)` / `forward(self, x)` |
| 2 | `# 2.自定义嵌入模块（Embedding）` | `class Embedding(nn.Module)` / `__init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` / `reset_parameters(self)` / `forward(self, token_ids)` |
| 3 | `# 3. 实现RMSNorm归一化层` | `class RMSNorm(nn.Module)` / `__init__(self, d_model, eps=1e-5, device=None, dtype=None)` / `forward(self, x)` |
| 4 | `# 4. 实现SwiGLU激活函数 <- SiLU+门控机制GLU` | `class SwiGLU(nn.Module)` / `__init__(self, d_model)` / `forward(self, x)`（内部 `w_gate` / `w_up` / `w_down`） |
| 5 | `# 5. 实现RoPE位置编码 <- 旋转位置编码（相对位置编码对长上下文友好）` | `class RotaryPositionalEmbedding(nn.Module)` / `__init__(self, theta: float, d_k: int, device=None)` / `_build_cache(self, seq_len, device, dtype)` / `forward(self, x)` |
| 6 | `# 6. 实现softmax函数` | `def Softmax(x: torch.Tensor, dim:int) -> torch.Tensor` |
| 7 | `# 7. 实现缩放点积注意力机制` | `def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None)` → `return output, attn_weights` |
| 8 | `# 8. 因果多头注意力机制实现` | `class MultiHeadAttention(nn.Module)` / `__init__(self, d_model: int, num_heads: int)` / `forward(self, x, mask=None)` |
| 9 | `# 9. 实现Transformer` | `class TransformerBlock(nn.Module)` / `__init__(self, d_model, num_heads, d_ff)` / `forward(self, x, mask=None)` |

**「二、 支持Transformer训练的板块构建」**

| 序 | 原文标题注释 | 原文类名/函数名与签名 |
|---|---|---|
| 1 | `# 1. 实现交叉熵损失函数` | `def cross_entropy(logits, targets)` |
| — | （紧接下一 cell） | `def calculate_perplexity(logits, targets)` |
| 2 | `# 2. 自定义实现AdamW优化器` | `class AdamW(Optimizer)` |
| 3 | `# 3. 实现LLamA类似的余弦退火学习率调度器` | `def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c)` |
| 4 | `# 4. 梯度裁剪` | `def run_gradient_clipping(params, max_norm, eps=1e-6)` |
| 5 | `# 5. 数据加载（给定前面的token预测下一个token）` | `def get_batch(x, batch_size, context_length, device)` |
| 6 | `# 6. 保存和加载checkpoint` | `def save_checkpoint(model, optimizer, iteration, out)` / `def load_checkpoint(src, model, optimizer)` |

**notebook 内的自测适配函数（原文，命名以 `run_` 开头）：**
```
run_embedding(num_embeddings, embedding_dim, weights, token_ids)
run_rmsnorm(d_model: int, x: torch.Tensor, eps: float)
run_swiglu(d_model: int, x: torch.Tensor)
run_rope(theta: float, d_k: int, x: torch.Tensor) -> torch.Tensor
run_softmax(x: torch.Tensor, dim: int)
run_multihead_self_attention(d_model, num_heads, x)
run_transformerBlock(d_model, num_heads, d_ff)
run_get_batch(x_list, batch_size, context_length, device_str)
run_save_checkpoint(model, optimizer, iteration, out)
run_load_checkpoint(src, model, optimizer) -> int
test_implementation()
test_gradient_clipping_logic()
```

**原文注释里的「测试框架」字样（说明这些 `run_*` 是给某个外部框架调用的）：**
- `# 测试框架提供的输入(batch, seq_len),这里模拟输入两句话每句话有3个token`
- `# 测试框架可能会以列表形式传入数据，需要确保它是numpy数组`

> **我的说明**：这两条注释暗示作者原本对齐过某个自动测试框架的调用约定，但**该框架本身没有被提交到仓库**（无 tests/、无 conftest.py）。

## 2.3 `CS336_Assignment1_BPE.ipynb` 的函数清单（原文）

```
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]
def iter_text_chunks_with_monitor(file_path: str, chunk_size: int = 1_000_000, log_every: int = 5)
def get_memory_mb()
def log_status(prefix, bytes_processed, start_time)
def train_bpe_tokenizer(train_file: str, val_file: str | None = None, vocab_size: int = 50257,
                        num_chunks: int = 8, output_dir: str = "./bpe_tokenizer")
def analyze_tokenizer(tokenizer, texts)
def load_stories(file_path, num_samples=None)
```

原文中的分词器构造（**用的是 HuggingFace `tokenizers` 库，不是从零手写 BPE**）：
```python
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
```
`special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]`

**原文中的两处实际不一致（我读到的事实，不是我推测）：**
1. `find_chunk_boundaries()` 被定义但**在 notebook 后续 cell 中从未被调用**。
2. `train_bpe_tokenizer(..., num_chunks=8, ...)` 的 `num_chunks` 形参**在函数体内未被使用**；`__main__` 里却传了 `num_chunks=16`。
3. `run_softmax` 的实现是 `return softmax(x, dim)`（小写 `softmax`），但本 notebook 定义的是大写 `Softmax`；该 cell 却带有 `execution_count: 468` 和输出——即**源码与执行输出不一致**（原文如此）。
4. `run_transformerBlock` 向 `TransformerBlock(d_model, num_heads, d_ff)` 传入 `d_ff`，但 `TransformerBlock.__init__` 接收 `d_ff` 后**未使用**（SwiGLU 自己算 `d_ff`）。

**与根 README 的矛盾（原文对照）**：仓库根 `README.md` 的章节表把第2章配套作业描述为「**手写 tokenizer 训练代码**」，但实际交付的 notebook 是**调用 `tokenizers` 库训练 BPE**。两者不一致。

## 2.4 `model.py` 的清单（原文，可直接落地的成品）

```
class Linear(nn.Module)                  __init__(in_features, out_features, device=None, dtype=None) / reset_parameters() / forward(x)
class Embedding(nn.Module)               __init__(num_embeddings, embedding_dim, device=None, dtype=None) / reset_parameters() / forward(token_ids)
class RMSNorm(nn.Module)                 __init__(d_model, eps=1e-5, device=None, dtype=None) / forward(x)
class SwiGLU(nn.Module)                  __init__(d_model) / forward(x)     # d_ff = (int(8/3*d_model)+63)//64*64
class RotaryPositionalEmbedding(nn.Module)  __init__(theta, d_k) / _build_cache(seq_len, device, dtype) / forward(x)
class MultiHeadAttention(nn.Module)      __init__(d_model, num_heads) / forward(x, mask)
class TransformerBlock(nn.Module)        __init__(d_model, num_heads) / forward(x, mask=None)
class TransformerLM(nn.Module)           __init__(vocab_size, d_model, num_heads, num_layers, max_seq_len)
                                         forward(idx) / generate(idx, max_new_tokens, temperature=1.0, top_k=None)
class SimpleTokenizer                     __init__(vocab_path) / encode(text) / decode(ids)
def generate_with_sampling(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None)
def decode_generated_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None, device='cpu')
class CustomAdamW(torch.optim.Optimizer)  __init__(params, lr=1e-3, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0) / step(closure=None)
```

## 2.5 `train.py` 的清单（原文）

```
class CustomAdamW(torch.optim.Optimizer)   # 与 model.py 中重复定义（复制粘贴）
def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c)
def run_gradient_clipping(params, max_norm, eps=1e-6)
class CausalMemmapDataset(Dataset)         __init__(data_path, context_length, start_block=0, end_block=None) / __len__ / __getitem__
def save_ppl_curve(train_ppls, val_ppls, save_dir)
def save_checkpoint(path, model, optimizer, iteration, epoch, config: dict)
def get_memory_usage(device)
def main()
```

`train.py` 的 argparse 参数（原文，默认值）：
```
--epochs        5
--batch_size    16
--context_length 128
--d_model       256
--num_heads     8
--num_layers    6
--vocab_size    50257
--lr            3e-4
--min_lr        3e-5
--checkpoint_dir ./ckpt
--data_path     data.bin
```

## 2.6 `get_train_data.py` 的清单（原文）

```
def load_tokenizer(tokenizer_json_path: str) -> Tokenizer
def text_to_token_ids(text: str, tokenizer: Tokenizer, add_eos: bool = True) -> List[int]
def build_random_data_bin(input_txt: str, tokenizer_json: str, output_bin: str,
                          target_samples: int = 10000, dtype=np.int32)
```
`__main__` 里的实际调用（原文）：
```python
build_random_data_bin(
    input_txt="TinyStoriesV2-GPT4-train.txt",
    tokenizer_json="bpe_tokenizer/tokenizer.json",
    output_bin="data.bin",
    target_samples=800000
)
```

## 2.7 消融实验的清单（原文，已抓取部分）

**README 原文的消融说明：**
```
│   ├── not_RMSNorm_model.py    # 完全移除归一化层
│   ├── not_RMSNorm_train.py    # 实验1训练函数
│   ├── post_Norm_model.py      # 后归一化
│   ├── post_Norm_train.py      # 实验2训练函数
│   ├── SiLU_model.py           # SwiGLU换成SiLU
│   ├── SiLU_train.py           # 实验3训练函数
```

**我实际抓取并核验的两个：**

- `SiLU_model.py`：新增 `class SiLU(nn.Module)`（`f_c1 = Linear(d_model, d_ff)`、`f_c2 = Linear(d_ff, d_model)`、`return self.f_c2(F.silu(self.f_c1(x)))`），并把 `TransformerBlock.ffn` 从 `SwiGLU(d_model)` 换成 `SiLU(d_model)`。**SiLU 不是门控**，参数比 SwiGLU 少一个投影。
- `not_RMSNorm_model.py`：`TransformerBlock` 中把 `self.norm1/norm2` **注释掉**，forward 改为 `x = x + self.attention(x, mask)` / `x = x + self.ffn(x)`。
  ⚠️ **原文与 README 描述不符**：README 说「**完全**移除归一化层」，但实际上 `TransformerLM.norm_final = RMSNorm(d_model)` **仍然保留**。所以只是移除了 block 内部的 norm，不是「完全移除」。

- `post_Norm_model.py` / `post_Norm_train.py` / `not_RMSNorm_train.py`：**未抓取**（本轮未请求），仅知 README 的「后归一化」描述。

- `SiLU_train.py`（已抓取）默认参数与 `train.py` 的差异：
```
--epochs        10            (train.py 是 5)
--data_path     data.in       (train.py 是 data.bin)
--checkpoint_dir ./Assignment1_Ablations/SiLU
tokenizer_file  "tokenizer.json"   (注意：不是 bpe_tokenizer/tokenizer.json)
if epoch % 10 == 0: 才保存 checkpoint     (train.py 是 epoch % 1)
PPL 图另存为 train_ppl3.png / val_ppl3.png  (train.py 是 train_ppl.png / val_ppl.png)
```

---

# 第三部分：交付物（原文事实）

README 里明确给出的三步流程（原文）：

**1. 准备训练数据**
```bash
python get_train_data.py
```
> 「这会生成 `data.bin` 文件（从训练文本中随机抽取 80 万个样本）。」

**2. 训练模型**
```bash
python train.py
# 或
python train.py --epochs 10 --batch_size 16 --d_model 256 --num_heads 8 --num_layers 6 --lr 3e-4
```

**3. 文本生成**
```bash
python model.py
```
> 「默认会加载 `ckpt/epoch_5.pt` 权重文件，并根据 `prompt` 生成文本。」

**环境要求（README 原文）：**
```bash
pip install torch numpy transformers tokenizers tqdm psutil matplotlib
```

## 3.1 交付物清单（据源码中实际写盘的产物）

| 产物 | 来源（原文） | 备注 |
|---|---|---|
| `data.bin` | `get_train_data.py` → `arr.tofile(output_bin)` | int32 二进制 token 流 |
| `ckpt/epoch_{epoch}.pt` | `train.py` → `save_checkpoint(path=f"ckpt/epoch_{epoch}.pt", ...)` | 含 `model` / `optimizer` / `iteration` / `epoch` / `config` |
| `ckpt/train_ppl.png` | `train.py` → `save_ppl_curve()` | log-scale PPL 曲线 |
| `ckpt/val_ppl.png` | 同上 | |
| 终端日志 | `train.py` 每 100 step 打印 `LR / Train Loss / Train PPL / Step Time / 内存占用` | |
| 生成文本 | `model.py` → `print("Generated:", generated_text)` | |
| 3 组消融的 PPL 图 | `*_train.py` | `train_ppl3.png` / `val_ppl3.png` 等 |

## 3.2 明确「没有」的交付物（核实结论）

- ❌ **没有书面报告 / 实验分析文档要求**（README、notebook、代码中均无「报告」「PDF 提交」「写一份分析」之类的要求）
- ❌ **没有 PDF 格式的作业要求文档可解析**（仓库有 `Assignment1_Basics.pdf`，但无法抓取，见末节）
- ❌ **没有 CI 配置、没有提交脚本、没有 Gradescope 相关内容**
- ❌ **没有 `requirements.txt` / `pyproject.toml`**（依赖只写在 README 的一句 `pip install` 里）
- ❌ **没有权重文件**（`ckpt/` 只有 `PPL.md`，见 1.2）

---

# 第四部分：测试与评分方式（原文事实）

## 4.1 diy-llm 侧的结论

| 项目 | 结论 | 证据 |
|---|---|---|
| pytest 测试套件？ | **没有** | API 实测 `assignment1-basics` 无 `tests/`、无 `conftest.py`；全文检索无 `pytest` 字样 |
| 本地测试脚本？ | **没有** | 无 `test_and_make_submission.sh`（assignment2 有，760 B） |
| 隐藏测试？ | **抓不到任何证据** | 无 adapters.py、无 CI 配置 |
| 评分标准文档？ | **没有** | README 只讲「怎么跑」，不讲「怎么评分」 |
| 实际验收方式 | **看产物**：训练日志里的 Loss/PPL、`train_ppl.png`/`val_ppl.png` 曲线、生成文本是否像英文小故事 | 源码产物路径 |

## 4.2 notebook 内的「测试」是什么

Transformer notebook 里那些 `run_*` / `test_*` 函数是**自写的临时验证代码**（用 `torch.randn` 造随机输入，或者用 `torch.manual_seed(42)` + 手算对照），原文注释写的是「测试框架提供的输入」「测试框架可能会以列表形式传入数据」——**但那个「测试框架」不在仓库里**。

BPE notebook 里的「验证」是：
```python
encoded = tokenizer.encode(" Hello, world! <|endoftext|>")
print(encoded.tokens)   # ['ĠHello', ',', 'Ġworld', '!', 'Ġ', '<|endoftext|>']
print(encoded.ids)      # [8431, 16, 1501, 5, 149, 0]
```
以及 `analyze_tokenizer` 统计 `avg_tokens` / `max_tokens`。这些是 notebook 内联的，不构成可自动化的测试套件。

## 4.3 `PPL.md` 的真实内容（重要，原文）

三个目标 PPL.md 抓取结果如下 —— **全部只有图片链接，没有任何数字**：

- `ckpt/PPL.md`（270 B）：
  ```html
  <img width="640" height="480" alt="train_ppl" src="https://github.com/user-attachments/assets/ccd1ffaa-..."/>
  <img width="640" height="480" alt="val_ppl"   src="https://github.com/user-attachments/assets/96807e5c-..."/>
  ```
- `Assignment1_Ablations/original_model/PPL.md`：同上结构，2 张图（`1131f13b...` / `ba048456...`）
- `Assignment1_Ablations/SiLU/PPL.md`：2 张图，alt 为 `train_ppl3` / `val_ppl3`

**含义：官方没有以文本形式记录任何 PPL 数值。** 唯一的结果记录是 640×480 的 log-scale 折线图 PNG。
→ **我没有用视觉模型去读图中的数值**，因为在对数坐标曲线上读数是不可靠的，会把猜测混进「事实」层。需要数字的话，只能自己重跑或人工看图。

---

# 第五部分：数据需求（原文事实 / 估计分开）

## 5.1 `get_train_data.py` 到底下载什么 URL —— 回答：**它不下载任何东西**

**原文事实：`get_train_data.py` 全文没有任何 URL，没有任何 `requests`/`urllib`/`wget`/`huggingface_hub` 调用，也没有任何下载逻辑。** 它只做本地文件读取：

```python
with open(input_txt, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)          # 第一遍：数总行数
...
with open(input_txt, "r", encoding="utf-8") as f:   # 第二遍：只处理抽中的行
```
默认 `input_txt="TinyStoriesV2-GPT4-train.txt"`（相对路径，**该文件不在仓库里**）。

**URL 只出现在 README 里（原文引用）：**
> 「- TinyStoriesV2-GPT4-train.txt - 训练文本数据（这里因为文件大小限制没有需要自己下载点击👉[数据集](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)）」

注意：README 给的是**HuggingFace 数据集目录页**，不是直链文件 URL。
（对比：Stanford 原版 README 给的是直链 —— 见第七节外部来源。）

## 5.2 仓库内已有的数据资产（原文事实，API 实测 size）

| 文件 | size (bytes) | 换算 | 说明 |
|---|---|---|---|
| `TinyStoriesV2-GPT4-valid.txt` | 22,502,601 | 21.46 MiB | **在仓库里，可直接用** |
| `bpe_tokenizer/tokenizer.json` | 3,822,009 | 3.64 MiB | 训练好的 BPE 分词器 |
| `Assignment1_Ablations/tokenizer.json` | 3,571,373 | 3.41 MiB | 消融用的另一份分词器 |
| `Assignment1_Ablations/data.in` | 13,766,960 | 13.13 MiB | int32 二进制 token 流（与 `data.bin` 同格式） |
| `TinyStoriesV2-GPT4-train.txt` | **不确定** | — | **不在仓库中**，且 HF 抓取全部失败（见末节） |

**由 `data.in` 的字节数可直接算出的确定值（算术，非估计）：**
> `13,766,960 ÷ 4 = 3,441,740` 个 int32 token
> 对应训练块数 = `(3,441,740 − 128 − 1) // 128 = 26,888` 块（按 `CausalMemmapDataset` 的原文公式）
> 其中训练集 80% = 21,510 块，验证集 20% = 5,378 块

## 5.3 数据集规模 —— 我的估计（**明确标注为估计，HF 抓取失败**）

**已抓到的硬事实**：`TinyStoriesV2-GPT4-valid.txt` = 22,502,601 B。
**抓取失败的**：TinyStoriesV2-GPT4-train.txt 的实际字节数与 token 数（HF 的 page / API / raw 三种路径全部 `fetch failed`，见末节）。

**我的估计（假设 + 公式，必读）：**
- **假设 1**：`target_samples=800000` 抽取的是 800,000 **行/故事**（README 原文写「随机抽取 80 万个样本」，变量名 `sampled_count` 也是按条计数），**不是 token 数**。
- **假设 2**：平均每个故事 ≈ 197 token。**依据是 BPE notebook 自己跑出来的输出**（原文 stdout）：
  ```
  训练集统计: {'avg_tokens': 197.0, 'max_tokens': 308}
  验证集统计: {'avg_tokens': 208.2, 'max_tokens': 434}
  ```
  ⚠️ 但这只是 **20 个随机样本**的统计，样本量极小，且与 `get_train_data.py` 按「行」抽样、notebook 按 `<|endoftext|>` 切故事是两种粒度。

→ **估计 token 总量 ≈ 800,000 × 197 ≈ 1.58 亿 token**
→ **估计 `data.bin` 磁盘占用 ≈ 1.58e8 × 4 B ≈ 630 MB**

**必须强调**：脚本 `print` 出来的是 `成功随机抽取并保存{len(arr):,}tokens到{output_bin}` —— 真实 token 数是数据相关的，**任何抓取到的文档里都没有给出这个数字**。上面全是我的估计。

---

# 第六部分：算力需求

## 6.1 模型规模（原文事实）

来自 `train.py` 默认参数 + `model.py` 的实现：

| 项 | 值 | 来源 |
|---|---|---|
| `vocab_size` | 50257 | `train.py` 默认；BPE notebook `vocab_size=50257` |
| `d_model` | 256 | 默认 |
| `num_heads` | 8（→ `d_k = 32`） | 默认 |
| `num_layers` | 6 | 默认 |
| `context_length` | 128 | 默认 |
| `batch_size` | 16 | 默认 |
| `epochs` | 5 | 默认 |
| `d_ff` | **704** | `model.py` 原文 `d_ff=int(8/3*256)=682` → `(682+63)//64*64=704` |
| 优化器 | `CustomAdamW(lr=3e-4, weight_decay=0.1)` | 原文 |
| 精度 | **fp32**（代码中无 AMP / autocast / bf16） | 原文 |
| 梯度累积 | **无** | 原文 |
| 梯度检查点 | **无** | 原文 |

## 6.2 参数量（算术推导，非估计；`train.py` 自己会打印这个数）

```
token_embedding : 50257 × 256                    = 12,865,792
每层 attention  : 4 × (256×256)                  =    262,144
每层 SwiGLU     : 3 × (256×704)                  =    540,672
每层 RMSNorm×2  : 2 × 256                        =        512
每层合计                                          =    803,328
6 层合计                                          =  4,819,968
norm_final      : 256                            =        256
lm_head         : 256 × 50257                    = 12,865,792
─────────────────────────────────────────────────────────────
总计                                              = 30,551,808  ≈ 30.55 M
```
> 交叉验证：`train.py` 原文有 `print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")` —— 跑一下就能对上这个 30.55M。

## 6.3 训练 token 数（分两种情况）

**(a) 默认配置（按 5.3 的估计数据量）**
```
训练块数 = (1.58e8 − 129) // 128 ≈ 1,231,250 块
训练集(80%) ≈ 985,000 块  →  985,000 × 128 ≈ 1.261e8 token/epoch
5 epochs ≈ 6.30e8 token           ← 估计
```
**(b) 消融配置（用 `data.in`，这里 token 数是确定的）**
```
26,888 块，训练集 21,510 块 → 21,510 × 128 = 2.753e6 token/epoch
SiLU_train.py epochs=10 → 2.753e7 token   ← 确定值
```

## 6.4 「原始训练用的什么卡」——**抓取失败**

**没有任何抓取到的内容说明官方是在什么 GPU 上训练的。** README、两个 notebook、三个 py 文件里都**没有提到 GPU 型号**。notebook 输出里唯一的硬件线索是 `psutil` 报的**内存**占用和 `c:\users\l1337\.conda\envs\llm`（Windows + conda），以及 `SiLU_train.py` 中 `get_memory_usage` 里对 `"cuda"` / `"mps"` 的分支判断。**没有显存数字，没有卡型号。**

## 6.5 官方建议的显存 —— **抓取失败 / 仓库中不存在**

README 里只有一条相关的问答（原文）：
> **Q2: 训练时内存不足怎么办？**
> A2: 减小 `batch_size` 或 `context_length` 参数。

**没有任何具体显存数字建议。** Stanford 原版作业 PDF 里的显存/算力建议也**没抓到**（PDF 无法 fetch，见末节）。

---

# 第七部分：RTX 4070 Super 12GB 能否完成

## 7.1 结论

> ### ✅ **能完整完成（不是「只能部分」）。** 默认配置（d_model=256 / 6 层 / ctx 128 / batch 16 / 5 epochs）在 12 GB 上**有巨大余量**，甚至不需要梯度检查点、不需要混合精度、不需要梯度累积。

## 7.2 显存估算（**这是我的估计，公式与假设全部列出**）

**公式：**
```
显存 ≈ M_参数 + M_梯度 + M_优化器 + M_激活值 + M_碎片余量

M_参数   = 4N                      (fp32)
M_梯度   = 4N
M_优化器 = 8N                      (AdamW 的 exp_avg + exp_avg_sq，各 4N)
M_激活 ≈ B×T×V×4 × 2 + O(L×B×T×d_model)
         ↑ logits 本体  ↑ CrossEntropyLoss 内部 log_softmax 需存给 backward
```
其中 `N = 30.55e6`，`B=16`，`T=128`，`V=50257`，`L=6`，`d_model=256`。

**代入：**
```
M_参数   = 4 × 30.55e6  = 122.2 MB
M_梯度   = 4 × 30.55e6  = 122.2 MB
M_优化器 = 8 × 30.55e6  = 244.4 MB
          ─────────────────────────
          小计           = 488.8 MB

logits   = 16 × 128 × 50257 × 4 B = 411.7 MB
CE 中间量 ≈ 411.7 MB
隐状态   = 6 层 × 若干 × (16×128×256×4 B = 2.1 MB) ≈ 数十 MB
因果 mask = 16 × 1 × 128 × 128 × 1 B = 0.26 MB
          ─────────────────────────
          激活小计 ≈ 850 MB ~ 900 MB

总计 ≈ 1.35 GB    (理论最小值)
实测预期 ≈ 1.5 ~ 2.5 GB   (含 PyTorch caching allocator 碎片与临时张量)
```
**假设与保守性说明：**
1. 假设 fp32（**代码确实没用 AMP**，这是原文事实，不是假设）。
2. 假设 `F.scaled_dot_product_attention(is_causal=True)` 生效 → 注意力是 O(T) 显存而非 O(T²)；代码里 `self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')`，PyTorch ≥ 2.0 必然为 True。即使走 fallback 分支，`16×8×128×128×4 B = 8.4 MB` 的 scores 也微不足道。
3. **最大头是 `lm_head` + CrossEntropyLoss 的 logits（合计 ~0.82 GB）**，因为 vocab=50257 而模型只有 30M 参数——这是这个配置的显存特征。
4. 未计入 cuDNN/cuBLAS workspace（通常几百 MB）。

**→ 12 GB 剩余约 9.5 GB，余量极为充足。** 可以放心把 `batch_size` 提到 64~128，或把 `context_length` 提到 512+（**这是我的估计，非抓取内容**）。

## 7.3 与「已知硬件约束」的对照（我的推断，标注清楚）

> **以下是基于抓取到的模型规模做的推断，不在抓取内容中：**
> - AGENTS.md 里说 4070S 12G「不能做全参微调 7B、大规模消融」——**对 a1 完全不构成限制**：a1 的模型只有 30.55M 参数，比 7B 小 230 倍，**属于「玩具级」规模**。
> - AGENTS.md 说 T400 4GB「不能做 tiny GPT 训练」——按上面的估算（理论 ~1.4 GB，实测 ~2 GB），**a1 的默认配置在 4 GB 卡上有机会跑起来，但不能保证**（4 GB 可用显存通常只剩 ~3.5 GB，logits 那 0.82 GB 加上碎片容易触顶）。要稳，还是上 4070S，或把 `batch_size` 降到 4~8。**这是我的估计。**

---

# 第八部分：需要多少小时（全部为估计）

## 8.1 训练算力估算（公式 + 假设）

**每 token 前向 FLOPs（按 MAC=2 FLOPs）：**
```
attention 4 个投影       : 4 × 256² × 2                       =   524,288
attention scores + 加权  : 2 × (2 × T × d_model)  (T=128)     =   131,072
SwiGLU (3 个投影)        : 3 × 256 × 704 × 2                  = 1,081,344
每层小计                                                       = 1,736,704
× 6 层                                                         = 10,420,224
lm_head                  : 2 × 256 × 50257                     = 25,731,584
─────────────────────────────────────────────────────────────────────────
前向 ≈ 36.15 M FLOPs / token
训练 ≈ 3 × 前向 ≈ 108.5 M FLOPs / token
```
**RTX 4070 Super 规格假设**：FP32 峰值 ≈ 35.5 TFLOPS（无 FP32 tensor core 加速）；小模型 + fp32 + Python 训练循环，**有效算力取峰值的 15%~30% → 5~10 TFLOPS**（这是估计，不同驱动/框架会有差异）。

**(a) 默认配置（5 epochs，按 6.30e8 token 估计）**
```
总 FLOPs = 6.30e8 × 108.5e6 = 6.84e16
@ 5 TFLOPS  → 13,680 s ≈ 3.8 h
@ 10 TFLOPS →  6,840 s ≈ 1.9 h
+ 验证集 (20% 数据，1/5 额外前向) ≈ +25%
+ Python 循环 / DataLoader / 每 100 步打印 / 存盘 ≈ +30%
────────────────────────────────────────────────
估计纯 GPU 训练时间 ≈ 3 ~ 6 小时
```
**(b) 消融配置（`SiLU_train.py`，data.in，10 epochs，2.753e7 token —— token 数确定）**
```
总 FLOPs = 2.753e7 × 108.5e6 = 2.99e15
@ 5 TFLOPS  →  ~10 min
@ 10 TFLOPS →  ~5 min
+ 验证 + 开销 ≈ 15 ~ 30 min / 每个消融
3 个消融 ≈ 0.75 ~ 1.5 h
```

## 8.2 数据准备时间（估计）

| 步骤 | 我的估计 | 依据 |
|---|---|---|
| 下载 `TinyStoriesV2-GPT4-train.txt` | 视网速，文件大小**未抓到** | HF 抓取失败 |
| `get_train_data.py` 第一遍数行数 | 数分钟 | 全文逐行 Python 迭代 |
| 第二遍 + 对 80 万条逐个调 `tokenizer.encode()` | **5 ~ 25 min** | 800k 次 Python→Rust 调用；若总计 ~1.58e8 token |
| **`data.bin` 生成小计** | **估计 10 ~ 30 min** | 纯 CPU，单进程 |
| BPE 分词器训练（若自己跑 notebook） | **估计 10 ~ 30 min** | notebook 自己输出的 `speed≈20-230 MB/s`，处理约 1.2 GB 文本 |

## 8.3 端到端总时间（估计）

| 场景 | 估计耗时 |
|---|---|
| **A. 最小跑通路径**（用仓库现成 `data.in` + `tokenizer.json`，`train.py --epochs 1`） | **20 ~ 60 min**（估计） |
| **B. 完整默认路径**（get_train_data.py 80 万样本 + train.py 5 epochs + model.py 生成） | **6 ~ 12 h**（估计），其中 GPU 训练 3~6 h |
| **C. 完整 + 3 组消融** | **8 ~ 15 h**（估计） |
| **D. 从零自己重写所有组件（学习为目的）** | **抓取内容中无任何工时建议**（未抓到） |

## 8.4 磁盘占用（估计 + 确定值混合）

| 项 | 大小 | 性质 |
|---|---|---|
| `TinyStoriesV2-GPT4-valid.txt` | 22.5 MB | **确定**（仓库内） |
| `TinyStoriesV2-GPT4-train.txt` | **未知** | 抓取失败 |
| `bpe_tokenizer/tokenizer.json` | 3.82 MB | **确定** |
| `data.bin` | **估计 ~630 MB**（按 5.3 的估计 token 数） | 估计 |
| `ckpt/epoch_N.pt` × 5 | **估计 ~366 MB/个 → ~1.83 GB** | 估计 |
| `train_ppl.png` / `val_ppl.png` | 数百 KB | — |
| **小计（不含 train.txt）** | **估计 ~2.5 GB** | 估计 |

> `epoch_N.pt` 的估算依据（原文）：`save_checkpoint` 存 `model.state_dict()`（4N = 122 MB）+ `optimizer.state_dict()`（AdamW 两个动量 = 8N = 244 MB）≈ 366 MB。

---

# 第九部分：「最小可完成版本」

## 9.1 关键发现：仓库自带了「跳过所有下载」的素材

因为 `data.in`、`tokenizer.json`、`TinyStoriesV2-GPT4-valid.txt` **都在仓库里**，存在一条**完全离线**的跑通路径。这是本报告最有操作价值的结论。

## 9.2 三条路径

### 路径 A：最快跑通（估计 20~60 min，甚至可以纯 CPU）

| 步 | 命令 / 动作 | 说明 |
|---|---|---|
| 1 | 克隆仓库，`pip install torch numpy transformers tokenizers tqdm psutil matplotlib` | README 原文的依赖 |
| 2 | **跳过** `get_train_data.py` 与 BPE 训练 | 直接用仓库自带 `Assignment1_Ablations/data.in`（13.77 MB，确定含 3,441,740 token）与 `bpe_tokenizer/tokenizer.json` |
| 3 | `python train.py --data_path Assignment1_Ablations/data.in --epochs 1` | `train.py` 的 `--data_path` 可覆盖；`tokenizer_file` 是硬编码 `bpe_tokenizer/tokenizer.json`，该文件存在 |
| 4 | `python model.py` | ⚠️ 它加载 `ckpt/epoch_5.pt`，跑 1 epoch 只会得到 `epoch_1.pt` → **需要改 `model.py` 里第 N 行硬编码的路径**，或者 `--epochs 5` |
| **产物** | `ckpt/epoch_1.pt`、`ckpt/train_ppl.png`、`ckpt/val_ppl.png`、终端 Loss/PPL 数字 | |

⚠️ **路径 A 的一个风险（我的提示，不是抓取内容）**：`data.in` 是用 `Assignment1_Ablations/tokenizer.json`（3,571,373 B）生成的，而 `train.py` 硬编码加载 `bpe_tokenizer/tokenizer.json`（3,822,009 B）。两份分词器词表若不同，token id 会语义错位，**训练照样能跑但结果无意义**。要严谨就得同时换 `tokenizer_file`，或干脆走路径 B。

### 路径 B：真实最小「端到端」（估计 1~3 h，无需下载 train 文件）

| 步 | 动作 | 依据 |
|---|---|---|
| 1 | 改 `get_train_data.py` 的 `__main__`：`input_txt="TinyStoriesV2-GPT4-valid.txt"`（**仓库自带 22.5 MB**），`target_samples` 改小（如 20000） | 脚本支持任意 txt |
| 2 | `python get_train_data.py` → 得到小号 `data.bin` | |
| 3 | `python train.py --epochs 5`（或更少） | |
| 4 | `python model.py`（`ckpt/epoch_5.pt` 会由第 3 步产生） | `train.py` 每 epoch 都存 |
| **产物** | `data.bin`、`ckpt/epoch_5.pt`、两张 PPL 图、一段生成文本 | |

### 路径 C：完全复刻官方（估计 6~15 h）

1. 从 HF 下 `TinyStoriesV2-GPT4-train.txt`（README 给的页面：`https://huggingface.co/datasets/roneneldan/TinyStories/tree/main`）
2. 跑 `CS336_Assignment1_BPE.ipynb` 训分词器（或直接用仓库里的）
3. `get_train_data.py`（`target_samples=800000`）→ `data.bin`
4. `train.py`（5 epochs）
5. `model.py` 生成
6. 可选：3 组消融（`SiLU_train.py` / `not_RMSNorm_train.py` / `post_Norm_train.py`，各用 `data.in`）

## 9.3 核心必做 vs 可跳过（逐项）

| # | 组件 | 必做？ | 理由（我的判断，基于抓取内容） |
|---|---|---|---|
| 1 | `Linear` | ✅ 核心 | a1 的立身之本 |
| 2 | `Embedding` | ✅ 核心 | |
| 3 | `RMSNorm` | ✅ 核心 | 且正是消融对象之一 |
| 4 | `SwiGLU` | ✅ 核心 | 且正是消融对象之一（换成 SiLU） |
| 5 | `RotaryPositionalEmbedding` | ✅ 核心 | 现代 LLM 必备，面试高频 |
| 6 | `Softmax`（手写，减 max 稳定） | ✅ 核心 | 数值稳定性必考 |
| 7 | `scaled_dot_product_attention`（手写） | ✅ 核心 | **但注意**：`model.py` 里走的是 `F.scaled_dot_product_attention`，手写版只在 notebook 里；面试验证时手写版才是重点 |
| 8 | `MultiHeadAttention` | ✅ 核心 | |
| 9 | `TransformerBlock`（Pre-Norm + 残差） | ✅ 核心 | |
| 10 | `TransformerLM` | ✅ 核心 | 只有 `model.py` 有，notebook **没有**这个类 |
| 11 | `cross_entropy`（手写） | 🟡 中 | notebook 有手写版；`model.py`/`train.py` 直接用 `nn.CrossEntropyLoss` |
| 12 | `calculate_perplexity` | 🟡 中 | 一行的事 |
| 13 | `CustomAdamW`（手写优化器） | ✅ 核心 | 面试高频（bias correction、weight decay） |
| 14 | `get_lr_cosine_schedule`（warmup + cosine） | ✅ 核心 | |
| 15 | `run_gradient_clipping` | ✅ 核心 | |
| 16 | `CausalMemmapDataset` / `get_batch` | 🟡 中 | 工程脚手架，理解 memmap + next-token shift 即可 |
| 17 | `save_checkpoint` / `load_checkpoint` | 🟡 中 | |
| 18 | **从零手写 BPE**（不用 `tokenizers` 库） | ❌ **可跳过** | **diy-llm 自己都没做**（用 `tokenizers` 库），仓库还直接提供了 `tokenizer.json` 成品 |
| 19 | 3 组消融实验 | ❌ 可跳过（先做主线） | 属于加分项；但这是**产出「有数字的消融对比」的最佳素材**，建议主线通了之后补 |
| 20 | 80 万样本的完整 `data.bin` | ❌ 可跳过 | 小数据即可验证 pipeline 正确性 |
| 21 | `generate_with_sampling`（top-k/top-p） | 🟡 中 | 推理侧采样逻辑，顺手 |

**一句话最小可完成版本**：
> **`model.py` 里的 9 个架构组件 + `TransformerLM` + `CustomAdamW` + `get_lr_cosine_schedule` + `run_gradient_clipping` + 一个能跑的 `train.py` 循环 —— 用仓库自带的 `data.in` + `tokenizer.json`，跑 1 个 epoch，拿到 PPL 下降曲线和一段能读的生成文本。这就是跑通。**

---

# 第十部分：外部对照 —— Stanford 原版（**外部来源，非 diy-llm 内容**）

> ⚠️ 以下内容来自 Stanford 官方仓库与课程主页，**不是 diy-llm 仓库的内容**，用于对照。

## 10.1 Stanford 官方 README 原文要点

来源：`https://raw.githubusercontent.com/stanford-cs336/assignment1-basics/main/README.md`

- 标题：**CS336 Spring 2025 Assignment 1: Basics**
- 完整作业说明在 PDF：`cs336_assignment1_basics.pdf`（**我无法抓取，octet-stream**）
- 环境用 `uv` 管理
- **原文明确写了 pytest：**
  ```sh
  uv run pytest
  ```
  > "Initially, all tests should fail with `NotImplementedError`s.
  > To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py)."
- **下载数据的原文命令（含直链 URL）：**
  ```sh
  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
  wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
  gunzip owt_train.txt.gz
  wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
  gunzip owt_valid.txt.gz
  ```
  → **对比：Stanford 用 TinyStories + OpenWebText 两个数据集；diy-llm 只用 TinyStories，且脚本里没有任何下载代码。**

- **Stanford `tests/` 目录实测文件清单（API 抓取）：**
  ```
  tests/__init__.py
  tests/_snapshots/          (dir)
  tests/adapters.py          25581 B
  tests/common.py             2353 B
  tests/conftest.py           9164 B
  tests/fixtures/            (dir)
  tests/test_data.py          2823 B
  tests/test_model.py         6352 B
  tests/test_nn_utils.py      3098 B
  tests/test_optimizer.py     2780 B
  tests/test_serialization.py 3878 B
  tests/test_tokenizer.py    16447 B
  tests/test_train_bpe.py     3246 B
  ```
  → `test_train_bpe.py` 的存在证明：**Stanford 原版要求从零手写 BPE 的可测试实现**；diy-llm 用 `tokenizers` 库代替，且删掉了整套 tests。

## 10.2 Stanford 课程主页（外部来源）

来源：`http://cs336.stanford.edu/`（Spring 2026 版）

- **Assignment 1: Basics** 原文描述：
  > "Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model.
  > Train a minimal language model."
- 提交方式：**全部通过 Gradescope 提交**（原文："All coursework are submitted via Gradescope by the deadline. Do not submit your coursework via email."）
- 迟交政策：6 个 late days，每个作业最多用 3 个
- AI 政策：允许问 LLM 概念问题，**禁止直接用来解题**，建议关闭 AI 自动补全
- **自学的 GPU 算力建议（原文，2026-03-28 的公开报价，单张 B200）：**
  | 平台 | 价格 |
  |---|---|
  | Modal（赞助方） | $6.25/hour，每月送 $30 |
  | Lambda Labs | $6.69/hour |
  | RunPod | $4.99/hour |
  | Nebius | $5.50/hour（抢占式 $3.05/hour） |
  | Together | $7.49/hour，最少 8 卡 |
  原文还建议：**"debugging correctness of your implementation on CPU first and then using GPU(s)"**
- **注意**：主页**没有给出 assignment 1 的具体显存/卡数要求**（正文里只说 "with the count recommended in the assignments"，具体数字在**没抓到的 PDF** 里）。

---

# 第十一部分：抓取失败项（完整清单，一个不漏）

| # | URL / 目标 | 状态 | 失败原因 |
|---|---|---|---|
| 1 | `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment1-basics/Assignment1_Basics.pdf` | ❌ **抓取失败** | `Error: unsupported content type "application/octet-stream"` —— web_fetch 不支持解析 PDF 二进制流 |
| 2 | `https://raw.githubusercontent.com/stanford-cs336/assignment1-basics/main/cs336_assignment1_basics.pdf` | ❌ **抓取失败** | 同上，`unsupported content type "application/octet-stream"` |
| 3 | `https://stanford-cs336.github.io/spring2025-assignment1-basics/` | ❌ **抓取失败** | `Error: cross-origin redirect to http://cs336.stanford.edu is not followed automatically` |
| 4 | `https://huggingface.co/api/datasets/roneneldan/TinyStories/tree/main` | ❌ **抓取失败** | `Error: web fetch failed: TypeError: fetch failed`（网络层面失败） |
| 5 | `https://huggingface.co/api/datasets/roneneldan/TinyStories` | ❌ **抓取失败** | `Error: web fetch failed: TypeError: fetch failed` |
| 6 | `https://huggingface.co/datasets/roneneldan/TinyStories` | ❌ **抓取失败** | `Error: web fetch failed: TypeError: fetch failed` |
| 7 | `https://huggingface.co/datasets/maveriq/tinystoriesv2_gpt4/raw/main/README.md` | ❌ **抓取失败** | `Error: web fetch failed: TypeError: fetch failed` |
| 8 | `https://datawhalechina.github.io/diy-llm/chapter2/chapter2_分词器.html` | ⚠️ **部分成功** | HTTP 200，但内容被截断（`Content truncated. Fetch a more specific URL or section for the full text.`）。**已抓到的部分不含任何作业要求、测试方式、算力要求** |
| 9 | `https://datawhalechina.github.io/diy-llm/chapter4/chapter4_第四章语言模型架构和训练的技术细节.html` | ⚠️ **部分成功** | HTTP 200，但内容被截断。**已抓到的部分（学习目标 + 4.1 节）不含任何作业要求、测试方式、算力要求** |
| 10 | `https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb` | ⚠️ **部分成功** | HTTP 200，但 `(Omitted 44135 bytes. Full formatted result stored at: ...7e659f6767ce-web_fetch.txt)`。**已通过读取该 spill 文件补读第 6/7/8/9 节与第二节的大部分；但被省略的 base64 图片输出未读** |
| 11 | Stanford `tests/adapters.py` 的**文件内容** | ❌ **未抓取** | 只抓取了 `tests/` 目录清单（文件名+大小）；adapters.py 的 25581 B 内容未请求，因此**无法列出 Stanford 期望的具体函数签名** |
| 12 | `TinyStoriesV2-GPT4-train.txt` 的实际字节数 / token 数 | ❌ **未获取** | 该文件不在仓库中；HF 的 3 条抓取路径全部失败（见 #4~#7）。**报告中的 1.58 亿 token / 630 MB 全部是我的估计** |
| 13 | Stanford 官方作业 PDF 里的**显存/GPU 数量建议** | ❌ **未获取** | 因 #2 失败，无法读到 assignment 内文的算力要求 |
| 14 | diy-llm 官方**用什么卡训练的** | ❌ **仓库中不存在** | 不是抓取失败，而是**所有抓取到的文件中都没写** GPU 型号（README/notebook/py 文件全文均无） |
| 15 | diy-llm 官方**建议的显存数字** | ❌ **仓库中不存在** | README 只有「内存不足就减小 batch_size 或 context_length」，**无任何数字** |
| 16 | `Assignment1_Ablations/post_Norm_model.py` / `post_Norm_train.py` / `not_RMSNorm_train.py` 的内容 | ❌ **未抓取** | 本轮未请求（只抓了 `SiLU_model.py` / `SiLU_train.py` / `not_RMSNorm_model.py`）。`post_Norm` 的「后归一化」定义**仅来自 README 的一句话描述**，未经源码核实 |
| 17 | `PPL.md` 里的 **PPL 数值** | ❌ **文本层不存在** | 三个 `PPL.md` 全文只有 `<img src="...user-attachments...">` 标签，**零数字**。数值只存在于 640×480 log-scale 折线图 PNG 中。**我没有用视觉模型读图**，因为在对数坐标曲线上读数不可靠，会把猜测污染进事实层 |
| 18 | `ckpt/` 下的训练权重 | ❌ **仓库中不存在** | API 实测 `ckpt/` 只有 `PPL.md`（270 B），**没有任何 `.pt`**。所以直接 `python model.py` 只会用随机初始化权重生成文本 |

---

# 第十二部分：给下游的三条最关键结论

1. **diy-llm 的 a1 不是「填空作业」，是「完整参考实现 + 中文讲解」。** 仓库里**没有任何 `NotImplementedError`、没有任何 `tests/`、没有任何评分标准**。它跟 Stanford 原版（`uv run pytest` + `tests/adapters.py` + 全套隐藏测试）是两种东西。拿它当「刷题式作业」会失望；拿它当「读完就能照着写出来的答案 + 消融素材」才是正确用法。

2. **12 GB 的 4070S 对这个作业是降维打击。** 模型只有 **30.55M 参数**，fp32 训练估算峰值 **~2 GB**，默认 5 epochs 估算 **3~6 小时** GPU 时间。**AGENTS.md 里「4070S 不能做大规模消融」的约束在这里完全不成立**——a1 的消融是 3 组、每组用 13 MB 的 `data.in`、估算每组 15~30 分钟。

3. **仓库自带 `data.in` + `tokenizer.json` + `TinyStoriesV2-GPT4-valid.txt`，存在一条完全离线的跑通路径**，估计 20~60 分钟就能拿到第一个 PPL 曲线和生成文本——**不需要下载任何 HuggingFace 数据**。这是本作业最低成本的正反馈点。

---

# 补充章节（A）：两个 notebook 的「填空」核实 —— **结论：没有任何填空**

> 这一节回答「notebook 里到底有哪些待填空的练习 / markdown 说明」。
> 数据来源：`CS336_Assignment1_BPE.ipynb` 全量抓取（HTTP 200，无截断）；`CS336_Assignment1_Transformer.ipynb` HTTP 200，正文有 44135 bytes 未内联显示，但**完整结果已被 harness 落盘到 spill 文件**，我用 `grep`/`read` 对该完整文件做了穷尽检索。

## A.1 直接结论（这是最重要的判断）

> ### ❌ **两个 notebook 里都没有任何「待填空」的练习。全是成品代码。**
> ### `NotImplementedError` = **0 处**；`TODO` / `FIXME` / `YOUR CODE HERE` / `pass` 占位 = **0 处**。

**检索方式（原文事实）**：对完整内容跑正则 `NotImplementedError|测试框架|pytest|test_|TODO|提交|评分`，命中项**全部**是 notebook 自己写的测试函数名（`test_weights`、`test_input`、`test_implementation`、`test_gradient_clipping_logic`）和两条注释，**没有一处是填空标记**。

**旁证（原文事实）**：Transformer notebook 的 code cell 带有大量**真实执行计数与输出**：
```
execution_count: 1, 2, 3, 4, 5, 6, 8, 12, 15, 45,
                 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 472, 473, 475
```
且输出了真实张量打印结果（如 `输出形状: torch.Size([1, 2048, 8])`、`Index 2匹配: True`、`掩码功能测试通过`、`验证结果是否一致：True`）。
BPE notebook 同样带有 `execution_count` 与真实 stdout（
`✅ BPE Tokenizer训练完成` / `💾 分词器已保存至./bpe_tokenizer/tokenizer.json` / `['ĠHello', ',', 'Ġworld', '!', 'Ġ', '<|endoftext|>']`）。
→ **被执行过、有真实输出 = 成品代码，不是骨架。**

## A.2 `CS336_Assignment1_BPE.ipynb` 全部 markdown cell（原文，共 8 个）

| # | markdown 原文内容 |
|---|---|
| 1 | `一、依赖安装` |
| 2 | `二、分块读取工具函数` |
| 3 | `三、按块读取文本` |
| 4 | `四、训练BPE分词器` |
| 5 | `Step1:在不修改tokenizer内部实现的前提下，实时监控内存占用与数据吞吐量，理解tokenizer训练的真实系统行为。` |
| 6 | `Step2:训练BPE Tokenizer` |
| 7 | `五、验证训练的BPE Tokenizer` |
| 8 | `Q: 为什么在GTP-2正则化风格中训练出来的tokenizer会在单词前添加一个'Ġ'？`<br>`A: 因为对于模型而言，在同一个单词中，没有带空格和带有空格表示的Token是不同的，因此为了区分这两种情况就会添加了一个'Ġ'来表示带空格的Token（比如"hello"和"Ġhello"，前者表示"hello"这个单词，后者表示" hello"这个单词）。` |

**特征**：第 1~7 个是**章节编号 / 步骤标签**（纯导航），第 8 个是**概念问答**。
**没有任何一个 markdown cell 是「任务说明」「实现要求」「请完成下面的函数」之类。**
`Step1` 那条虽然看起来像任务，但原文写的是「监控…理解…系统行为」，**是学习目标陈述，不是编码任务**；而且紧跟着的 code cell 已经把 `get_memory_mb()` / `log_status()` 全实现了。

## A.3 `CS336_Assignment1_Transformer.ipynb` 全部 markdown cell（原文，共 6 个）

> 该 notebook 结构核算：`"cell_type": "markdown"` 命中 6 处，`"cell_type": "code"` 命中 28 处，合计 **34 个 cell**。

| # | markdown 原文内容（摘要） |
|---|---|
| 1 | `Transformer架构` + 一张「阶段/输入维度/输出维度」对照表：<br>`嵌入层(Embedding) (B,L) → (B,L,D)`、`位置编码 (B,L,D) → (B,L,D)`、`中间运算层 (B,L,D) → (B,L,D)`、`分类层(Linear) (B,L,D) → (B,L,V)`；<br>后接「每个板块的核心本质：」4 条 bullet（嵌入层=特征映射、位置编码=注入位置信息打破词袋、中间运算层=Self-Attention & MLP、分类层=映射回词表算下一个词概率） |
| 2 | `一、基本板块的构建` |
| 3 | `Q: 反向传播中更新改变了什么？`<br>`A: 在反向传播中，更新权重改变的是单词的'内涵'，而不是'名字'` + 1 条「不变关系」+ 1 条「数值变换」 |
| 4 | `Q: RMSNorm的作用和优势是什么？`<br>`A:` 很长一段，含 BatchNorm vs LayerNorm vs RMSNorm 对比、`RMSNorm vs. LayerNorm：从"全处理"到"精简化"`、`RMSNorm的核心优势 A/B/C`（计算效率的飞跃 / 更加温和的残差分支保护 / 保持特征表示强度）、以 `💡定义：RMSNorm是一种计算成本更低…` 收尾 |
| 5 | `Q: 旋转位置编码的优势在于什么，以及theta如何选择？`<br>`A:` 前半段（RoPE 核心优势、相对位置、cos/sin 可缓存为 buffer、与 KV cache 的兼容性）正常；<br>⚠️ **从「其中，theta是控制旋转频率分布的超参数：theta越大，整体旋转频率」之后，正文被大段无意义字符污染**（原文如此，形如 `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pfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwCgx4iIiJyCQx6iIiIyCUw6CEiIiKXwKCHiIiIXAKDHiIiInIJDHqIiIjIJTDoISIiIpfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwCgx4iIiJyCQx6iIiIyCUw6CEiIiKXwKCHiIiIXAKDHiIiInIJDHqIiIjIJTDoISIiIpfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwCgx4iIiJyCQx6iIiIyCUw6CEiIiKXwKCHiIiIXAKDHiIiInIJDHqIiIjIJTDoISIiIpfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwCgx4iIiJyCQx6iIiIyCUw6CEiIiKXwKCHiIiIXAKDHiIiInIJDHqIiIjIJTDoISIiIpfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwCgx4iIiJyCQx6iIiIyCUw6CEiIiKXwKCHiIiIXAKDHiIiInIJDHqIiIjIJTDoISIiIpfAoIeIiIhcAoMeIiIicgkMeoiIiMglMOghIiIil8Cgh4iIiFwC///Z` / 一串 base64 噪声）。<br>**这一段无法作为可信内容引用。** |
| 6 | `二、 支持Transformer训练的板块构建` |

**特征**：6 个 markdown cell 中，1 个是架构总览表，2 个是章节标题，**3 个是概念问答（Q&A）**。
**没有任何一个 markdown cell 出现「练习」「填空」「实现要求」「TODO」「你的任务」之类的字样。**
`二、支持Transformer训练的板块构建` 之后的 6 个 code cell（cross_entropy / AdamW / get_lr_cosine_schedule / run_gradient_clipping / get_batch / save+load_checkpoint）**完全没有配 markdown 说明**，只有 cell 内的 `# 1.` `# 2.` … 编号注释，且全部已实现。

## A.4 对结论的影响（对「手搓实际工作量」的判断）

**这显著改变了「手搓」的性质：**

| 判断维度 | 结论 |
|---|---|
| 有没有「填空题」？ | **没有。** 0 处 `NotImplementedError`、0 处 `TODO`、0 处占位 `pass` |
| 「手搓」的实际含义是什么？ | **不是「补全骨架」，而是「照着成品答案自己重写一遍」** |
| 需要自己写的组件数量 | **明面上 15 个**（第一节 9 个 + 第二节 6 个），另加 notebook 里**没有**但 `model.py` 里有的 `TransformerLM` |
| 有没有「隐藏测试」逼你真的写对？ | **没有。** 没有任何自动化校验，抄错也没人拦 |
| 学习价值从哪来？ | 完全依赖**自律**：关掉参考实现自己写，再用 notebook 里的 `run_*`/`test_*` cell 自测 |
| 风险 | notebook 里连测试都是「成品」——`test_implementation()`、`test_gradient_clipping_logic()` 直接给了断言与预期输出，**照抄即通过**，无法形成有效反馈 |

> **一句话**：diy-llm 的 a1 是一份**「答案册 + 讲解」**，不是「题册」。它的正确用法是**先盖住答案自己写，再用它对照**；把它当作业逐格填空是找不到空格的。

## A.5 补充章节（B）：`Assignment1_Basics.pdf`

**❌ 抓取失败。**

- URL：`https://raw.githubusercontent.com/datawhalechina/diy-llm/main/coursework/assignment1-basics/Assignment1_Basics.pdf`
- HTTP 响应：**200**（文件确实存在，419,506 B）
- 失败原因（原文错误信息）：`Error: unsupported content type "application/octet-stream"`
- 说明：`web_fetch` 工具不解析 PDF 二进制流。**因此我无法给出该 PDF 里的作业题目清单。**
- **我没有用任何先验知识去猜这份 PDF 的内容。** 它是 diy-llm 仓库里唯一可能承载「官方作业要求 / 评分标准 / 算力建议」的文档，**这个空白是本报告最大的信息缺口**。
- 同类失败：Stanford 原版 `cs336_assignment1_basics.pdf` 也是 `unsupported content type`，同样抓取失败。

**可行的替代获取方式（未执行，留给下游决定）**：用本地 `curl -o` 下载 PDF 到工作区，再用 PDF 解析工具（如 `pypdf`）提取文本。
