# CS336 作业4（assignment4-data，预训练数据处理）测绘报告

**测绘对象**：`datawhalechina/diy-llm` → `coursework/assignment4-data/`（Stanford CS336 Spring 2025 中文共建版）
**测绘日期**：本次会话
**结论一句话**：全部过滤/去重/分类器作业都是 **纯 CPU**，唯一的 GPU 环节是最后用 `cs336-basics` 训 GPT-2 small；单张 RTX 4070 Super 12GB **够用**，真正的瓶颈是 **375 GB 的 Common Crawl WET 数据** 和 **仓库里缺失的训练入口 `train.py`**。
**`get_assets.sh` 下载量（字节级已核实）**：**只有 2 个文件，合计 1,983,999,558 字节 = 1.848 GiB ≈ 1.98 GB**（991,753,734 + 992,245,824），且**不含任何 Common Crawl 数据**。详见第 5.1 节。

> **标注约定**
> - 【原文】= 本次实际抓取到的仓库/网页原文（含中文 notebook 里的 handout 翻译）
> - 【我的推断】= 我从原文推导出的结论，不在抓取内容中
> - 【我的估计】= 我给出的数值估计，非原文
> - 【抓取失败】= 见文末「抓取失败项」

---

## 1. 作业任务清单（原文，来自仓库 notebook 的 handout 中文翻译）

本作业的题目分节保存在两个 notebook 里：`cs336_data/作业一.ipynb`（含作业一～作业七）与 `cs336_data/作业二.ipynb`（含作业八～作业十）。**注意：这里的"作业一/二/三…"是 CS336 Problem 编号，不是独立的大作业。**

| 编号 | 题目 | 分值（原文） | 交付物（原文） |
|---|---|---|---|
| 作业一 | `look_at_cc` | 4 分 | (a) 2-3 句回应；(b) 3-4 句回应；(c) 1-2 句回应；(d) 对 25 个 WET 文档的简短注释（语言、域名、页面类型等） |
| 作业二 | 提取文本（`extract_text`） | 3 分 | (a) 实现适配器 `[run_extract_text_from_html_bytes]`，通过 `pytest -k test_extract_text_from_html_bytes`；(b) 2-3 句对比回应 |
| 作业三 | 语言识别 | 6 分 | (a) 实现适配器 `[run_identify_language]`，通过 `pytest -k test_identify_language`；(b) 2-5 句；(c) 20 个随机样本人工标注对比 + 英语占比 + 置信度阈值建议，2-5 句 |
| 作业四 | 屏蔽 PII | 3 分 | 1) `[run_mask_emails]`；2) `[run_mask_phone_numbers]`；3) `[run_mask_ips]`；4) 2-5 句分析；5) 20 个样本假阳性/假阴性分析，2-5 句 |
| 作业五 | 有害内容 | 6 分 | 1) `[运行NSFW分类]`（中文译名，原文即如此）；2) `[运行分类有毒言论]`；3) 2-5 句；4) 20 个样本对比 + 有害比例 + 置信度阈值，2-5 句 |
| 作业六 | `gopher_quality_filters` | 3 分 | (a) `[run_gopher_quality_filter]`，通过 `pytest -k test_gopher`；(b) 20 个样本对比，2-5 句 |
| 作业七 | `quality_classifier` | 15 分 | (a) 训练一个质量分类器；(b) `[run_classify_quality]`，通过 `pytest -k test_classify_quality` |
| 作业八 | `exact_deduplication` | 3 分 | `[run_exact_line_deduplication]`，通过 `pytest -k test_exact_line_deduplication` |
| 作业九 | MinHash 去重 | 8 分 | `[run_minhash_deduplication]`，通过 `pytest -k test_minhash_deduplication` |
| 作业十 | `filter_data` | 6 分 | (a) 并行过滤 CC WET 的脚本 + **每个过滤步骤丢弃比例的书面说明**；(b) **数据过滤管道的运行时间**（5000 个 WET 要多久？整个 CC 10 万个 WET 要多久？） |

**合计 57 分**（【我的推断】= 我按原文分值相加；抓取内容中未出现总分，也未出现"隐藏测试"字样）。

**作业十原文（关键约束）**：
- 「我们在 `/data/CC/CC*.warc.wet.gz` 位置为你准备了 **5000 个 WET 文件**作为起点」
- 「给定你过滤后的数据集，你将在此数据上训练一个 **GPT-2 small 形状的模型，进行 20 万次迭代**，并评估其在 C4 100 上的困惑度」
- 「**你不应修改模型架构或训练过程**，因为目标是构建最佳的**数据**」
- 「**允许**使用 Paloma 验证数据来构建过滤器或分类器，但**不允许**将验证数据直接复制到你的训练数据中」
- 「即使是 5000 个 WET 文件也是大量的数据，**约 375GB 的压缩文本**」
- 推荐用 `fastwarc` 遍历 WET、`tldextract` 抽域名；并行用 `concurrent.futures` 或 `submitit`

**作业七原文**：Wikipedia 参考 URL 文件在 `/data/wiki/enwiki-20240420-extracted_urls.txt.gz`，「该文件包含截至 2024 年 4 月在英语维基百科页面上找到的 **43,500 个外部链接**的列表」。

> 【我的推断】注意 handouts 说 43,500 个 URL，但 `作业一.ipynb` 里实际跑该文件的进度条显示总行数是 **43,579,680**（`Processing URLs: 0%| | 191/43579680`），作者加注「使用以下官方给的命令，**并发太大，消耗内存导致 wsl 崩溃**！」并中断（KeyboardInterrupt）。所以本地必须二次抽样。

---

## 2. 要写哪些模块/函数（原文，函数名级别）

### 2.1 测试要求的适配器（`tests/adapters.py`，**这是官方评分入口**）

原文里以下签名已存在，其中 6 个是 `raise NotImplementedError`：

| 函数签名（原文） | 仓库当前状态 |
|---|---|
| `run_extract_text_from_html_bytes(html_bytes: bytes) -> str \| None` | 已实现，但 **`return 1`**（原文如此，明显是 bug，会把 int 返回给断言 `moby_expected_text == ...`） |
| `run_identify_language(text: str) -> tuple[Any, float]` | 已实现，用 **`langdetect.detect_langs`**（不是 fastText） |
| `run_mask_emails(text: str) -> tuple[str, int]` | 已实现（含一行调试 `print("ENTER run_mask_emails")`） |
| `run_mask_phone_numbers(text: str) -> tuple[str, int]` | 已实现 |
| `run_mask_ips(text: str) -> tuple[str, int]` | 已实现 |
| `run_classify_nsfw(text: str) -> tuple[Any, float]` | **`raise NotImplementedError`** |
| `run_classify_toxic_speech(text: str) -> tuple[Any, float]` | **`raise NotImplementedError`** |
| `run_classify_quality(text: str) -> tuple[Any, float]` | **`raise NotImplementedError`** |
| `run_gopher_quality_filter(text: str) -> bool` | **`raise NotImplementedError`** |
| `run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike)` | **`raise NotImplementedError`** |
| `run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)` | **`raise NotImplementedError`** |

### 2.2 `cs336_data/filter.py`（原文，仓库里真正的实现文件）

```
run_extract_text_from_html_bytes(html_bytes: bytes) -> str          # 用 resiliparse
run_mask_emails(text: str) -> Tuple[str, int]
run_mask_phone_numbers(text: str) -> Tuple[str, int]
run_mask_ips(text: str) -> Tuple[str, int]
run_gopher_quality_filter(text: str) -> bool
```
Gopher 规则（原文注释）：字数 <50 或 >100000；平均词长超出 3~10；>30% 的行以 `...` 结尾；含字母的词 <80%。

### 2.3 `cs336_data/extracted_data.py`（原文，WET → 训练样本）

```
desensitize_text(text) -> str
process_text(text) -> Optional[str]
classify_quality(model, text) -> Tuple[str, float]
extract_text_from_wet_record(record_lines: List[str]) -> Optional[str]
parse_wet_file(file_obj) -> List[List[str]]
extract_samples_from_wet(wet_paths, target_count, model_path="quality_classifier.bin",
                         quality_threshold=0.5, random_seed=42) -> Tuple[List[str], List[str], dict]
find_wet_files(datasets_dir: Path) -> List[str]        # 只 glob "*.warc.wet.gz"
save_samples(hq_samples, lq_samples, output_dir: Path)
main()                                                  # target_count = 10_0000, quality_threshold = 0.8
```

### 2.4 `cs336_data/cc2train.py`（原文，WARC → 负样本）

```
desensitize_text(text) -> str
extract_text_from_warc_record(record) -> Optional[str]   # 只取 rec_type=='response' 且 Content-Type 含 text/html
process_text(text) -> Optional[str]
extract_negative_samples_from_warc(warc_paths, target_count, random_seed=42) -> List[str]
count_positive_samples(positive_file) -> int
find_warc_files(datasets_dir: Path) -> List[str]
main()
```

### 2.5 `cs336_data/train_fasttext.py`（原文，质量分类器训练）

```
merge_and_shuffle_samples(positive_file, negative_file, output_file, shuffle=True, random_seed=42) -> int
split_train_val(merged_file, train_file, val_file, val_ratio=0.1, random_seed=42) -> Tuple[int, int]
train_quality_classifier(train_path, model_path, val_path=None, **kwargs) -> fasttext.FastText
main()
```
原文训练超参：`lr=0.1, epoch=25, wordNgrams=2, dim=100, loss='softmax', minCount=5, bucket=2_000_000, thread=8`。

### 2.6 `cs336_data/run_identify_language.py`（原文）

```
_load_model()                       # @lru_cache(maxsize=1)，加载 lid.176.bin
run_identify_language(text)         # fasttext.predict(k=1)，zh*→"zh"
```
硬编码路径：`MODEL_PATH = "/mnt/d/项目/cs336/CS336-Chinese-co-construction/coursework/Assignment4_Data/cs336_data/lid.176.bin"`（与仓库实际文件名 `lid.176.ftz` 不一致）。

### 2.7 只存在于 notebook 里的代码（**不在任何 .py 文件里**）

`作业二.ipynb` 中的去重实现：
```
_line_hash(line) -> str
exact_deduplication(file_paths: List[str]) -> None
normalize_text(text) -> str
get_shingles(text: str, n: int) -> list[str]
estimate_jaccard(sig1: list, sig2: list) -> float
run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
```
测试语料生成器：`random_sentence`、`generate_random_document`、`generate_template_document`、`perturb_text`、`generate_test_corpus`。
调用示例（原文）：`run_minhash_deduplication(input_files=input_files, num_hashes=100, num_bands=20, ngrams=5, jaccard_threshold=0.8, output_directory='deduplicated_output')`。

`作业一.ipynb` 中的正样本清洗：
```
read_extracted_files(extracted_dir) -> Generator[dict, None, None]
desensitize_text(text) -> str
```
（以及 `train_fasttext.py` / `cc2train.py` 的 `%%writefile` 源文）

### 2.8 notebook 未覆盖的部分

【原文】两个 notebook 到「作业十」的 WET 提取就结束了，**没有 tokenize、没有训练 LM、没有提交排行榜的代码**。
`cs336-basics/configs/experiment/your_data.yaml` 只是把路径指向上一作业的产物：
```yaml
train_bin: .../cs336_data/tokenized/train.bin
valid_bin: .../cs336_data/tokenized/val.bin
wandb_entity: kangkang262614
wandb_project: cs336-data
```

---

## 3. 交付物（原文）

- **代码**：`tests/adapters.py` 里 11 个适配器函数的实现 + 一个数据过滤脚本（作业十 a）
- **书面回答**：约 15 处 1-5 句的问答（作业一 a/b/c、作业二 b、作业三 b/c、作业四 4/5、作业五 3/4、作业六 b、作业十 a 的丢弃比例说明）
- **作业一 d**：25 个 WET 文档的人工注释表
- **作业十 b**：过滤 5000 个 / 10 万个 WET 的**运行时间**
- **质量分类器模型文件**：`quality_classifier.bin`（fastText）
- **排行榜提交**（原文来自 [leaderboard README](https://raw.githubusercontent.com/stanford-cs336/assignment4-data-leaderboard/main/README.md)）：PR 里要包含 ①最终验证损失 ②**带 wallclock-time 的学习曲线链接** ③做法描述；「**Make sure you save a snapshot of your best data pipeline so it can be reproduced by us!**」；前 5 名会被复现重排。
- **提交打 zip**：`cs336-spring2025-assignment-4-submission.zip`

**无图表类交付物要求**（除排行榜要求的学习曲线）；抓取内容中未见"必须提交数据集本身"的要求。

---

## 4. 官方测试与评分方式

### 4.1 `test_and_make_submission.sh` 原文（完整）

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

# Set the name of the output tar.gz file
output_file="cs336-spring2025-assignment-4-submission.zip"
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

**原文事实**：`|| true` 意味着 pytest 失败**不会**让脚本退出；zip 排除 `*.txt`、`*.bin`、`*.json`、`*.pt` 等。
> 【我的推断】① 若把书面回答写成 `.txt`，会被 zip 排除，等于没交；② `quality_classifier.bin` 因 `*.bin` 也被排除；③ 因为 `|| true`，脚本本身不构成"评分"，真正评分是助教跑测试/复现。

### 4.2 逐题测试命令（原文，来自 handout 翻译）
`uv run pytest -k test_extract_text_from_html_bytes` / `test_identify_language` / `test_mask_emails` / `test_mask_phones` / `test_mask_ips` / `test_classify_nsfw` / `test_classify_toxic_speech` / `test_gopher` / `test_classify_quality` / `test_exact_line_deduplication` / `test_minhash_deduplication`

### 4.3 测试清单（原文函数名 + 我逐条判定的当前状态）

共 **21 个 test 函数**（【我的推断】= 我数出来的）：

| 文件 | test 函数 | 测什么（原文） | 当前能否通过 |
|---|---|---|---|
| `test_extract.py` | `test_extract_text_from_html_bytes` | 读 `fixtures/moby.html`，断言等于 `fixtures/moby_extracted.txt` | ❌ 缺 `moby_extracted.txt`，且 adapters 返回 `1` |
| `test_deduplication.py` | `test_exact_line_deduplication` | 断言输出 5 个文件且内容与 `documents_line_deduplicated/` 一致 | ❌ 缺 fixtures 目录 + NotImplementedError |
| | `test_minhash_deduplication_exact_duplicates` | 断言输出 4 个文件（doc2 与 doc1 完全重复） | ❌ 同上 |
| | `test_minhash_deduplication_fuzzy_duplicates` | 断言输出 2 个文件，两个 MIT license 模糊重复只留 1 个 | ❌ 同上 |
| `test_langid.py` | `test_identify_language_english` | 读 `fixtures/moby_extracted.txt`，断言 `"en"` 且 score>0 | ❌ 缺 fixture |
| | `test_identify_language_chinese_simplified` | `"欢迎来到我们的网站"` → `"zh"` 且 score>0 | ✅【我的推断】langdetect 返回 `zh-cn`，被映射为 `zh` |
| `test_pii.py` | `test_mask_emails_single` / `_multiple` / `_existing_string` | 断言 `\|\|\|EMAIL_ADDRESS\|\|\|` 与数量；第 3 个含"已存在的占位串"不能误数 | ✅【我的推断】正则能过 |
| | `test_mask_phones_single` | 4 种美国格式：`2831823829` / `(283)-182-3829` / `(283) 182 3829` / `283-182-3829` | ✅【我的推断】 |
| | `test_mask_ips` | `192.0.2.146` → `\|\|\|IP_ADDRESS\|\|\|` | ✅【我的推断】 |
| `test_quality.py` | `test_classify_quality` | `low_quality_cc.txt`→`"cc"`，`high_quality_wiki_reference.txt`→`"wiki"` | ❌ 缺 fixtures + NotImplementedError |
| | `test_gopher_valid_input` | 100 次重复的高质量串应通过 | ❌ NotImplementedError |
| | `test_gopher_less_than_50_non_symbol_words` | 短文本拒绝、重复 100 次接受 | ❌ |
| | `test_gopher_more_than_100000_non_symbol_words` | 5 万次重复拒绝、5 千次接受 | ❌ |
| | `test_gopher_average_word_length_less_than_3` | `"the be "` 拒绝、`"the with "` 接受 | ❌ |
| | `test_gopher_average_word_length_greater_than_10` | 超长词拒绝 | ❌ |
| | `test_gopher_more_than_30_percent_lines_ending_with_ellipsis` | 70/100 省略号拒绝、30/260 接受 | ❌ |
| | `test_gopher_less_than_80_percent_words_with_alphabetic_character` | 8 数字 + 2 词 → 拒绝 | ❌ |
| `test_toxicity.py` | `test_classify_nsfw` | 两句 Jigsaw 样本 → `"nsfw"` / `"non-nsfw"` | ❌ NotImplementedError |
| | `test_classify_toxic_speech` | 两句 → `"toxic"` / `"non-toxic"` | ❌ NotImplementedError |

**关键事实**：`tests/fixtures/` 经 GitHub API 核实**只有 1 个文件** `moby.html`（1256 字节）。测试所需的 `moby_extracted.txt`、`documents_with_line_duplicates/`、`documents_line_deduplicated/`、`documents_with_fuzzy_duplicates/`、`low_quality_cc.txt`、`high_quality_wiki_reference.txt` **全部不存在**。
> 【我的推断】仓库里的这段测试套件**开箱即用是跑不通的**：约 6/21 能过，15/21 必然失败。缺的是上游 Stanford 原始仓库的 fixtures，diy-llm 没有搬进来。

### 4.4 隐藏测试
【原文】抓取到的 README / CHANGELOG / 测试文件 / handout 翻译中，**均未出现"隐藏测试 / hidden test"字样**。
> 【我的推断】CS336 通常有隐藏测试，但**本次抓取内容中没有证据**，不作为事实陈述。

### 4.5 官方排行榜（原文，外部）
[stanford-cs336/assignment4-data-leaderboard](https://github.com/stanford-cs336/assignment4-data-leaderboard)：提交 PR 加到 Markdown 表格，按 loss 升序；`naive baseline = 4.00`（Verified），当前最优 `3.19456`（Verified）。

---

## 5. 数据下载量与磁盘需求

### 5.1 【原文】`get_assets.sh` 全文与它下载的东西

```bash
#!/bin/bash
SOURCE_DIR="/data/classifiers"
ASSETS_DIR="$(pwd)/cs336_data/assets"

handle_file() { ... if [ -e "$ASSETS_DIR/$filename" ]; then skip
                 elif [ -f "$SOURCE_DIR/$filename" ]; then ln -s ...
                 else wget "$url" -O "$ASSETS_DIR/$filename"; fi }

handle_file "dolma_fasttext_nsfw_jigsaw_model.bin" \
  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
handle_file "dolma_fasttext_hatespeech_jigsaw_model.bin" \
  "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"
```

**原文事实**：
- 只下载 **2 个** Dolma Jigsaw fastText 分类器模型；
- **脚本里没有写任何文件大小**；
- 优先从集群软链 `/data/classifiers`，本地不存在才 wget。

### 【原文】`get_assets.sh` 两个下载文件的**精确大小（已核实到字节级）**

`get_assets.sh` 自身不含大小，但两个下载 URL 对应的 HF 仓库元数据可以读到文件大小。
`huggingface.co` 直连在本环境失败（解析到非公网 IP），**改用镜像 `hf-mirror.com` 的 API 成功（HTTP 200）**：

| 文件（脚本里的本地名） | URL（get_assets.sh 原文） | `model.bin` 精确大小（HF 元数据原文） |
|---|---|---|
| `dolma_fasttext_nsfw_jigsaw_model.bin` | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin` | **991,753,734 字节**（sha256 `dad02e9a612b85643d8f9acfcf810dcc26abb3df111b3ccb51619e5d3626ac27`） |
| `dolma_fasttext_hatespeech_jigsaw_model.bin` | `https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin` | **992,245,824 字节**（sha256 `3b4b994163f5ec6955a722c0c93cbadd6828a023249b1fdffeaf681659607452`） |
| **`get_assets.sh` 合计下载量** | — | **1,983,999,558 字节 = 1.848 GiB ≈ 1.98 GB**（【我的推断】= 我把上面两行相加） |

两个仓库还各带 `.gitattributes`(1565 B)、`README.md`(30 B)、`requirements.txt`(15 B)，可忽略；两个仓库都 `gated: false`、`license: cc-by-sa-3.0`。

**交叉验证（关键）**：`作业一.ipynb` 从 `dolma-artifacts.org` 下载的同族模型，wget 实测 `Length:` 分别是 **991,753,734** 与 **992,245,824** 字节 —— 与上表 **逐字节完全一致**。
> 【我的推断】两个来源字节数完全相同，说明 `get_assets.sh` 经 HF 拿到的就是 Dolma 官方那两个 Jigsaw 分类器（CHANGELOG `[0.0.1]` 也提到「fix link to Dolma NSFW and hatespeech classifiers, since the HF links point to the same model binary」）。严格讲这是"字节数一致"而非"哈希一致"，但两处两模型都一致，可信度很高。
> 【我的推断】脚本没有 `mkdir -p "$ASSETS_DIR"`，若 `cs336_data/assets/` 不存在，`wget -O` 和 `ln -s` 都会失败。

**一句话回答"get_assets.sh 下载量"**：**恰好 2 个文件，合计 1,983,999,558 字节（≈1.85 GiB / 1.98 GB）**；脚本本身不写大小，这个数字来自两个 HF 仓库元数据，并与 notebook 里同族模型的实测字节数一一吻合。`get_assets.sh` **完全不含 Common Crawl 数据**，也不含 `lid.176.ftz`（后者已直接提交在仓库里，938,013 字节）。

### 5.2 【原文】`作业一.ipynb` 里实测的下载大小（wget 的 `Length:` 行）

| 资产 | URL（原文） | 实测大小（原文） |
|---|---|---|
| CC WARC 分片 | `https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz` | **1,121,135,968 B = 1.04 GB** |
| CC WET 分片 | `.../CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz` | **80,904,812 B = 77 MB** |
| CC WARC ×3 | `.../CC-MAIN-2025-51/segments/1764871306713.64/warc/CC-MAIN-20251204191828-20251204221828-0000{0,1,2}.warc.gz` | **872 MB / 838 MB / 865 MB** |
| CC WET ×6 | `.../CC-MAIN-2025-51/.../wet/CC-MAIN-20251204191828-20251204221828-0000{0..5}.warc.wet.gz` | 75,410,076 / 76,619,390 / 74,196,199 / 74,965,696 / 75,273,977 / 76,312,320 B ≈ **452 MB 合计** |
| WET 路径清单 | `https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-51/wet.paths.gz` | **200,800 B = 196 KB** |
| fastText 语言识别（全量） | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin` | **131,266,198 B = 125 MB** |
| fastText 语言识别（压缩版） | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` | **938,013 B = 916 KB** |
| NSFW 分类器 | `dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin` | **991,753,734 B = 946 MB** |
| 仇恨言论分类器 | `dolma-artifacts.org/.../jigsaw_fasttext_bigrams_hatespeech_final.bin` | **992,245,824 B = 946 MB** |
| 维基外链 URL 清单 | `https://downloads.cs.stanford.edu/nlp/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz` | **1,055,989,383 B = 1007 MB** |
| 英文维基全量 dump | `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2` | **24,796,216,934 B = 23 GB**（作者实际改用分片：articles1 = 280 MB、articles10 = 563 MB、articles11 = 540 MB） |

**仓库里已提交的资产**：`cs336_data/lid.176.ftz` = **938,013 字节**（API 核实，与上面 wget 到的 ftz 同尺寸）。

### 5.3 【原文】Common Crawl 要下多少
- handout：「我们在 `/data/CC/CC*.warc.wet.gz` 位置为你准备了 **5000 个 WET 文件**作为起点」
- handout：「即使是 5000 个 WET 文件也是大量的数据，**约 375GB 的压缩文本**」
- 作业十 (b)：「过滤整个 Common Crawl 数据转储（**100,000 个 WET 文件**）需要多长时间？」

### 5.4 【我的估计】磁盘需求（关键：这些数字是估计，不是原文）

| 场景 | 下载量 | 需预留磁盘 | 依据 |
|---|---|---|---|
| **最小跑通**（作业一~九：看样本、实现函数、过测试） | WET 1 个 77 MB + WARC 1 个 1.04 GB + 2 个 Jigsaw 模型 1.85 GB + lid.176.ftz 0.9 MB ≈ **3.0 GB** | **8-10 GB**（含 uv 环境 torch CUDA ~7-9 GB） | 我按 5.2 的原文数值相加 |
| **作业七质量分类器**（需爬 wiki 正样本 + 训 fastText） | 上面 + URL 清单 1007 MB + 抽样网页 | **15-25 GB** | 我估；fastText `bucket=2e6, dim=100` 的 bucket 表本身就 ~800 MB 内存，模型落盘 ~1 GB 级 |
| **作业十完整版（5000 WET）** | WET 375 GB + 过滤后文本 | **1.2-1.5 TB**（375 GB 原始 + 解压中间态 + tokenized bin） | 我估：解压/中间态通常 1.5-2× 压缩体积，GPT-2 tokenized uint16 约 2 B/token |
| 整个 CC（10 万 WET） | ~7.5 TB | — | 【我的估计】= 5000 个 375 GB × 20 |

> **可行性判据（我的推断）**：单机做"完整版作业十"需要 ~1.2 TB 以上可用磁盘 + 长时间下载，**这是绝大多数个人机的真实门槛，不是 GPU 门槛**。作业十 (b) 只要求"估算"整个 CC 的时间，所以**不必真下 10 万个 WET**。

---

## 6. 是否需要 GPU

| 环节 | 是否需要 GPU | 依据（原文） |
|---|---|---|
| 作业一~九全部（正则、resiliparse 抽文本、langdetect/fastText 推理、Gopher 规则、exact/minhash 去重、20 样本人工标注） | **不需要，纯 CPU** | 测试与实现里没有任何 CUDA 调用 |
| `train_fasttext.py` 训质量分类器 | **不需要 GPU**（fastText 只有 CPU 实现） | 原文超参含 `thread=8`，是 CPU 多线程 |
| Dolma Jigsaw / fastText 推理 | **不需要 GPU** | `fasttext.load_model` + `model.predict`，纯 CPU |
| 作业十的数据过滤 | **不需要 GPU**，但要 **多核 CPU + 大内存** | handout 原文推荐 `concurrent.futures` / `multiprocessing`、`submitit` 的 `cpus_per_task=2, mem_gb=2` |
| **训练验证（`cc2train.py` 产出的数据 → `cs336-basics` 训 LM）** | **需要 GPU** | `train_config.py`：`device: str = "cuda"`、`dtype: str = "bfloat16"`、`compile: bool = True` |

`cs336-basics/cs336_basics/train_config.py` 原文配置：
```python
# 注释掉的"原始"形状：
# vocab_size=50257, context_length=128, d_model=768, d_ff=2048, num_layers=12, num_heads=12
@dataclass  # 笔记本8G显存可以跑     ← 仓库作者原注释
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 128
    d_model: int = 384      # 减半
    num_layers: int = 6     # 减半
    num_heads: int = 6
    d_ff: int = 1024
    dropout: float = 0.1; weight_decay: float = 0.01; max_lr: float = 6e-4; warmup_steps: int = 2000

@dataclass
class TrainingConfig:
    dtype: str = "bfloat16"
    train_batch_size: int = 128
    train_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    compile: bool = True
    eval_iterations: int = 1_000
    eval_interval: int = 2_000
    device: str = "cuda"
```
`ddp_utils.py` 存在 `_setup_process_group` / `_cleanup_process_group`（多卡可用，但非必需）。

**CHANGELOG 原文时间线**：`[1.0.4] - 2025-05-19 · Changed: code: Halve training tokens for the leaderboard run`。
> 【我的推断】handout 翻译里写"20 万次迭代"，而仓库 `train_config.py` 是 `train_steps = 100_000`，两者对应 CHANGELOG 的"训练 token 数减半"。

**重要缺口（原文树核实）**：`cs336-basics/` 的递归 API 树只有 15 个 entry —— `README.md`、`pyproject.toml`、`uv.lock`、`configs/*`、`cs336_basics/{__init__,data,ddp_utils,model,optimizer,train_config}.py`。**没有 `train.py`，`pyproject.toml` 里也没有 `[project.scripts]` 入口。**
`configs/config.yaml` 原文只有 5 行（`defaults: paths: base_paths / model: base_model / training: base_training / _self_`），而 `configs/` 目录树里**只有 `config.yaml` 和 `experiment/your_data.yaml`**，没有 `paths/`、`model/`、`training/` 子目录文件（这些组是在 `train_config.py:register_configs()` 里用 hydra ConfigStore 注册的）。
> 【我的推断】要跑通"训练验证/排行榜"这段，必须**自己补训练入口脚本**（或从 Stanford 原版仓库取）。这是本仓库的一个真实缺口。

---

## 7. 在单张 RTX 4070 Super 12GB 上能不能做

### 结论：【我的推断】**能做完（含训练验证），GPU 不是瓶颈；瓶颈是数据量、磁盘和 CPU 时间。**

**理由（逐项）**
1. 作业一~九 **100% 纯 CPU**：只要 CPU + 内存，12GB 显存完全不参与。4070 Super 机器的 CPU/内存才是关键（fastText 训练建议 ≥32 GB 内存；【我的估计】）。
2. 训练验证部分：仓库作者自己在 `train_config.py` 里注明减半后的模型「**笔记本8G显存可以跑**」——这是原文注释。4070 Super 12GB > 8GB，**必然够**。
3. 即使按"原始"GPT-2 small 形状（d_model 768 / 12 层 / ctx 128 / batch 128）跑：模型约 124M 参数，bf16 + AdamW 的优化器态约 1.5-2 GB，激活在 seq=128、batch=128 下很小。
   【我的估计】显存占用约 **3-6 GB**，12 GB 有余量；不会 OOM。
4. **算力也不是问题**：4070 Super 支持 bf16（Ada sm_89），`dtype="bfloat16"` 直接可用（这点比 T400 强：T400 是 sm_75、无 bf16）。100k steps × batch 128 × ctx 128 = **1.6384e9 tokens**（【我的推断】= 我算的）；GPT-2 small 每 step 约 1.22e13 FLOP → 总计 ~1.22e18 FLOP。
   【我的估计】在 4070 Super 上按 50-70 TFLOPS 有效算力，约 **5-7 小时**；换成仓库减半的模型约 **1-1.5 小时**。
5. **唯一"不能"的部分是数据规模，不是显存**：完整作业十 = 375 GB WET + ~1.2 TB 磁盘 + 数十小时下载/过滤。这个在**任何单卡机器上**都吃力，与 4070 Super 无关。

### 建议的路线（【我的推断】）
- **必做且能做完**：作业一~九 + 作业十的小规模版本（6-20 个 WET），过适配器 + 写分析题 + 本地训一个 fastText 质量分类器。
- **可跳过**：真正下载 5000 个 WET（375 GB）跑全量过滤；整个 CC（10 万 WET）本来 handout 也只要求估算时间。
- **训练验证**：用减半模型（作者注释"8G 可跑"）在 4070 Super 上完成，产出验证 loss；补 `train.py` 入口。

---

## 8. 工时估算（全部为【我的估计】，非原文）

| 阶段 | 估计耗时 | 说明 |
|---|---|---|
| 读 handout + 两个 notebook + 补 fixtures | 4-8 h | 中英对照 + 读懂 MinHash/LSH 节 |
| 实现 11 个适配器 + Gopher + 两种去重 | 6-12 h | 适配器本身简单，MinHash/LSH 最容易踩坑 |
| 作业一 d（25 文档注释）+ 各分析题（约 15 处） | 4-8 h | 纯写作 + 人工阅读 |
| 下载最小数据集（~3 GB） | 0.5-3 h | 取决于带宽；作者实测 1-9 MB/s |
| 下载 5000 WET（375 GB） | **10-100 h** | 作者实测 1.5-10 MB/s → 10 h @ 10 MB/s，52 h @ 2 MB/s，104 h @ 1 MB/s |
| 过滤 6 个 WET（作者实测） | **约 1-3 min** | 作者单进程 27.69 s/文件（~900-1000 rec/s） |
| 过滤 5000 WET（单进程） | 约 **38 h**（【我的估计】= 5000 × 27.7 s） | 用 8-16 进程并行降到 3-6 h |
| fastText 质量分类器（作者实测） | **22 分 42 秒**处理 780,806 篇 + 10 s 保存 | 前提是已有 247,635 条正样本 |
| 爬 wiki 正样本（43.58M URL 清单） | **不可行**，必须抽样 | 作者按官方命令直接崩 WSL |
| GPT-2 tokenize + 训练 100k steps | 1-7 h | 见第 7 节 |
| **合计（最小可交付版）** | **约 40-70 h** | |
| **合计（完整作业十含 375 GB）** | **约 100-200 h，且 70% 时间在下载** | |

---

## 9. 核心必做 vs 可跳过（逐项）

| 项 | 判定 | 理由 |
|---|---|---|
| 作业一 look_at_cc (a)(b)(c) | **可跳过/速做** | 纯读书报告，2-4 句话，不涉及代码 |
| 作业一 (d) 25 文档注释 | **半必做** | 4 分，但耗时；可缩到 10 篇并注明 |
| 作业二 `run_extract_text_from_html_bytes` | **必做** | 后续所有流程的前置 |
| 作业三 `run_identify_language` | **必做** | 有测试；仓库用 langdetect，handout 用 fastText |
| 作业四 三个 PII 函数 | **必做且最容易** | 仓库已实现，测试可过 |
| 作业四 5（假阳性分析） | **可跳过** | 纯写作 |
| 作业五 NSFW + toxic 适配器 | **必做** | 有测试；需从 `filter.py`/notebook 搬到 `adapters.py` |
| 作业五 4（20 样本分析） | **可跳过** | 纯写作 |
| 作业六 `run_gopher_quality_filter` | **必做** | 8 个测试；实现已在 `filter.py`，只需接线 |
| 作业七 quality_classifier (a)(b) | **必做，最重（15 分）** | 需要正样本爬取 + fastText 训练 |
| 作业八 `run_exact_line_deduplication` | **必做** | 仓库未实现（只在 notebook 里），需移植 |
| 作业九 `run_minhash_deduplication` | **必做** | 8 分；仓库未实现，需从 notebook 移植并修 bug |
| 作业十 (a) 过滤脚本 | **必做（小规模）** | 可只跑 6 个 WET 并报告比例 |
| 作业十 (a) 的书面丢弃比例 | **必做** | 明确写在交付物里 |
| 作业十 (b) 运行时间估算 | **必做且廉价** | 只需外推，不必真跑 10 万 WET |
| 真下 5000 WET（375 GB） | **可跳过** | 只为排行榜分数；本地磁盘/带宽不允许 |
| 排行榜提交 | **可跳过** | 需要 Stanford 集群/复现要求，个人机不现实 |
| 补 `train.py` + 训练验证 | **半必做** | 想拿"训练验证"这条就必须补；只求跑通流程可跳过 |

### 仓库当前状态的坑（原文核实，动手前必须知道）
1. `tests/fixtures/` **只有 `moby.html`**，缺 6 类 fixture → 15/21 测试必挂。
2. `tests/adapters.py` 里 6 个函数是 `NotImplementedError`，而实现散落在 `cs336_data/filter.py` 和两个 notebook 里，**没有接线**。
3. `adapters.py::run_extract_text_from_html_bytes` **`return 1`**（bug）。
4. `run_identify_language.py` 硬编码 Windows 路径 `/mnt/d/项目/.../lid.176.bin`，与仓库文件名 `lid.176.ftz` 不符；同一功能在 `adapters.py` 里用的是 langdetect（两套实现）。
5. `cs336-basics` **没有 `train.py` 与任何 CLI 入口**。
6. 两个 notebook 到"作业十的数据提取"就结束了，**没有 tokenize / 训练 / 提交代码**。
7. `README.md` 原文写「只需运行 `cs336_systems/作业1.ipynb` 和 `cs336_systems/作业2.ipynb`」，但实际路径是 `cs336_data/`（路径笔误，原文如此）。
8. `test_and_make_submission.sh` 用 `|| true` 吞掉测试失败，且 zip 排除 `*.txt` / `*.bin`。

---

## 10. 抓取失败项（全部列出，无一省略）

| # | URL / 资源 | 失败原因 |
|---|---|---|
| 1 | `https://stanford-cs336.github.io/spring2025-assignment4-data/` | cross-origin redirect 到 `http://cs336.stanford.edu`，工具不自动跟随 |
| 2 | `http://cs336.stanford.edu/spring2025-assignment4-data/` | **HTTP 404**（GitHub Pages "Page not found"） |
| 3 | `https://cs336.stanford.edu/spring2025-assignment4-data/` | **HTTP 404** |
| 4 | `https://stanford-cs336.github.io/spring2025/` | cross-origin redirect，同上 |
| 5 | `.../相关文档/cs336_spring2025_assignment4_data.pdf`（英文 handout，156 KB） | `Error: unsupported content type "application/octet-stream"` —— 工具不支持解析 PDF 二进制 |
| 6 | `.../相关文档/【翻译】cs336_spring2025_assignment4_data.pdf`（34 MB） | **未尝试下载**（英文版 PDF 已因 `application/octet-stream` 失败，同格式预期同样失败） |
| 7 | `.../相关文档/【脑图】cs336_spring2025_assignment4_data.jpeg` | **未抓取**（图片，非文本；不在任务要求的精读清单内） |
| 8 | `https://huggingface.co/api/models/allenai/dolma-jigsaw-fasttext-bigrams-nsfw` | `URL hostname "huggingface.co" resolves to a non-public IP address`（**已用镜像绕过并成功**，见下「已绕过」表） |
| 9 | `https://huggingface.co/api/models/allenai/dolma-jigsaw-fasttext-bigrams-nsfw?blobs=true` | `TypeError: fetch failed`（同上，已绕过） |
| 10 | `https://huggingface.co/api/models/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech?blobs=true` | `TypeError: fetch failed`（同上，已绕过） |
| 11 | `https://api.github.com/repos/datawhalechina/diy-llm/contents/coursework/assignment4-data/tests` | **HTTP 403 API rate limit exceeded**（不影响结论：7 个测试文件已逐个 raw 抓取成功，fixtures 清单已在此之前用 API 核实） |
| 12 | `.../uv.lock`（341 KB） | **未抓取**（仅为依赖锁文件，与本次测绘结论无关） |
| 13 | `.../cs336-basics/cs336_basics/model.py`（16 KB）、`optimizer.py`、`__init__.py`、`tests/__init__.py`、`tests/fixtures/moby.html`（内容） | **未逐一读取内容**；存在性与大小已通过 GitHub API 树核实 |
| 14 | Stanford 2025 官方 handout 全文（PDF 正文） | 因 #1-#4、#5 全部失败，**本次未能读到 2025 版 handout 原文**；所有 handout 内容均转引自仓库 notebook 内的中文翻译 |

### 已绕过的失败项（原 URL 失败 → 替代路径成功，结论不受影响）

| 原 URL | 失败原因 | 替代路径 | 结果 |
|---|---|---|---|
| `huggingface.co/api/models/allenai/dolma-jigsaw-fasttext-bigrams-nsfw`（含 `?blobs=true`） | 域名解析到非公网 IP / `fetch failed` | `https://hf-mirror.com/api/models/allenai/dolma-jigsaw-fasttext-bigrams-nsfw?blobs=true` | **HTTP 200** → `model.bin` = **991,753,734 字节** + sha256 |
| `huggingface.co/api/models/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech?blobs=true` | `fetch failed` | `https://hf-mirror.com/api/models/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech?blobs=true` | **HTTP 200** → `model.bin` = **992,245,824 字节** + sha256 |
| `api.github.com/repos/.../assignment4-data/tests`（#11） | HTTP 403 API rate limit exceeded | 7 个测试文件已逐个从 `raw.githubusercontent.com` 抓取成功；`tests/fixtures/` 清单在 403 之前已用 API 核实 | 结论不受影响 |
| 中文路径 `相关文档/【导读】....md` | 无需兜底，未失败 | `raw.githubusercontent.com` + URL 编码（`%E7%9B%B8%E5%85%B3%E6%96%87%E6%A1%A3/%E3%80%90%E5%AF%BC%E8%AF%BB%E3%80%91...`） | **HTTP 200**，直接成功；**未使用** GitHub API contents 兜底 |

### 抓取成功的完整清单（供核对）
`README.md`、`CHANGELOG.md`、`get_assets.sh`、`test_and_make_submission.sh`、`pyproject.toml`、`.gitignore`、`tests/adapters.py`、`tests/common.py`、`tests/test_extract.py`、`tests/test_deduplication.py`、`tests/test_langid.py`、`tests/test_pii.py`、`tests/test_quality.py`、`tests/test_toxicity.py`、`cs336_data/{__init__,filter,extracted_data,cc2train,train_fasttext,run_identify_language}.py`、`cs336_data/作业一.ipynb`、`cs336_data/作业二.ipynb`、`cs336-basics/README.md`、`cs336-basics/pyproject.toml`、`cs336-basics/configs/config.yaml`、`cs336-basics/configs/experiment/your_data.yaml`、`cs336-basics/cs336_basics/{train_config,data,ddp_utils}.py`、`相关文档/【导读】....md`、GitHub API：`assignment4-data/` 顶层、`cs336_data/`、`tests/fixtures/`、`相关文档/`、`cs336-basics` 递归树、`configs/` 递归树、HF 元数据镜像 `hf-mirror.com/api/models/allenai/dolma-jigsaw-fasttext-bigrams-{nsfw,hatespeech}?blobs=true`、[leaderboard README](https://raw.githubusercontent.com/stanford-cs336/assignment4-data-leaderboard/main/README.md)、[Stanford CS336 Spring **2026** assignment4-data README](https://raw.githubusercontent.com/stanford-cs336/assignment4-data/refs/heads/main/README.md)（**注意是 2026 版**：写着 "The final training run uses 8 B200 GPUs"、默认下载 2.5k WET files —— 与 2025 版不是同一份，仅作对照）。
