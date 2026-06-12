# AGENTS.md — WeSpeaker 开发指南

## 项目概述

WeSpeaker 是一个**说话人嵌入（Speaker Embedding）学习**的研究与生产工具包，主要用于**说话人验证（Speaker Verification）**和**说话人日志（Speaker Diarization）**。由 WeNet 团队维护（GitHub: wenet-e2e/wespeaker），Apache 2.0 协议。

- **语言要求**: Python 3.8+
- **核心依赖**: PyTorch >= 1.12.0, torchaudio >= 0.12.0, silero-vad, kaldiio
- **关键版本锁定**: `hdbscan==0.8.37`, `umap-learn==0.5.6`（不可随意升级，聚类行为会变）

---

## 目录结构速览

```
wespeaker/               # 核心 Python 包
  cli/                   # CLI 入口 + Speaker 类 + Hub 模型下载
    speaker.py           # Speaker 类（extract_embedding, compute_similarity, diarize, register, recognize）
    hub.py               # 预训练模型下载（6 个模型：chinese, english, campplus, eres2net, vblinkp, vblinkf）
    utils.py             # 命令行参数解析
  models/                # 16+ 种模型架构（CAMPPlus, ERes2Net, ResNet, ECAPA_TDNN, etc）
    speaker_model.py     # 模型工厂路由器（根据 config.yaml 中 model 名字路由到具体类）
  bin/                   # 22 个训练/推理/导出脚本（train.py, extract.py, export_onnx.py, etc）
  dataset/               # 数据加载器
  diar/                  # 说话人日志（UMAP/HDBSCAN/AHC 聚类器）
  utils/                 # checkpoint 加载、PLDA、评分工具
  ssl/                   # 自监督预训练（DINO, MoCo, SimCLR）
  frontend/              # 前端特征提取（s3prl, whisper）
runtime/                 # C++ 推理引擎（onnxruntime / MNN / Triton Server）
examples/                # 各数据集的复现配方（voxceleb, cnceleb, sre, voxconverse, diarization）
tools/                   # shell/python 数据处理辅助脚本
docs/                    # Sphinx 文档
```

---

## 核心 API 使用方式

### 安装（开发模式）

```bash
pip install -e .
# 或
pip install -r requirements.txt
pip install .
```

### CLI 入口

```bash
wespeaker --task embedding --audio_file audio.wav
wespeaker --task similarity --audio_file a.wav --audio_file2 b.wav
wespeaker --task diarization --audio_file meeting.wav
```

CLI 由 `setup.py` 中的 `entry_points` 定义，指向 `wespeaker.cli.speaker:main`。

### Python API

```python
from wespeaker import load_model, load_model_local

# 加载预训练模型（自动下载到 ~/.wespeaker/）
model = load_model('chinese')   # 或 'english', 'campplus', 'eres2net'

# 加载本地模型（需要 config.yaml + avg_model.pt）
model = load_model_local('/path/to/model_dir')

# 设置设备
model.set_device('cuda:0')

# 提取 embedding
embedding = model.extract_embedding('audio.wav')  # 返回 torch.Tensor

# 计算两段音频的相似度
score = model.compute_similarity('a.wav', 'b.wav')  # 返回 float, 范围 [0, 1]

# 说话人日志
diar_result = model.diarize('meeting.wav')
```

### 相似度归一化

**关键**：`Speaker.cosine_similarity()` 中的相似度归一化公式为 `(cosine_score + 1.0) / 2`，将余弦相似度从 `[-1, 1]` 映射到 `[0, 1]`。这个归一化**仅应用于配对比较时的 API 显示输出**，而 `extract_embedding` 返回的原始 embedding 是**未归一化的向量**，在 pipeline 脚本中直接对各 embedding 向量用 `sklearn.metrics.pairwise.cosine_similarity` 计算，**不做**[0,1] 映射，结果范围是 `[-1, 1]`。

---

## 开发环境

### Conda 环境

本项目使用 `3dspeaker` conda 环境，开发前需先激活：

```bash
conda activate 3dspeaker
```

### 安装（开发模式）

```bash
pip install -e .
# 或
pip install -r requirements.txt
pip install .
```

## 开发命令

### 代码格式化和 Lint

```bash
# 安装 pre-commit hooks（开发前必须执行一次）
pre-commit install

# 手动运行所有 hooks
pre-commit run --all-files

# 单独运行 Python 格式化
yapf --recursive -i wespeaker/

# 单独运行 flake8
flake8 wespeaker/
```

**注意**：
- Python 格式化使用 **yapf**（不是 black/ruff），版本锁定 `v0.32.0`
- flake8 版本锁定 `3.8.2`，配置见 `.flake8`，**max-line-length = 80**
- C++ 格式化使用 **clang-format**（Google 风格，缩进=2，行宽=80）
- C++ lint 使用 **cpplint**，配置见 `CPPLINT.cfg`（root 为 `runtime/core`）

### 构建 C++ 运行时

```bash
# OnnxRuntime 引擎
cd runtime/onnxruntime && mkdir build && cd build && cmake .. && make

# MNN 引擎（需要先下载 MNN 库）
cd runtime/mnn && mkdir build && cd build && cmake .. && make
```

### 文档构建

```bash
cd docs
pip install -r requirements.txt
make html
```

---

## 无测试框架

此项目**没有** pytest / tox / unittest 测试框架配置。没有 `pytest.ini`、`tox.ini`、`conftest.py`。如果要写测试实践，需自行引入测试框架。`examples/extract_and_conclude_similarities/v2/` 下有少量 `test_*.py` 文件，但它们是该 pipeline 的独立测试脚本而非项目级测试。

---

## 模型加载约定

### 本地模型加载

`load_model_local(model_dir)` 要求 `model_dir` 下必须有两个文件：

| 文件 | 内容 |
|------|------|
| `config.yaml` | 模型配置（含 `model` 名字和 `model_args` 参数） |
| `avg_model.pt` | PyTorch checkpoint（由 `wespeaker.utils.checkpoint.load_checkpoint` 加载，使用 `strict=False`） |

`config.yaml` 中的 `model` 字段决定使用哪个模型类（见 `wespeaker/models/speaker_model.py:30-57` 的路由逻辑）。

### 预训练模型下载

`Hub.get_model(lang)` 从 ModelScope 下载模型 `<lang>` 到 `~/.wespeaker/<lang>/`，解压后需包含 `avg_model.pt` 和 `config.yaml`。

---

## Embedding 数据的存储格式

### 核心格式：Pickle (.pkl)

WeSpeaker 生态中的所有 embedding 都以 **Python pickle** 文件存储，结构如下：

```python
# 单条 utterance embedding:
{
    'embedding': np.ndarray,    # 1D flattened float32 array (通常 256/512 维)
    'dataset': str,             # 数据集名称
    'speaker_id': str,          # 说话人 ID
    'utterance_id': str,        # 音频文件名（不含扩展名）
    'original_path': str,       # 原始音频路径
}

# 说话人级 embedding (平均后的):
{
    'embedding': np.ndarray,    # 平均 embedding
    'dataset': str,
    'speaker_id': str,
    'num_utterances': int,
    'utterance_list': [str],    # 被平均的所有 utterance ID 列表
    'embedding_dim': int,
    'embedding_stats': {        # 统计信息
        'mean': float, 'std': float, 'min': float, 'max': float
    }
}
```

### 输出文件目录结构

所有输出的标准目录布局（以 pipeline 为例）：

```
{data_root}/
├── audio/                    # 原始音频
│   └── {dataset}/
│       └── {speaker_id}/
│           └── {utterance_id}.wav
├── embeddings_utterances/    # 步骤1输出
│   └── {dataset}/
│       └── {speaker_id}/
│           └── {utterance_id}.pkl
├── embeddings_speakers/      # 步骤2输出
│   └── {dataset}/
│       └── {speaker_id}.pkl
├── speaker_similarity_analysis/  # 步骤3输出
│   ├── speaker_similarities.json
│   ├── similarity_matrix.npy
│   ├── speaker_keys_mapping.json
│   ├── similarity_statistics.json
│   ├── analysis_summary.json
│   ├── upper_triangular_similarities.npy
│   ├── upper_triangular_statistics.json
│   ├── extreme_similarity_pairs.json
│   ├── threshold_statistics.json
│   └── speaker_top_similarities.json
└── utterance_similarities_per_speaker/  # 步骤4输出
    └── {dataset}/
        └── {speaker_id}_utterance_similarities.json
```

---

## examples/extract_and_conclude_similarities/v2_organized 详细分析

### 总体概述

这是一个**完整的说话人嵌入提取和相似度分析 pipeline**，将大语音数据集（多种来源，合并后的多说话人数据）进行结构化处理，输出三个层次的相似度分析：
1. **说话人-说话人**相似度（哪些说话人声纹接近）
2. **utterance-utterance**相似度（同一说话人内部各录音的一致性）

`v2_organized` 是 `v2` 的重构版本，将混在一处的脚本**拆解为逐步执行的 shell 脚本**，每步有各自的参数配置和错误检查。

### Pipeline 步骤串行依赖

```
step0 (音频重采样到16kHz)
  → step1 (8卡GPU提取utterance级embedding)
    → step2 (CPU多进程计算说话人平均embedding)
      → step3 (CPU多进程计算说话人-说话人相似度)
        → step4 (CPU多进程计算每个说话人内部utterance相似度)
          → step5 (清理JSON文件减少体积)
```

**每步必须按顺序执行**，后续步骤依赖前序输出。

### 各步骤详解

#### Step 0 — 音频重采样 (`step0_resample_audio_to_16k.sh`)

- **Python 脚本**: `local/resample_audio_to_16k.py`
- **功能**: 遍历数据根目录 `{dataset}/{speaker}/{utterance}.{wav,flac,mp3}`，将非 16000Hz 的音频用 librosa 重采样到 16kHz
- **关键参数**:
  - `--num_workers=16`: 并行 worker 数（使用 `ProcessPoolExecutor`）
  - `--skip_existing=true`: 跳过已是 16kHz 的文件（基于 `soundfile.info` 快速检查采样率，不加载完整音频）
  - `--backup=false`: 是否备份原始音频
  - `--res_type="fft"`: librosa 重采样方法（`fft` 质量最高，`kaiser_fast` 速度快）
- **额外依赖**: `librosa`, `soundfile`（不在 requirements.txt 中，需单独安装）
- **并行策略**: 每批处理 `num_workers * 10` 个文件，超时 300 秒/文件
- **multiprocessing 启动方式**: `mp.set_start_method('spawn', force=True)`

#### Step 1 — 多 GPU 提取 Embedding (`step1_run_wespeaker_embedding_extraction_optimized.sh`)

- **Python 脚本**: `local/extract_wespeaker_embeddings_optimized.py`
- **功能**: 使用 WeSpeaker 模型对所有音频文件逐条提取 embedding，保存为 `.pkl`
- **关键参数**:
  - `--gpus="0,1,2,3,4,5,6,7"`: 8 张 GPU 并行（通过 `CUDA_VISIBLE_DEVICES` 设置）
  - `--batch_size=24`: 每条 GPU 的 batch size
  - `--num_workers=6`: 每条 GPU 的 I/O 线程数
  - `--skip_existing=true`: 跳过已有 embedding 的文件（先扫描现有 `.pkl` 文件构建跳过集合，O(1) 查找）
  - `--random_shuffle=true` + `--random_seed=42`: 随机打乱文件顺序以实现更好的负载均衡
- **并行架构**:
  - 使用 `torch.multiprocessing.spawn` + `torch.distributed`(NCCL) 实现多 GPU 数据并行
  - Rank 0 先扫描所有音频文件，通过共享文件 `/tmp/audio_files_list.pkl` 分发给各 GPU
  - 每条 GPU 有独立的 I/O 线程池（异步保存 embedding）
  - 内存优化环境变量: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`, `OMP_NUM_THREADS=2`
- **模型加载**: 使用 `load_model_local(model_dir)` 加载本地模型，需要 `avg_model.pt` 和 `config.yaml`

#### Step 2 — 多进程计算说话人平均 Embedding (`step2_run_compute_speaker_embeddings_multiprocess.sh`)

- **Python 脚本**: `local/compute_speaker_embeddings_multiprocess.py`
- **功能**: 将 step1 输出的每条 utterance embedding 按说话人分组，计算每个说话人的**平均 embedding**
- **关键参数**:
  - `--num_processes=$(nproc)`: 使用全部 CPU 核
  - `--min_utterances=1`: 最少要有多少条录音才处理
  - `--exclude_filename_prefix=voiceprint`: 排除文件名以 "voiceprint" 开头的 utterance（用于过滤特定来源的录音）
- **输出**: 每个说话人保存为一个 `.pkl`，包含平均 embedding + 统计信息 + utterance 列表
- **multiprocessing 启动方式**: `mp.set_start_method('spawn', force=True)`
- **并行策略**: 将说话人按数量均分给各进程，每个进程处理一组说话人

#### Step 3 — 快速计算说话人-说话人相似度 (`step3_run_compute_speaker_similarities_fast.sh`)

- **Python 脚本**: `local/compute_speaker_similarities_fast.py`
- **功能**: 计算任意两个说话人之间的余弦相似度（全量 O(N²)），做详细统计分析
- **关键参数**:
  - `--num_workers=32`: CPU 并行数
  - `--batch_size=100`: 每批处理的说话人数
  - `--top_k=100`: 每个说话人的 top-k 最相似说话人
  - `--resume=false`: 是否从上次进度恢复（断点续跑支持）
  - `--skip_similarity=false`: 跳过相似度计算（只保存说话人 embedding）
  - `--max_speakers=`: 限制说话人数量（用于测试，留空=全部）
- **计算架构**:
  1. 扫描所有 `.pkl` 文件，按说话人分组
  2. 多进程并行计算每个说话人的平均 embedding
  3. 多进程并行计算余弦相似度矩阵（`sklearn.metrics.pairwise.cosine_similarity`）
  4. 提取上三角矩阵（排除对角线），分析最相似/不相似的说话人对
  5. 计算相似度分布直方图、阈值统计
- **输出文件**:
  - `speaker_similarities.json`: 全量 N×N 相似度字典（可能非常大，注意内存）
  - `similarity_matrix.npy`: NumPy 格式的相似度矩阵（快速加载）
  - `speaker_keys_mapping.json`: `{index: speaker_key}` 的映射
  - `similarity_statistics.json`: 基本统计（mean/std/min/max）
  - `upper_triangular_similarities.npy`: 上三角矩阵（不含对角线）
  - `upper_triangular_statistics.json`: 上三角相似度的统计（含中位数、四分位数）
  - `extreme_similarity_pairs.json`: Top 1000 最相似 + Bottom 1000 最不相似说话人对
  - `threshold_statistics.json`: 各阈值下的相似对数及占比
  - `analysis_summary.json`: 综合分析报告
  - `speaker_top_similarities.json`: 每个说话人的 top-k 最相似说话人

#### Step 4 — 计算每个说话人内部 Utterance 相似度 (`step4_run_compute_utterance_similarities_per_speaker.sh`)

- **Python 脚本**: `local/compute_utterance_similarities_per_speaker.py`
- **功能**: 对每个说话人的所有 utterance，计算 utterance 之间的余弦相似度矩阵，分析说话人内部录音的一致性
- **关键参数**:
  - `--num_workers=1`: 同时处理几个说话人（设为 1 以逐个处理，避免同时处理多个大 speaker 引发 OOM）
  - `--num_workers_internal=64`: **每个说话人内部**的并行 worker 数
  - `--similarity_threshold=0.7`: 只保存相似度 >= 0.7 的 utterance 对（减少输出体积）
  - `--max_utterances_limit=5000`: utterance 数超过此值的说话人直接跳过（相似度矩阵太大 O(n²)）
  - `--min_utterances=2`: utterance 少于 2 条的跳过（无法计算相似度）
  - `--skip_existing=true`: 跳过已有输出文件的说话人
- **分布式架构—双层并行**:
  - **外层**: `ProcessPoolExecutor(max_workers=num_workers)` — 每个 worker 独享一个说话人进程（默认 1 worker，串行处理，避免 OOM）
  - **内层**: 对 utterance 数 >100 的说话人，内部再使用 `ProcessPoolExecutor(max_workers=num_workers_internal)` 并行加载 embedding 和计算相似度矩阵
- **容错机制**:
  - 每个说话人最多重试 3 次（MemoryError/TimeoutError 时自动降级：减少 utterance 数量）
  - 单说话人超时 4 小时
  - 原子写入：先写到 `.tmp` 文件，成功后再 `replace` 到正式文件名
- **输出 JSON 结构**:
  ```json
  {
    "dataset": "...",
    "speaker_id": "...",
    "num_utterances": 512,
    "num_utterances_total": 512,
    "path_to_id": {"/path/to/audio1.wav": 0, "/path/to/audio2.wav": 1, ...},
    "similarity_pairs": [
      {"id_1": 0, "id_2": 1, "similarity": 0.9234},
      {"id_1": 3, "id_2": 7, "similarity": 0.8912},
      ...
    ],
    "statistics": {
      "mean": 0.721, "std": 0.053, "min": 0.312, "max": 0.998, "median": 0.734
    },
    "num_pairs_total": 130816,
    "num_pairs_saved": 87432,
    "similarity_threshold": 0.7
  }
  ```
- **格式特点**: 使用 `path_to_id` 映射（而非直接存储路径），`similarity_pairs` 中只用数字 ID 引用，大幅减少 JSON 体积
- **重构路径**: 可通过 `[path for path, _ in sorted(path_to_id.items(), key=lambda x: x[1])]` 还原完整的 utterance_paths 列表

#### Step 5 — 移除 JSON 中冗余字段 (`step5_remove_utterance_paths_from_json.sh`)

- **Python 脚本**: `local/remove_utterance_paths_from_json.py`
- **功能**: 从 step4 生成的 JSON 文件中移除 `utterance_paths` 字段（如果存在），进一步减小文件体积
- **参数**: `--num_workers=8`, `--dry_run=false`
- **操作**: 原子写入 `.tmp` + `replace`

### 环境变量要求

每个步骤的 shell 脚本中需要设置：

```bash
source path.sh  # 或手动设置：
export PYTHONPATH=../../../:$PYTHONPATH   # 将项目根目录加入 Python 路径
export PYTHONIOENCODING=UTF-8             # 避免 LC_ALL=C 时的 UnicodeDecodeError
```

### multiprocessing 启动方式

**所有 pipeline 脚本**必须使用 `mp.set_start_method('spawn', force=True)`，原因是：
- 在 `fork` 模式下，子进程会继承父进程的 CUDA 上下文和文件描述符，导致死锁或 OOM
- `spawn` 模式创建全新的 Python 解释器进程，完全隔离

### 依赖注意事项

Pipeline 脚本除 `requirements.txt` 外还额外依赖：
- `librosa`（音频重采样，step0 需要）
- `soundfile`（音频元数据读取和写入，step0 需要）
- `scikit-learn`（已在 requirements.txt 中，用于 cosine_similarity）

---

## 环境变量与路径约定

### Shell 脚本的 `path.sh`

所有 shell 脚本通过 `source path.sh` 设置：

```bash
export PATH=$PWD:$PATH
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH   # 相对路径，将项目根目录加入 Python 搜索路径
```

### 共享工具

`tools/parse_options.sh` 提供了 bash 脚本中的命名参数解析，用法：

```bash
. tools/parse_options.sh || exit 1
stage=1
stop_stage=1
# 用户可以 --stage 2 --stop_stage 5 等参数
```

---

## CI 工作流

### `.github/workflows/lint.yml`

三个 job：
1. **quick-checks**: 检查代码中无 tab 字符、无行尾空白
2. **flake8-py3**: flake8 3.8.2 + bugbear/comprehensions/executable/pyi 插件
3. **cpplint**: cpplint 1.6.1 递归检查整个仓库

### `.github/workflows/doc.yml`

- 推送 main 分支时自动构建 Sphinx 文档并部署到 gh-pages

### `.github/workflows/runtime.yml`

- PR 变更 `runtime/**` 路径时，在 macOS 和 Ubuntu 上构建 C++ 运行时

---

## 关键注意事项

### 1. 不要随意升级 hdbscan 和 umap-learn

`setup.py` 中锁定了特定版本：`hdbscan==0.8.37`, `umap-learn==0.5.6`。这两个包的新版本可能改变聚类行为，导致说话人日志结果不一致。

### 2. checkpoint 加载使用 strict=False

`load_checkpoint` 函数在加载模型权重时使用 `strict=False`（`wespeaker/utils/checkpoint.py:22`），这意味着**允许部分 key 缺失或不匹配**。修改模型结构后不会报错，但可能无声地失败。

### 3. 相似度归一化的不一致性

- `Speaker.compute_similarity()` 在 API 层面返回 `(cos+1)/2`，值域 `[0, 1]`
- Pipeline 脚本中直接使用 `sklearn` 的 `cosine_similarity`，不归一化，值域 `[-1, 1]`
- **这些值不能直接比较**

### 4. 预训练模型从 ModelScope 下载

Hub 从 `https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree` 获取模型列表。在中国大陆可能需要良好的网络或使用镜像。

### 5. 音频格式支持

- 支持 `.wav`, `.flac`, `.mp3`
- 内部处理统一转换为 16kHz 单声道
- VAD 功能依赖 `silero_vad`

### 6. C++ 与 Python 代码分离

`runtime/` 下的 C++ 代码与 Python 包完全独立。`CPPLINT.cfg` 指定 lint 根目录为 `runtime/core`。Python 代码变更不需要关注 C++ lint。

### 7. Embedding 维度约定

不同模型的 embedding 维度不同（如 ResNet34=256, ResNet221=512, ERes2Net=256 等）。存储和比较时需确保使用同一模型提取的 embedding。

---

## 常见问题

### Q: 如何运行单个数据集的训练？

参考 `examples/` 下的配方。例如 VoxCeleb：
```bash
cd examples/voxceleb/v2
bash run.sh --stage 1 --stop_stage 1  # 分阶段执行
```
配方使用 `tools/parse_options.sh` 实现 Kaldi 风格的 stage/stop_stage 控制。

### Q: 如何导出 ONNX 模型？

```bash
python wespeaker/bin/export_onnx.py --config config.yaml --checkpoint avg_model.pt --output model.onnx
```

### Q: WeSpeaker 包在 examples 中的副本有何作用？

`examples/extract_and_conclude_similarities/v2_organized/wespeaker/` 是一个完整嵌入的 `wespeaker` 包副本。这是为了确保脚本在不安装 wespeaker 包的情况下也能运行（通过 `PYTHONPATH` 找到它）。修改 pipeline 时要注意：修改的是副本还是主包。

### Q: pipeline 脚本中的 `SKIP_EXISTING` 参数如何工作？

- Step 0/1/2/4 支持 `--skip_existing` 参数
- 实现方式是先扫描输出目录，构建已有文件的集合（Set），然后跳过
- **不会做内容校验**（不检查 pkl 文件是否完整），只检查文件是否存在
- 如果需要重新处理某些文件，需要手动删除对应的输出文件或关闭 `--skip_existing`
