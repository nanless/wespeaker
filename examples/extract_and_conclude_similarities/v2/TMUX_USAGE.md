# WeSpeaker Embedding Extraction - Tmux 后台运行版本

## 🎯 功能特点

- **后台运行**: 在tmux会话中运行，不受SSH断开影响
- **完整日志**: 所有输出保存到日志文件
- **进程管理**: 支持启动、停止、状态查看
- **会话管理**: 可以随时attach查看实时进度
- **GPU监控**: 显示GPU使用状态

## 🚀 快速使用

### 1. 启动后台任务
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action start
```

### 2. 查看运行状态
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action status
```

### 3. 查看日志
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action logs
```

### 4. 连接到会话查看实时进度
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action attach
```

### 5. 停止任务
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action stop
```

## 📋 详细命令说明

### 启动任务 (start)
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action start
```
- 创建名为 `wespeaker_embedding_extraction` 的tmux会话
- 在后台运行embedding提取
- 生成时间戳日志文件
- 显示管理命令提示

### 查看状态 (status)
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action status
```
显示信息：
- ✅ 会话是否正在运行
- 📊 进程ID和状态
- 🖥️ GPU使用情况
- 📝 最近10行日志

### 查看完整日志 (logs)
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action logs
```
- 使用 `less` 查看完整日志
- 支持搜索和翻页
- 按 `q` 退出查看

### 连接会话 (attach)
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action attach
```
- 连接到正在运行的tmux会话
- 查看实时输出
- **按 `Ctrl+B` 然后 `D` 来断开会话（不停止任务）**

### 停止任务 (stop)
```bash
./run_wespeaker_embedding_extraction_tmux.sh --action stop
```
- 终止tmux会话
- 停止所有相关进程
- 清理PID文件

## 📁 文件结构

```
./logs/                                          # 日志目录
├── embedding_extraction_20231123_143022.log    # 时间戳日志文件
├── extraction.pid                              # 进程ID文件
└── ...
```

## 💡 使用场景

### 1. 长时间任务
```bash
# 启动后台任务
./run_wespeaker_embedding_extraction_tmux.sh --action start

# 断开SSH连接（任务继续运行）
exit

# 重新连接后查看状态
ssh your_server
cd /path/to/wespeaker/examples/extract_and_conclude_similarities/v2
./run_wespeaker_embedding_extraction_tmux.sh --action status
```

### 2. 监控进度
```bash
# 查看最新状态
./run_wespeaker_embedding_extraction_tmux.sh --action status

# 连接查看实时进度
./run_wespeaker_embedding_extraction_tmux.sh --action attach

# 断开但不停止任务（按 Ctrl+B 然后 D）
```

### 3. 错误排查
```bash
# 查看完整日志
./run_wespeaker_embedding_extraction_tmux.sh --action logs

# 查看最近日志
./run_wespeaker_embedding_extraction_tmux.sh --action status
```

## 🖥️ 日志格式

```
=== WeSpeaker Embedding Extraction Started at Mon Nov 23 14:30:22 CST 2023 ===
Session: wespeaker_embedding_extraction
PID: 12345
Working directory: /root/code/github_repos/wespeaker/examples/extract_and_conclude_similarities/v2

📊 Counting audio files...
Total audio files to process: 50000

🔥 Starting embedding extraction...
GPU 0: Processing 12500 files (from 0 to 12499)
GPU 1: Processing 12500 files (from 12500 to 24999)
GPU 2: Processing 12500 files (from 25000 to 37499)
GPU 3: Processing 12500 files (from 37500 to 49999)

... 处理进度 ...

🎉 Embedding extraction completed successfully at Mon Nov 23 16:45:33 CST 2023!
✅ Results saved in: /path/to/output
```

## ⚠️ 注意事项

### 1. 会话管理
- 同时只能运行一个提取会话
- 如果已有会话在运行，需要先停止或使用不同的SESSION_NAME

### 2. 日志文件
- 每次启动都会创建新的日志文件（时间戳命名）
- 日志文件会保留所有运行记录
- 定期清理旧日志文件

### 3. 进程管理
- PID文件用于跟踪主进程
- 会话结束后会自动清理PID文件
- 异常终止时可能需要手动清理

### 4. GPU资源
- 确保没有其他程序占用GPU
- 监控GPU内存使用情况
- 必要时调整GPU配置

## 🛠️ 故障排除

### 1. 会话无法启动
```bash
# 检查tmux是否安装
tmux -V

# 检查是否有同名会话
tmux list-sessions

# 强制清理旧会话
tmux kill-session -t wespeaker_embedding_extraction
```

### 2. 日志查看问题
```bash
# 手动查看最新日志
ls -la logs/
tail -f logs/embedding_extraction_*.log

# 查看所有tmux会话
tmux list-sessions
```

### 3. 进程异常
```bash
# 检查GPU进程
nvidia-smi

# 检查Python进程
ps aux | grep extract_wespeaker_embeddings

# 清理异常进程
pkill -f extract_wespeaker_embeddings
```

## 📈 性能监控

### 实时监控命令
```bash
# 在另一个终端运行
watch -n 5 './run_wespeaker_embedding_extraction_tmux.sh --action status'

# GPU监控
watch -n 2 nvidia-smi

# 磁盘I/O监控
iostat -x 2
```

## 🎉 完成后操作

任务完成后，会话会保持活跃60秒供您查看结果，然后自动结束。

```bash
# 查看最终结果
./run_wespeaker_embedding_extraction_tmux.sh --action logs

# 检查输出目录
ls -la /path/to/output/directory
```

---

**推荐工作流程:**
1. `--action start` 启动任务
2. `--action status` 定期检查状态  
3. `--action attach` 需要时查看实时进度
4. `--action logs` 任务完成后查看完整日志 