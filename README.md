# AutoSpotless: 一键式 SpotLessSplats 自动化管线

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://spotlesssplats.github.io/) [![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2406.20055) [![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)

**AutoSpotless** 是基于 [SpotLessSplats](https://github.com/lilygoli/SpotLessSplats) 的自动化封装版本。它旨在解决原始项目中繁琐的数据准备流程，提供从**单目视频**到**去干扰 3D 高斯重建**的一键式体验。

该项目包含一个命令行工具和一个基于 Web 的可视化界面，能够自动完成以下任务：

1.  **视频抽帧与预处理**：自动调用 FFmpeg 抽帧，并针对视频数据优化 COLMAP 重建（使用序列匹配）。
2.  **语义特征提取**：利用 Stable Diffusion 提取多尺度语义特征，无需手动运行 Notebook。
3.  **鲁棒性训练**：自动启动 SpotLessSplats 训练，去除场景中的动态干扰物（如行人、车辆）。

---

## ✨ 核心功能

*   **📺 Gradio Web UI**：拖拽上传视频，实时查看训练日志，网页端直接预览渲染结果。
*   **🔄 全自动管线**：`Video` -> `COLMAP SfM` -> `Stable Diffusion Features` -> `SpotLessSplats Training` 一气呵成。
*   **⚡ 视频优化**：针对视频输入，内置了 COLMAP `sequential_matcher` 策略，大幅提高 SfM 建图成功率。
*   **🧠 智能跳过**：支持 `--skip-preprocessing` 参数，方便在调整训练参数时跳过耗时的 COLMAP 和特征提取步骤。
*   **🛡️ 鲁棒性配置**：默认启用 UBP (Utilization-based Pruning) 和语义 Mask，自动处理遮挡和瞬态物体。

---

## 🛠️ 安装指南

本项目基于 `gsplat` 和 `SpotLessSplats`。请确保您的系统已安装 **CUDA**、**FFmpeg** 和 **COLMAP**。

### 1. 克隆代码

```bash
git clone --recursive https://github.com/lumingfan/AutoSpotless-Pipeline
cd AutoSpotless
```

### 2. 环境配置

建议使用 Conda 创建环境：

```shell
# 1. 创建名为 spotless 的环境
conda create -n spotless python=3.10 -y

# 2. 激活环境
conda activate spotless


```

### 3. 确认 CUDA 环境

1. 检查系统 CUDA 版本：

   ```bash
   nvcc --version
   
   # 如果提示没有nvcc, 不要按照服务器提示的命令安装, 再次确认是否已经安装
   # 如果ls -d命令有输出，则已经安装只是没有加入环境变量
   ls -d /usr/local/cuda*
   export PATH=/usr/local/cuda-12.2/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
   # 再次检查
   nvcc --version
   # 如果提示某个文件夹不存在就创建对应文件夹
   ```

   *   如果显示 11.x (比如 11.8)，安装 PyTorch CUDA 11.8 版本。
   *   如果显示 12.x (比如 12.1)，安装 PyTorch CUDA 12.2 版本。

2. 安装 PyTorch (假设服务器是 CUDA 12.x，如果是 11.x 请自行修改 URL):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
   ```

### 4. 克隆并安装代码

```shell
# 3. 安装 Python 依赖 (在 examples 目录下)
pip install -r examples/requirements.txt

# 4. 编译并安装 SpotLessSplats (这步最关键，会编译 CUDA kernel)
# 注意：这步可能需要几分钟，期间 CPU 会满载编译
export CUDA_HOME=/usr/local/cuda-12.2
pip install . --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```



### 5. 系统依赖 (必须)

*   **COLMAP**: 必须安装并添加到系统环境变量 `PATH` 中。
    *   Ubuntu: `sudo apt-get install colmap`
*   **FFmpeg**: 用于视频处理。
    *   Ubuntu: `sudo apt-get install ffmpeg`

---

## 🚀 使用方法

### 方式一：Web UI (推荐)

最简单的使用方式，适合直观操作。

```bash
python app.py
```

启动后，访问终端显示的链接（通常为 `http://0.0.0.0:7860`）。

1.  **Upload Video**: 上传你的 MP4 视频。
2.  **Settings**: 设置项目名称（如 `my_desk`），选择降采样倍率（默认 8）以及训练轮次（默认10000）。
3.  **Start**: 点击运行，右侧 Log 窗口会实时滚动显示 COLMAP 和训练进度。
4.  **Result**: 训练完成后，结果视频会自动展示在界面上。

### 方式二：命令行 (CLI)

适合批量处理或服务器后台运行。

```bash
# 运行完整流程：视频 -> 3D模型
python spotless/run_pipeline.py --video /path/to/video.mp4 --project-name my_scene_01

# 如果已经跑过 COLMAP，只想调整训练参数（跳过预处理）
python spotless/run_pipeline.py --video dummy.mp4 --project-name my_scene_01 --skip-preprocessing --max-steps 15000
```

**参数说明：**

*   `--video`: 输入视频路径。
*   `--project-name`: 项目名称，结果将保存在 `results/<project_name>`。
*   `--data-factor`: 图像降采样倍率（默认为 8，数值越大训练越快但细节越少）。
*   `--max-steps`: 训练迭代步数（默认 30000）。
*   `--skip-preprocessing`: 跳过 COLMAP 和特征提取阶段（需确保数据已存在）。

---

## 📂 目录结构与输出

运行pipeline后，数据将按以下结构组织：

```text
├── my_data_cache/               # 预处理中间数据
│   └── <project_name>/
│       ├── images_for_colmap/   # 重命名后的序列帧 (clutter/extra)
│       ├── sparse/              # COLMAP 稀疏重建结果
│       ├── undistorted/         # 去畸变后的数据集 (训练输入)
│       │   ├── images/          # RGB 图像
│       │   └── SD/              # Stable Diffusion 特征 (.npy)
│       └── database.db          # COLMAP 数据库
│
├── results/                     # 训练结果
│   └── <project_name>/
│       ├── point_cloud/         # 最终 PLY 点云
│       ├── cameras.json         # 相机参数
│       └── videos/              # 渲染的对比视频 (mp4/gif)
│
├── app.py                   # Gradio 入口
├── run_pipeline.py          # CLI 入口
└── pipeline/
  ├── step1_process_video.py      # COLMAP 自动化脚本
  └── step2_extract_features.py   # 语义特征提取脚本
```

---

## ⚠️ 注意事项

1.  **COLMAP 词汇树 (Vocab Tree)**:
    *   在 `pipeline/step1_process_video.py` 中，针对视频数据使用了 `sequential_matcher`。
    *   请手动下载 [vocab_tree_faiss_flickr100K_words256K.bin](https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin)，并在 `step1_process_video.py` 中修改 `vocab_path` 路径。
2.  **Stable Diffusion 模型**:
    *   首次运行 `step2` 会自动从 HuggingFace 下载 `sd2-community/stable-diffusion-2-1` 模型。请确保网络通畅，或预先下载并指定本地路径。
3.  **显存要求**:
    *   特征提取阶段需要加载 Stable Diffusion UNet，建议至少拥有 8GB 显存。

---

## 🔗 引用

本项目是 SpotLessSplats 的第三方自动化实现，原始论文：

```bibtex
@article{sabourgoli2024spotlesssplats,
    title={{SpotLessSplats}: Ignoring Distractors in 3D Gaussian Splatting},
    author={Sabour, Sara and Goli, Lily and Kopanas, George and Matthews, Mark and Lagun, Dmitry and Guibas, Leonidas and Jacobson, Alec and Fleet, David J. and Tagliasacchi, Andrea},
    journal={arXiv:2406.20055},
    year={2024}
}
```







