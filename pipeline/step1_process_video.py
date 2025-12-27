import os
import subprocess
import shutil
from pathlib import Path
import sys

def run_command(cmd):
    """打印并执行命令"""
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

def process_video(video_path, output_dir, fps=10):
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    # 1. 临时目录：存放 FFmpeg 刚切出来的纯数字图片
    raw_extract_dir = output_dir / "temp_raw_extract"
    
    # 2. COLMAP 输入目录：存放重命名后的图片 (clutter_xx.jpg)
    # 这才是 COLMAP 真正看到的目录
    colmap_input_images_dir = output_dir / "images_for_colmap"
    
    # 3. 最终输出目录
    undistorted_dir = output_dir / "undistorted"
    
    # 清理旧数据
    if output_dir.exists():
        print(f"Warning: {output_dir} exists. Cleaning up...")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True)
    raw_extract_dir.mkdir(parents=True)
    colmap_input_images_dir.mkdir(parents=True)

    # --- Step 1: FFmpeg 切帧 (临时存放) ---
    print(f"Step 1: Extracting frames from {video_path}...")
    run_command(f"ffmpeg -i '{video_path}' -qscale:v 1 -r {fps} '{raw_extract_dir}/%04d.jpg'")

    # --- Step 2: 立即重命名 (The Crucial Fix) ---
    print("Step 2: Renaming images BEFORE COLMAP...")
    
    raw_images = sorted(list(raw_extract_dir.glob("*.jpg")))
    if len(raw_images) == 0:
        print("❌ Error: No frames extracted from video!")
        sys.exit(1)

    for idx, old_path in enumerate(raw_images):
        # 按照 8:1 分割训练集和测试集
        # 注意：这里我们构造文件名，COLMAP 将会把这些名字写入数据库
        if idx % 8 == 0:
            new_name = f"extra_{idx:04d}.jpg"
        else:
            new_name = f"clutter_{idx:04d}.jpg"
        
        shutil.copy(old_path, colmap_input_images_dir / new_name)
    
    # 此时 colmap_input_images_dir 里全是 clutter_xxx.jpg 和 extra_xxx.jpg
    # 我们删掉临时的 raw 目录
    shutil.rmtree(raw_extract_dir)

    # --- Step 3: COLMAP 流程 (输入已经是重命名好的图片了) ---
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir()

    print("Step 3: COLMAP Feature Extraction...")
    # 注意：这里 input 指向 colmap_input_images_dir
    run_command(f"colmap feature_extractor --database_path {db_path} --image_path {colmap_input_images_dir} --ImageReader.camera_model SIMPLE_RADIAL")

    # print("Step 4: COLMAP Exhaustive Matcher...")
    # run_command(f"colmap exhaustive_matcher --database_path {db_path}")
    vocab_path = "/mnt/DISK/xjk/code/spotless/pipeline/vocab_tree_faiss_flickr100K_words256K.bin"
    print("Step 4: -> Using Sequential Matcher (Crucial for video data)")
    cmd = f"colmap sequential_matcher --database_path {db_path} --SequentialMatching.overlap 20 --SequentialMatching.loop_detection 1 --SequentialMatching.vocab_tree_path {vocab_path}"
    run_command(cmd)

    print("Step 5: COLMAP Mapper (SfM)...")
    run_command(f"colmap mapper --database_path {db_path} --image_path {colmap_input_images_dir} --output_path {sparse_dir}")

    # 检查 Mapper 结果
    if not (sparse_dir / "0" / "cameras.bin").exists():
        print("❌ Error: COLMAP mapping failed. Trying loose reconstruction options...")
        # 如果失败，这里可以加 fallback 逻辑，但暂时先报错退出
        sys.exit(1)

    # --- Step 6: 图像去畸变 (Undistortion) ---
    undistorted_dir.mkdir()
    print("Step 6: Image Undistortion...")
    
    # COLMAP 会读取数据库里的文件名 (clutter_xx.jpg)，并在 undistorted/images 里生成同样名字的文件
    run_command(f"colmap image_undistorter --image_path {colmap_input_images_dir} --input_path {sparse_dir}/0 --output_path {undistorted_dir} --output_type COLMAP")

    # 验证文件名一致性
    check_file = sorted(list((undistorted_dir / "images").glob("*.jpg")))[0]
    if "clutter" not in check_file.name and "extra" not in check_file.name:
         print(f"❌ Critical Error: Filenames mismatch! Found {check_file.name}")
         sys.exit(1)

    print(f"✅ Preprocessing done! Output at: {undistorted_dir}")
    print(f"   (Verified that filenames contain 'clutter'/'extra')")
    
    return undistorted_dir

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step1_process_video.py <video_path> <output_dir>")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])