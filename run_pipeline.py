import argparse
import sys
from pathlib import Path
import subprocess

# 引用我们刚才写的模块
# 注意路径问题，这里用 subprocess 调用更稳妥，防止内存泄漏
from pipeline.step1_process_video import process_video

def main():
    parser = argparse.ArgumentParser(description="SpotlessSplats Auto Pipeline")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--project-name", required=True, help="Name of the output folder")
    args = parser.parse_args()

    # 1. 设置路径
    root_dir = Path("my_data_cache")  # 所有中间数据放这里
    project_dir = root_dir / args.project_name
    
    # 2. 运行 Step 1: Video -> COLMAP (CPU/GPU)
    # 返回的是 undistorted 目录，这才是真正的 dataset 目录
    dataset_dir = process_video(args.video, project_dir)
    
    # 3. 运行 Step 2: Feature Extraction (GPU)
    # 我们用 subprocess 调用脚本，确保跑完后 Python 进程结束，彻底释放显存
    print("\n=== Running Feature Extraction ===\n")
    subprocess.check_call([sys.executable, "pipeline/step2_extract_features.py", str(dataset_dir)])

    # 4. 运行 Step 3: Training (GPU)
    print("\n=== Running Spotless Training ===\n")
    
    output_model_dir = Path("results") / args.project_name
    
    cmd = [
        sys.executable, "examples/spotless_trainer.py",
        "--data-dir", str(dataset_dir),
        "--result-dir", str(output_model_dir),
        "--loss-type", "robust",
        "--semantics",
        "--no-cluster",
        "--train-keyword", "clutter",
        "--test-keyword", "extra",
        "--ubp",
        "--data-factor", "8" # 只有 1 才能看清细节，但需要显存
    ]
    
    subprocess.check_call(cmd)
    
    print(f"\n🎉 Pipeline Complete! Results at: {output_model_dir}")

if __name__ == "__main__":
    main()