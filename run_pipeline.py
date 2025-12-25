import argparse
import sys
from pathlib import Path
import subprocess

# 引用我们刚才写的模块
# 注意路径问题，这里用 subprocess 调用更稳妥，防止内存泄漏
from pipeline.step1_process_video import process_video

def main():
    parser = argparse.ArgumentParser(description="SpotlessSplats Auto Pipeline")
    parser.add_argument("--video", required=True, help="Path to input .mp4") # 虽然跳过时不需要读视频，但为了兼容性保留
    parser.add_argument("--project-name", required=True, help="Name of the output folder")
    parser.add_argument("--data-factor", type=str, default="8", help="Downscale factor")
    parser.add_argument("--max-steps", type=str, default="30000", help="Max training iterations")
    
    # [新增] 跳过预处理开关
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip COLMAP and SD extraction if data exists")
    
    args = parser.parse_args()

    # 1. 设置路径
    root_dir = Path("my_data_cache")
    project_dir = root_dir / args.project_name
    
    # 预测数据集路径 (根据 step1 的逻辑)
    dataset_dir = project_dir / "undistorted"

    # 定位 spotless_trainer.py (逻辑不变)
    trainer_script = Path("examples") / "spotless_trainer.py"
    if not trainer_script.exists():
        trainer_script = Path("spotless_trainer.py")
        if not trainer_script.exists():
            print(f"❌ Error: Could not find spotless_trainer.py")
            sys.exit(1)

    print(f"🚀 Pipeline Start: {args.project_name}")

    # =========================================================
    # 逻辑分支：跳过 vs 不跳过
    # =========================================================
    if args.skip_preprocessing:
        print("\n⏭️  Skipping Preprocessing (COLMAP & Feature Extraction)...")
        
        # 严谨的检查：数据真的存在吗？
        if not dataset_dir.exists() or not (dataset_dir / "images").exists() or not (dataset_dir / "SD").exists():
            print(f"❌ Error: Cannot skip! Data not found at: {dataset_dir}")
            print(f"   Please run without --skip_preprocessing first.")
            sys.exit(1)
        else:
            print(f"✅ Found existing data at: {dataset_dir}")
            
    else:
        # --- 正常流程 ---
        
        # 2. 运行 Step 1: Video -> COLMAP
        dataset_dir = process_video(args.video, project_dir)
        
        # 3. 运行 Step 2: Feature Extraction
        print("\n=== Running Feature Extraction (Stable Diffusion) ===\n")
        step2_script = Path("pipeline_scripts") / "step2_extract_features.py"
        subprocess.check_call([sys.executable, str(step2_script), str(dataset_dir)])


    # =========================================================
    # Step 3: Training (总是运行)
    # =========================================================
    print("\n=== Running Spotless Training ===\n")
    
    output_model_dir = Path("results") / args.project_name
    
    # 如果跳过预处理，可能想在一个新的文件夹输出结果，避免覆盖？
    # 这里为了简单，我们还是用同一个结果目录，spotless_trainer 会处理覆盖问题
    
    cmd = [
        sys.executable, str(trainer_script),
        "--data_dir", str(dataset_dir),
        "--result_dir", str(output_model_dir),
        "--loss_type", "robust",
        "--semantics",
        "--no-cluster",
        "--train_keyword", "clutter",
        "--test_keyword", "extra",
        "--ubp",
        "--data-factor", str(args.data_factor),
        "--max-steps", str(args.max_steps)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print(f"\n🎉 Pipeline Complete! Results at: {output_model_dir}")

if __name__ == "__main__":
    main()
