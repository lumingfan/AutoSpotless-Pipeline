from collections import deque
import gradio as gr
import subprocess
import sys
import os
import time
import shutil
from pathlib import Path
import signal

# ---------------------------------------------------------
# 配置区
# ---------------------------------------------------------
PYTHON_EXEC = sys.executable  # 获取当前环境的 python 路径
SCRIPT_PATH = "run_pipeline.py"
ROOT_OUTPUT_DIR = Path("results")

# ---------------------------------------------------------
# 核心逻辑：异步运行管线并流式输出日志
# ---------------------------------------------------------
def run_spotless_pipeline(video_file, project_name, data_factor, max_steps, skip_preprocessing, progress=gr.Progress()):
    """
    Generator 函数：使用 Rolling Buffer 优化日志输出，防止浏览器卡死
    """
    # 1. 输入校验
    if not video_file and not skip_preprocessing:
        yield "❌ Error: Please upload a video first.", None, None
        return
    
    if not project_name:
        project_name = f"demo_{int(time.time())}"
    
    # 清理项目名称
    project_name = "".join([c if c.isalnum() else "_" for c in project_name])
    
    # 定义全量日志保存路径 (Server Side Logging)
    os.makedirs("logs", exist_ok=True)
    log_file_path = Path("logs") / f"{project_name}.log"
    
    # 2. 构造命令
    cmd = [
        PYTHON_EXEC, "-u", SCRIPT_PATH,
        "--video", video_file if video_file else "dummy.mp4", # 如果跳过，传个假路径防报错
        "--project-name", project_name,
        "--data-factor", str(data_factor),
        "--max-steps", str(max_steps) # 新增
    ]
     # [修改 2] 如果勾选，加入参数
    if skip_preprocessing:
        cmd.append("--skip-preprocessing")
    
    cmd_str = " ".join(cmd)
    initial_log = f"🚀 Launching Pipeline...\nCommand: {cmd_str}\n"
    initial_log += f"📝 Full logs will be saved to: {log_file_path}\n"
    initial_log += "-" * 50 + "\n"
    
    # 3. 初始化 Rolling Buffer (关键修改！)
    # maxlen=1000 意味着内存里只保留最后 1000 行，旧的会自动挤出去
    # 这能保证浏览器永远只渲染少量文本，绝对不会卡
    log_queue = deque([initial_log], maxlen=1000)
    
    yield "".join(log_queue), None, None
    
    # 4. 启动子进程
    with open(log_file_path, "w", encoding="utf-8") as f_log:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1 
            )
            
            # 5. 实时读取日志
            # 优化策略：每读取一行，写入文件，更新队列
            # 但为了减少 yield 频率（减少网络闪烁），我们可以每隔几行 yield 一次
            line_counter = 0
            
            for line in proc.stdout:
                # A. 写入全量日志文件 (硬盘)
                f_log.write(line)
                f_log.flush() # 确保实时写入
                
                # B. 写入滚动队列 (内存)
                log_queue.append(line)
                
                # C. 前端刷新控制 (每接收 5 行或者遇到关键信息才刷新一次前端)
                # 这样可以显著降低浏览器负载
                line_counter += 1
                if line_counter % 10 == 0 or "Step" in line or "Error" in line:
                    yield "".join(log_queue), None, None
                
            # 等待进程结束
            proc.wait()
            
            if proc.returncode != 0:
                log_queue.append(f"\n❌ Pipeline failed with return code {proc.returncode}")
                yield "".join(log_queue), None, None
                return

        except Exception as e:
            err_msg = f"\n❌ System Error: {str(e)}"
            log_queue.append(err_msg)
            if 'f_log' in locals(): f_log.write(err_msg)
            yield "".join(log_queue), None, None
            return

    # 6. 寻找结果文件 (保持不变)
    output_dir = ROOT_OUTPUT_DIR / project_name
    
    video_candidates = list((output_dir / "videos").glob("*.mp4")) + \
                       list((output_dir / "videos").glob("*.gif"))
    
    result_video = None
    if video_candidates:
        result_video = str(sorted(video_candidates, key=os.path.getmtime)[-1])
        log_queue.append(f"\n✅ Found video: {result_video}")
    else:
        log_queue.append(f"\n⚠️ Warning: No video found in {output_dir}/videos")
        
    
    log_queue.append("\n\n🎉 ALL DONE! You can download the results below.")
    yield "".join(log_queue), result_video

# ---------------------------------------------------------
# 前端布局 (Gradio Blocks)
# ---------------------------------------------------------
with gr.Blocks(title="AutoSpotless Pipeline", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🧹 AutoSpotless Pipeline
        ### Remove moving people/objects from videos and reconstruct clean 3D scenes.
        Based on *SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting*.
        """
    )
    
    with gr.Row():
        # --- 左侧：输入区 ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input")
            input_video = gr.Video(label="Upload Video (mp4)", sources=["upload"])
            
            with gr.Accordion("Advanced Settings", open=True):
                project_name_input = gr.Textbox(
                    label="Project Name", 
                    value="my_scene_01", 
                    placeholder="e.g. desk_scan"
                )
                # [修改 3] 新增 Checkbox
                skip_checkbox = gr.Checkbox(
                    label="⚡ Skip Preprocessing (Use existing COLMAP/SD data)", 
                    value=False,
                    info="Check this if you already ran this project and just want to retrain with different settings."
                )
                data_factor_slider = gr.Slider(
                    minimum=1, maximum=8, step=1, value=8, 
                    label="Downscale Factor (1=Best Quality, 8=Fastest)"
                )
                max_steps_slider = gr.Slider(
                    minimum=1000, maximum=30000, step=1000, value=10000,
                    label="Max Steps (Training Iterations)"
                )
            
            run_btn = gr.Button("🚀 Start Training Pipeline", variant="primary", size="lg")

        # --- 右侧：输出区 ---
        with gr.Column(scale=1):
            gr.Markdown("### 2. Status & Logs")
            # 这里的 Log 框设为自动滚动
            log_output = gr.Textbox(
                label="Process Logs", 
                lines=15, 
                max_lines=20, 
                autoscroll=True,
                value="Ready to start..."
            )
            
            gr.Markdown("### 3. Results")
            result_video_output = gr.Video(label="Rendered Trajectory")
            # result_ply_output = gr.File(label="Download 3D Point Cloud (.ply)")

    # ---------------------------------------------------------
    # 事件绑定
    # ---------------------------------------------------------
    run_btn.click(
        fn=run_spotless_pipeline,
        inputs=[input_video, project_name_input, data_factor_slider, max_steps_slider, skip_checkbox],
        outputs=[log_output, result_video_output]
    )

# 启动服务
if __name__ == "__main__":
    # share=True 会生成一个临时的公网链接 (类似 *.gradio.live)
    # server_name="0.0.0.0" 允许局域网访问
    demo.queue().launch(server_name="0.0.0.0", share=True)