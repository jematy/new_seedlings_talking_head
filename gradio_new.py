import gradio as gr
import subprocess
import os
import shutil
import uuid
import re
import tempfile
from subprocess import Popen, PIPE

MODEL_ROOT = "/data/new_seedlings/model"
DATA_ROOT = "data"

# === 工具函数 ===
def list_model_names():
    return [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]

def list_processed_ids():
    return [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]

def get_reference_image(model_name):
    validation_path = os.path.join(MODEL_ROOT, model_name, "validation")
    if not os.path.exists(validation_path):
        return None

    pattern = r"ngp_ep(\d{4})_(\d{4})_rgb\.png"
    candidates = []

    for fname in os.listdir(validation_path):
        match = re.match(pattern, fname)
        if match:
            ep = int(match.group(1))
            frame = int(match.group(2))
            candidates.append((ep, frame, fname))

    if not candidates:
        return None

    max_ep = max(c[0] for c in candidates)
    filtered = [c for c in candidates if c[0] == max_ep]
    selected = max(filtered, key=lambda x: x[1])

    return os.path.join(validation_path, selected[2])

def update_reference_image(model_name):
    return get_reference_image(model_name)

# ========== 语音生成 ==========
def process_audio_with_log(audio_file, model_name):
    audio_path = f"/tmp/{uuid.uuid4()}.wav"
    shutil.copy(audio_file, audio_path)

    workspace = os.path.join(MODEL_ROOT, model_name)
    output_dir = os.path.join(workspace, "results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    command = [
        "python", "main.py", "data/jkx",
        "--workspace", workspace,
        "-O", "--torso", "--test", "--test_train",
        "--asr_model", "ave",
        "--aud", audio_path,
        "--au45"
    ]

    log = "🚀 开始生成视频...\n"

    try:
        process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
        for line in process.stdout:
            log += line
            yield log, None
        process.wait()
        if process.returncode != 0:
            raise Exception("命令运行失败")
    except Exception as e:
        log += f"\n❌ 出错：{e}"
        yield log, None
        return

    if not os.path.exists(output_dir):
        log += "\n❌ 输出目录不存在"
        yield log, None
        return

    for fname in os.listdir(output_dir):
        if fname.endswith("_audio.mp4"):
            video_path = os.path.join(output_dir, fname)
            log += f"\n✅ 生成成功：{fname}"
            yield log, video_path
            return

    log += "\n❌ 未找到生成的视频文件"
    yield log, None


# ========== 步骤 1：处理数据 ==========
def handle_data_processing_with_log(video_file, user_id):
    if not user_id:
        yield "❌ 请输入 ID"
        return

    user_dir = os.path.join(DATA_ROOT, user_id)
    os.makedirs(user_dir, exist_ok=True)
    video_target = os.path.join(user_dir, f"{user_id}.mp4")
    shutil.copy(video_file, video_target)

    command = ["python", "data_utils/process.py", video_target, "--asr", "ave"]

    log = f"📦 开始处理视频：{user_id}.mp4\n"

    try:
        process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
        for line in process.stdout:
            log += line
            yield log
        process.wait()
        if process.returncode != 0:
            raise Exception("处理失败")
    except Exception as e:
        log += f"\n❌ 错误：{e}"
        yield log
        return

    log += "\n✅ 数据处理完成！"
    yield log

# ========== 步骤 2：训练模型 ==========
def train_model(user_id, use_portrait):
    if not user_id:
        yield "❌ 请选择一个已处理的 ID", gr.update()

    workspace = f"model/trial_{user_id}"
    cmd = [
        "python", "main.py", f"data/{user_id}",
        "--workspace", workspace,
        "-O", "--test", "--asr_model", "ave"
    ]
    if use_portrait:
        cmd.append("--portrait")

    log = f"🏋️ 开始训练模型：{user_id}\n"

    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        for line in p.stdout:
            log += line
            yield log, gr.update()
        p.wait()
        if p.returncode != 0:
            raise Exception("训练失败")
    except Exception as e:
        log += f"\n❌ 错误：{e}"
        yield log, gr.update()
        return

    log += "\n✅ 训练完成 ✅"
    yield log, gr.update(choices=list_model_names(), value=f"trial_{user_id}")

# ========== 音频转 WAV ==========
def convert_to_wav(input_file):
    if not input_file:
        return None, "❌ 请上传一个音/视频文件"

    tmp_output = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        tmp_output
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_output, "✅ 转换成功"
    except Exception as e:
        return None, f"❌ 转换失败：{e}"

# ========== 页面加载时刷新 ==========
def refresh_model_dropdown():
    return gr.update(choices=list_model_names(), value=list_model_names()[0] if list_model_names() else None)

def refresh_id_dropdown():
    return gr.update(choices=list_processed_ids(), value=list_processed_ids()[0] if list_processed_ids() else None)

# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("# 🌱 新苗计划平台 - 分步训练与生成")

    # ========== 语音生成 ==========
    gr.Markdown("## 🎧 上传语音生成视频")

    model_dropdown = gr.Dropdown(label="选择人物模型", choices=[], value=None)
    reference_image = gr.Image(label="自动选取的参考图", height=240)

    model_dropdown.change(fn=update_reference_image, inputs=model_dropdown, outputs=reference_image)

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="上传音频（.wav）")

    with gr.Row():
        run_btn = gr.Button("🎬 开始生成")
        clear_btn = gr.Button("🧹 清空日志")

    status_box = gr.Textbox(label="生成日志输出", lines=12, interactive=False)
    video_output = gr.Video(label="生成的视频", height=360)

    run_btn.click(
        fn=process_audio_with_log,
        inputs=[audio_input, model_dropdown],
        outputs=[status_box, video_output]
    )

    clear_btn.click(
        fn=lambda: ("", None),
        outputs=[status_box, video_output]
    )


    gr.Markdown("---")

    # ========== 步骤一：处理数据 ==========
    gr.Markdown("## 🧩 第一步：上传视频并处理数据")

    with gr.Row():
        raw_video = gr.Video(label="上传原始视频 (.mp4)")
        input_id = gr.Textbox(label="人物 ID（例如 jkx）", placeholder="只能是英文字母或数字")

    process_btn = gr.Button("📦 处理视频生成数据")
    process_log = gr.Textbox(label="处理日志", lines=8, interactive=False)
    process_btn.click(fn=handle_data_processing_with_log, inputs=[raw_video, input_id], outputs=[process_log])

    gr.Markdown("---")

    # ========== 步骤二：训练模型 ==========
    gr.Markdown("## 🏋️ 第二步：训练模型")

    with gr.Row():
        id_dropdown = gr.Dropdown(label="选择已处理数据 ID", choices=[])
        portrait_cb = gr.Checkbox(label="使用人脸模式", value=False)

    train_btn = gr.Button("🚀 开始训练模型")
    train_log = gr.Textbox(label="训练日志输出", lines=12, interactive=False)

    train_btn.click(fn=train_model, inputs=[id_dropdown, portrait_cb], outputs=[train_log, model_dropdown])

    gr.Markdown("---")

    # ========== 附加功能：转为 WAV ==========
    gr.Markdown("## 🎼 转换为 WAV 格式")

    with gr.Row():
        input_audio_file = gr.File(label="上传音频/视频文件（任意格式）")
        convert_btn = gr.Button("🎵 转换为 .wav")

    with gr.Row():
        wav_output = gr.File(label="转换后的 WAV 文件")
        convert_status = gr.Textbox(label="转换状态", interactive=False)

    convert_btn.click(
        fn=convert_to_wav,
        inputs=input_audio_file,
        outputs=[wav_output, convert_status]
    )

    # 页面初始化
    demo.load(fn=refresh_model_dropdown, outputs=model_dropdown)
    demo.load(fn=refresh_id_dropdown, outputs=id_dropdown)

demo.launch()
