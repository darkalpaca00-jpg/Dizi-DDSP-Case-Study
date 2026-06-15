import os
import glob
import subprocess

# 配置路径
MIDI_DIR = "./synthetic_midi_dataset"
OUTPUT_DIR = "./synthetic_audio_dataset"
SF2_PATH = "Dizi.sf2"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
midi_files = glob.glob(os.path.join(MIDI_DIR, "*.mid"))

print(f"🚀 开始渲染，共 {len(midi_files)} 个文件...")

for midi_file in midi_files:
    file_name = os.path.basename(midi_file)
    output_wav = os.path.join(OUTPUT_DIR, file_name.replace(".mid", ".wav"))
    
    # 使用 fluidsynth 渲染
    # -g 1.0 是增益，-r 48000 是采样率
    # 注意：fluidsynth 会自动加载 SF2 中的 Program 0 作为默认音色
    # 严格的参数顺序：选项参数在前，文件路径在后
    cmd = [
        "fluidsynth", 
        "-ni",           # 无交互界面
        "-q",            # 静默模式
        "-r", "48000",   # 采样率
        "-F", output_wav,# 输出文件参数必须放在文件路径之前
        SF2_PATH,        # SoundFont 文件
        midi_file        # MIDI 文件
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 完成: {file_name}")
    except subprocess.CalledProcessError:
        print(f"❌ 失败: {file_name}")

print("🎉 全部渲染任务完成！")