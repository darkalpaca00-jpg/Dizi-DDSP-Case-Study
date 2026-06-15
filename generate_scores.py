import os
import random
import numpy as np
import pretty_midi

# 创建输出目录
output_dir = "./synthetic_midi_dataset"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 核心规则函数：生成单首竹笛 MIDI 乐谱
# ==========================================
def generate_single_dizi_score(tempo, base_key, duration_seconds=30):
    """
    根据论文规则生成一首竹笛 MIDI 乐谱
    tempo: 速度 (BPM)
    base_key: 调式根音 (例如 7 代表 G)
    duration_seconds: 每首曲子的目标时长（秒）
    """
    # 1. 初始化 MIDI 对象和设置速度
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # 2. 创建竹笛乐器轨道 (General MIDI 编号 73 是 Flute 笛子)
    dizi_program = pretty_midi.instrument_name_to_program('Flute')
    dizi_track = pretty_midi.Instrument(program=dizi_program)
    
    # 3. 物理限制：定义曲笛在 G 大调下的合法音符集合 (MIDI 编号 62 到 86)
    # G 大调包含的音名相对于C的半音偏移: [7(G), 9(A), 11(B), 0(C), 2(D), 4(E), 6(F#)]
    g_major_intervals = [7, 9, 11, 0, 2, 4, 6]
    allowed_pitches = [p for p in range(62, 87) if p % 12 in g_major_intervals]
    
    # 4. 初始状态
    current_time = 0.0
    current_pitch = random.choice(allowed_pitches)  # 随机选一个起点音符
    
    # 节奏权重表：[十六分, 八分, 四分, 二分] 音符在 120BPM 下的秒数
    # 论文提到：音符时值从常见时值的加权分布中采样
    note_types = [0.25, 0.5, 1.0, 2.0]  # 以拍为单位
    rhythm_weights = [0.1, 0.4, 0.4, 0.1]  # 八分和四分音符概率最高(各40%)
    
    # 转换拍数到秒数的系数
    beats_to_seconds = 60.0 / tempo
    
    # 5. 循环生成音符，直到达到目标时长
    while current_time < duration_seconds:
        # 【规则二：节奏加权采样】
        chosen_beat = random.choices(note_types, weights=rhythm_weights)[0]
        note_duration = chosen_beat * beats_to_seconds
        
        # 【规则三：音高转换约束在一个八度内】
        # 找出所有在合法音域内、且与当前音高距离 <= 12 的音符
        valid_next_pitches = [p for p in allowed_pitches if abs(p - current_pitch) <= 12]
        
        # 为了让旋律更有级进的线条感，我们可以稍微给贴近的音更高的权重，这里采用均匀采样
        current_pitch = random.choice(valid_next_pitches)
        
        # 【规则四：力度高斯分布】
        # 均值80，标准差5，表现出克制、平稳、自然的微小情绪波动
        raw_velocity = np.random.normal(loc=80, scale=5)
        # 限制在 MIDI 的标准边界 1~127 之间
        final_velocity = int(np.clip(raw_velocity, 1, 127))
        
        # 6. 创建音符对象并添加进轨道
        note = pretty_midi.Note(
            velocity=final_velocity,
            pitch=current_pitch,
            start=current_time,
            end=current_time + note_duration
        )
        dizi_track.notes.append(note)
        
        # 时间轴向前推进
        current_time += note_duration
        
    # 将竹笛轨道添加到 MIDI 文件中
    pm.instruments.append(dizi_track)
    return pm

# ==========================================
# 批量自动化车间：一口气冲刺 1,200 首！
# ==========================================
print("🚀 开始批量生产 1,200 首基于规则的竹笛乐谱...")

# 论文提到：自动生成具有可调参数（如键、音高范围和速度）的无限 MIDI 分数
tempos = [80, 90, 100, 110, 120, 130]  # 不同的速度流派
keys = [7, 0, 2, 5]  # 不同的调式根音（G, C, D, F）

total_files = 1200
files_per_config = total_files // (len(tempos) * len(keys)) # 平均分配参数组合

file_counter = 0

for t in tempos:
    for k in keys:
        for i in range(files_per_config):
            # 生成单首乐谱
            midi_score = generate_single_dizi_score(tempo=t, base_key=k, duration_seconds=30)
            
            # 命名规范：保存速度和调式元数据，方便后面特征提取
            file_name = f"dizi_t{t}_k{k}_id{i:03d}.mid"
            file_path = os.path.join(output_dir, file_name)
            
            # 写入磁盘
            midi_score.write(file_path)
            file_counter += 1

print(f"🎉 大功告成！已成功生成 {file_counter} 首独立的 MIDI 乐谱并存入 '{output_dir}' 文件夹。")

