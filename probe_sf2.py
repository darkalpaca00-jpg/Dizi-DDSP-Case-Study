from sf2utils.sf2parse import Sf2File

# 确保你的 Dizi.sf2 文件和这个脚本在同一个文件夹里
sf2_path = 'Dizi.sf2' 

with open(sf2_path, 'rb') as sf2_file:
    sf2 = Sf2File(sf2_file)
    print("成功读取音色库！包含以下预设：")
    for preset in sf2.presets:
        print(f"名称: {preset.name}, Bank: {preset.bank}, Preset: {preset.preset}")
        