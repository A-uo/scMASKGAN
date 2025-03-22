import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 使用 ggplot 样式美化图形
plt.style.use('ggplot')

# 设置全局字体属性为 Times New Roman、加粗、黑色
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelweight"] = "bold"

# 指定 CSV 文件所在的文件夹路径
folder_path = "D:\\scMASKGAN\\figure2-metric\\sc10x_csv"  # 根据实际情况修改

# 用于存储每个方法对应的数据和标签
data_list = []
labels = []

# 预先定义好方法名称列表（按照给定顺序）
methods = [
    "raw", "DCA", "Deepimpute", "scIGANs", "scMASKGAN",
    "AutoImpute", "DrImpute", "ENHANCE", "MAGIC", "SAVER",
    "SCRABBLE", "VIPER", "scGAIN", "scimpute"
]

# 遍历每个方法，查找文件名中包含该字符串（若有多个，仅取第一个）
for method in methods:
    file_found = None
    for f in os.listdir(folder_path):
        if f.endswith('.csv') and method in f:
            file_found = os.path.join(folder_path, f)
            break
    if file_found is not None:
        print(f"正在处理 {file_found} 对应方法: {method}")
        try:
            # 根据 CSV 文件格式选择是否需要 header、index_col 参数
            df = pd.read_csv(file_found, header=0, index_col=0)
            # 将数据转换为一维数组，并过滤 NaN 和 0 值
            values = df.values.flatten()
            values = values[~np.isnan(values)]
            values = values[values > 0]
            # 标准化数据（Z-score）
            standardized_values = (values - np.mean(values)) / np.std(values)
            data_list.append(standardized_values)
            labels.append(method)
        except Exception as e:
            print(f"处理文件 {file_found} 时发生错误: {e}")
    else:
        print(f"未找到与 {method} 对应的 CSV 文件，跳过。")

# 如果至少有一个文件被处理，则绘制竖直箱线图
if data_list:
    # 设置局部字体大小和线宽
    plt.rcParams.update({'font.size': 18, 'lines.linewidth': 2})

    # 定义箱线图各元素的属性
    boxprops = dict(linewidth=2, color='black')
    whiskerprops = dict(linewidth=2, color='black')
    capprops = dict(linewidth=2, color='black')
    medianprops = dict(linewidth=2, color='red')

    plt.figure(figsize=(12, 6))
    # 默认绘制竖直箱线图（vert=True），并使用 patch_artist 支持箱体上色
    bp = plt.boxplot(data_list, labels=labels, showfliers=False,
                     patch_artist=True,
                     boxprops=boxprops,
                     whiskerprops=whiskerprops,
                     capprops=capprops,
                     medianprops=medianprops)

    # 为每个箱体赋予不同的颜色（使用 tab10 颜色映射）
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("45%", fontsize=22, fontweight="bold", color="black")
    plt.xlabel("Method", fontsize=22, fontweight="bold", color="black")
    plt.ylabel("Distribution", fontsize=22, fontweight="bold", color="black")
    plt.xticks(rotation=90, fontsize=14, fontweight="bold", color="black")
    plt.yticks(fontsize=14, fontweight="bold", color="black")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 指定输出文件夹路径，若不存在则创建
    output_folder = "outputpng"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "boxplot_standardized_vertical_time.png")

    # 保存图像，dpi 可调，数值越大图像越清晰
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")

    plt.show()
else:
    print("在文件夹中未找到任何匹配的 CSV 文件。")
