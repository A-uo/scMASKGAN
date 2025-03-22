import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体属性为 Times New Roman、加粗、黑色
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelweight"] = "bold"

# 指定 CSV 文件所在的文件夹路径
folder_path = "D:\\AutoImpute-master\\AutoImpute-master\\AutoImpute Model\\GanData"  # 根据实际情况修改

# 预先定义好 Dropout率列表（顺序对应于文件的排序）
dropout_rates = ["95%", "91%", "97%", "85%", "96%", "97%", "97%", "97%", "84%", "97%"]

# 读取文件夹中所有 CSV 文件，并排序后取前 10 个
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort()  # 保证顺序一致
if len(csv_files) < 10:
    print("警告：CSV 文件数量少于 10 个")
csv_files = csv_files[:10]

data_list = []
labels = []

# 遍历这 10 个 CSV 文件
for i, f in enumerate(csv_files):
    file_path = os.path.join(folder_path, f)
    print(f"正在处理文件: {file_path} 对应 Dropout率: {dropout_rates[i]}")
    try:
        df = pd.read_csv(file_path, header=0, index_col=0)
        # 将数据转换为一维数组
        values = df.values.flatten()
        # 过滤 NaN 和 0 值（0值可能导致标准化分母过小）
        values = values[~np.isnan(values)]
        values = values[values > 0]
        # 对数据进行标准化处理（Z-score 标准化）
        standardized_values = (values - np.mean(values)) / np.std(values)
        data_list.append(standardized_values)
        labels.append(dropout_rates[i])
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")

# 如果至少有一个文件被处理，则绘制横向箱线图，并为每个箱上色
if data_list:
    # 设置局部字体大小和线宽
    plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2})

    # 定义箱线图各元素的属性
    boxprops = dict(linewidth=2, color='black')
    whiskerprops = dict(linewidth=2, color='black')
    capprops = dict(linewidth=2, color='black')
    medianprops = dict(linewidth=2, color='red')

    # 创建图像，调整尺寸为较高的竖向图
    plt.figure(figsize=(8, 10))
    # 使用 patch_artist=True 来支持箱体上色，并设置 vert=False 绘制横向箱线图
    bp = plt.boxplot(data_list, labels=labels, showfliers=False,
                     patch_artist=True,
                     boxprops=boxprops,
                     whiskerprops=whiskerprops,
                     capprops=capprops,
                     medianprops=medianprops,
                     vert=False)

    # 获取不同的颜色（这里使用 tab10）
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel("Dropout", fontsize=22, fontweight="bold", color="black")
    plt.xlabel("Standardized", fontsize=22, fontweight="bold", color="black")
    plt.yticks(fontsize=22, fontweight="bold", color="black")
    plt.xticks(fontsize=22, fontweight="bold", color="black")
    plt.grid(True, linestyle="--", alpha=0.7)

    # 设置横轴坐标范围为 -0.3 到 0.4
    plt.xlim(-0.3, 0.4)

    plt.tight_layout()

    # 指定输出文件夹（若不存在则创建）
    output_folder = "outputpng"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "horizontal_boxplot_dropout_colored.png")

    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()
else:
    print("在文件夹中未找到任何 CSV 文件。")
