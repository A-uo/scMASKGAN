import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# 使用 ggplot 样式美化图形
plt.style.use('ggplot')

# 设置全局字体属性为 Times New Roman、加粗、黑色
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelweight"] = "bold"

# 指定 CSV 文件所在的文件夹路径
folder_path = "D:\\scMASKGAN\\figure2-metric\\timecourse_csv"  # 根据实际情况修改

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
            #标准化数据（Z-score）
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
    plt.title("55%", fontsize=22, fontweight="bold", color="black")
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

    #保存图像，dpi 可调，数值越大图像越清晰
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")

    plt.show()

    # -------------------------------
    # 计算与 raw 方法比较的统计指标
    # -------------------------------
    if "raw" not in labels:
        print("未找到 raw 方法数据，无法进行对比。")
    else:
        raw_index = labels.index("raw")
        raw_data = data_list[raw_index]
        results = []
        for i, method in enumerate(labels):
            method_data = data_list[i]
            if method == "raw":
                ks_stat, ks_p = 0.0, 1.0
                js_distance = 0.0
                emd = 0.0
                mse = 0.0
            else:
                # Kolmogorov-Smirnov 检验
                ks_res = ks_2samp(raw_data, method_data)
                ks_stat, ks_p = ks_res.statistic, ks_res.pvalue

                # 计算 Jensen-Shannon 距离与 MSE：先构造直方图（统一50个 bin）
                combined = np.concatenate((raw_data, method_data))
                bins = np.histogram_bin_edges(combined, bins=50)
                p_raw, _ = np.histogram(raw_data, bins=bins, density=False)
                p_method, _ = np.histogram(method_data, bins=bins, density=False)
                p_raw = p_raw / p_raw.sum()
                p_method = p_method / p_method.sum()

                js_distance = jensenshannon(p_raw, p_method)
                mse = np.mean((p_raw - p_method)**2)

                # Earth Mover's Distance（Wasserstein 距离）
                emd = wasserstein_distance(raw_data, method_data)
            results.append({
                "Method": method,
                "KS_stat": ks_stat,
                "KS_pvalue": ks_p,
                "JS_distance": js_distance,
                "EMD": emd,
                "MSE": mse
            })

        metrics_df = pd.DataFrame(results)
        print("\n各方法与 raw 方法比较的统计指标：")
        print(metrics_df)
        # 如有需要，可保存结果到 CSV 文件
        metrics_output = os.path.join(output_folder, "metrics_comparison_time.csv")
        metrics_df.to_csv(metrics_output, index=False)
        print(f"指标表格已保存至: {metrics_output}")
else:
    print("在文件夹中未找到任何匹配的 CSV 文件。")
