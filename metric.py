import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = "supplementaryTable3.csv"  # 替换为实际文件路径
df = pd.read_csv(file_path, encoding="ISO-8859-1")


# 设置绘图风格
sns.set_theme(style="whitegrid", palette="pastel")  # 设置主题和颜色

# 设置绘图区域
plt.figure(figsize=(16, 24))

# 定义颜色调色盘（每个方法不同颜色）
palette = sns.color_palette("Set2", n_colors=df["Method"].nunique())

# 函数：加粗边框
def set_spines_bold(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

# 绘制 Accuracy 箱线图
plt.subplot(3, 1, 1)  # 4行1列，第1个图
ax1 = sns.boxplot(x="Method", y="ACC", data=df, palette=palette)
plt.title("Accuracy Comparison", fontsize=40, fontweight="bold")  # 标题字体加粗
plt.xlabel("Method", fontsize=40)
plt.ylabel("Accuracy", fontsize=40)
plt.xticks(rotation=90, fontsize=30)  # X轴标签旋转并加大字体
plt.yticks(fontsize=30)  # Y轴刻度字体
set_spines_bold(ax1)  # 加粗边框

# 绘制 F1 Score 箱线图
plt.subplot(3, 1, 2)  # 4行1列，第2个图
ax2 = sns.boxplot(x="Method", y="F1Score", data=df, palette=palette)
plt.title("F1 Score Comparison", fontsize=40, fontweight="bold")
plt.xlabel("Method", fontsize=40)
plt.ylabel("F1 Score", fontsize=40)
plt.xticks(rotation=90, fontsize=30)
plt.yticks(fontsize=30)
set_spines_bold(ax2)  # 加粗边框

# # 绘制 Recall 箱线图
# plt.subplot(1, 1, 1)  # 4行1列，第3个图
# ax3 = sns.boxplot(x="Method", y="Pearson Correlation", data=df, palette=palette)
# plt.title("Pearson Correlation Comparison", fontsize=22, fontweight="bold")
# plt.xlabel("Method", fontsize=22)
# plt.ylabel("Pearson Correlation", fontsize=22)
# plt.xticks(rotation=90, fontsize=22)
# plt.yticks(fontsize=18)
# set_spines_bold(ax3)  # 加粗边框

# 绘制 AUC 箱线图
plt.subplot(3, 1, 3)  # 4行1列，第4个图
ax3 = sns.boxplot(x="Method", y="AUC", data=df, palette=palette)
plt.title("AUC Comparison", fontsize=40, fontweight="bold")
plt.xlabel("Method", fontsize=40)
plt.ylabel("AUC", fontsize=40)
plt.xticks(rotation=90, fontsize=30)
plt.yticks(fontsize=30)
set_spines_bold(ax3)  # 加粗边框

# 调整布局和标题间距
plt.tight_layout(pad=3.0)  # 设置图与图之间的间距
plt.savefig("metric.pdf", dpi=300)  # 保存图像为高分辨率
plt.show()
