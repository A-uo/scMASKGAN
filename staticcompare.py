import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.tpu.feature_column_v2 import EmbeddingDevice

# 设置全局字体属性为 Times New Roman、加粗、黑色
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelweight"] = "bold"

# 读取保存好的指标表格（请根据实际路径修改）
metrics_df = pd.read_csv("outputpng/metrics_comparison_all.csv")

# 将数据从宽格式转换为长格式，便于绘制分面箱线图
# 这里选择展示 KS_stat、JS_distance 和 EMD 三个指标，若需要 MSE 可添加到 value_vars 中
metrics_long = pd.melt(metrics_df, id_vars=["Method"],
                       value_vars=["EMD"],
                       var_name="Metric", value_name="Value")
# , "JS_distance", "EMD"
# 设置 Seaborn 样式
sns.set(style="whitegrid", font_scale=2.4)

# 使用 FacetGrid 分面绘制箱线图，每个面板展示一种指标
g = sns.catplot(
    x="Method", y="Value", col="Metric", data=metrics_long,
    kind="box", col_wrap=1, height=5, aspect=1.5, palette="muted"
)

# 调整每个子图的 x 轴标签（旋转、字体等）
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",
                       fontsize=18, fontweight="bold", color="black")
    ax.set_xlabel("Method", fontsize=18, fontweight="bold", color="black")
    ax.set_ylabel("Value", fontsize=18, fontweight="bold", color="black")

plt.tight_layout()
plt.savefig("outputpng/metrics_comparison_EMD.png",dpi=300)
plt.show()
