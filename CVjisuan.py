import os
import glob
import pandas as pd
import numpy as np

# 修改为你的CSV文件所在的文件夹路径
data_folder = "D:\\scMASKGAN\\figure2-metric\\timecourse_csv"  # 例如 "data/csv_files"
csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))

# 用于存储结果的列表
results = []


# 定义计算单个Series CV的函数（均值为0时返回NaN）
def coefficient_of_variation(series):
    mean_val = series.median()
    if mean_val == 0:
        return np.nan
    return series.std() / mean_val


# 遍历所有CSV文件
for csv_file in csv_files:
    try:
        # 读取CSV文件，默认认为第一行为表头
        df = pd.read_csv(csv_file, header=0)
        # 如果第一列数据不是数值型，则认为是行名，重新读取文件
        if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            df = pd.read_csv(csv_file, header=0, index_col=0)
    except Exception as e:
        print(f"读取文件 {csv_file} 时出错：{e}")
        continue

    # 转换为数值型数据（非数值型自动转换为 NaN）
    df = df.apply(pd.to_numeric, errors='coerce')

    # 对每个数值型列计算 CV，过滤均值为0的列
    cv_values = []
    for col in df.columns:
        series = df[col].dropna()
        mean_val = series.mean()
        if mean_val != 0:
            cv = series.std() / mean_val
            cv_values.append(cv)

    # 对该文件所有列的CV取平均值，如果没有有效列则记为NaN
    avg_cv = np.mean(cv_values) if cv_values else np.nan

    results.append({
        "Filename": os.path.basename(csv_file),
        "Average_CV": avg_cv
    })

# 将结果转换为DataFrame并显示
cv_df = pd.DataFrame(results)
print(cv_df)

# 保存结果到指定文件夹下的CSV文件中
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "timecourse_csv.csv")
cv_df.to_csv(output_file, index=False)
print(f"CV汇总结果已保存至 {output_file}")
