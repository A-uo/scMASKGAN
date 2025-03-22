from pydoc import resolve

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# **1. 加载数据**
# 原始数据 (Observed)
original_data_path = "./ALL/Original/GSM5768752_GN1_UMI_COUNTS_RAW.csv"  # 修改为原始数据文件的路径
df_original = pd.read_csv(original_data_path, index_col=0)  # 行名是基因，列名是细胞
adata_original = sc.AnnData(df_original.T)  # 转置为细胞 × 基因格式

# 插补数据 (Imputed)
# imputed_data_path = "./GanData/scIGANs-GSM5768752.csv"  # 修改为插补数据文件的路径
imputed_data_path = "./ALL/DCA/GSM5768752.csv"
df_imputed = pd.read_csv(imputed_data_path, index_col=0)
adata_imputed = sc.AnnData(df_imputed.T)

# **2. 数据整合**
# 添加批次信息
adata_original.obs['batch'] = 'original'
adata_imputed.obs['batch'] = 'imputed'


def preprocess(adata):
    """
    对 AnnData 对象进行标准预处理，包括：
    1. 筛选低质量细胞和基因
    2. 数据归一化和对数化
    3. 筛选高变基因
    4. 数据标准化和 PCA
    """
    # # 筛选细胞和基因
    sc.pp.filter_cells(adata, min_genes=200)  # 每个细胞至少有 200 个基因被检测到
    sc.pp.filter_genes(adata, min_cells=3)  # 每个基因至少在 3 个细胞中被检测到
    # #
    # # #归一化并对数化
    sc.pp.normalize_total(adata, target_sum=1e4)  # 每个细胞的表达量归一化到总和为 1e4
    sc.pp.log1p(adata)  # 对数据进行 log(1+x) 变换
    # # #
    # # # # 筛选高变基因
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat")  # 筛选 top 2000 高变基因
    adata = adata[:, adata.var.highly_variable]  # 仅保留高变基因
    # #
    # # #标准化和降维
    sc.pp.scale(adata, max_value=10)  # 数据标准化，限制最大值为 10
    sc.tl.pca(adata, n_comps=50)  # PCA 降维
    return adata


# **3. 对数据进行预处理**
adata_original = preprocess(adata_original)
adata_imputed = preprocess(adata_imputed)
sc.pp.neighbors(adata_original, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_original)
sc.pp.neighbors(adata_imputed, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_imputed)
# **4. 数据整合**
# 添加批次信息
adata_original.obs['batch'] = 'original'
adata_imputed.obs['batch'] = 'imputed'

# 合并原始数据和插补数据
adata_combined = adata_original.concatenate(
    adata_imputed, batch_key="batch", batch_categories=["original", "imputed"]
)
sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_combined)

# 检查 UMAP 是否生成
print("UMAP coordinates stored in:", list(adata_combined.obsm.keys()))  # 应输出 ['X_umap']

# **5. 差异基因分析**
# 使用 Scanpy 的 rank_genes_groups 方法
sc.tl.rank_genes_groups(adata_combined, groupby='batch', reference='original', method='wilcoxon')
print(adata_combined.uns.keys())  # 检查是否存在 'rank_genes_groups'


print(adata_combined.uns['rank_genes_groups'])

# **6. 提取差异基因结果**
def extract_rank_genes(adata):
    """
    提取差异基因分析的结果，包括基因名称、logFC 和 p 值。
    """
    if 'rank_genes_groups' not in adata.uns:
        raise ValueError(
            "`rank_genes_groups` is not found in `adata.uns`. Please check if the `rank_genes_groups` function ran successfully.")

    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names  # 提取分组名称

    # 确保所有必要的字段存在
    if not all(key in result for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']):
        raise ValueError(
            "Missing one of the keys (`names`, `logfoldchanges`, `pvals`, `pvals_adj`) in `rank_genes_groups` results.")

    # 初始化空 DataFrame
    gene_df = pd.DataFrame()

    # 遍历每个分组提取数据
    for group in groups:
        group_data = pd.DataFrame({
            "group": group,
            "genes": result["names"][group],
            "logfoldchanges": result["logfoldchanges"][group],
            "pvals": result["pvals"][group],
            "pvals_adj": result["pvals_adj"][group]
        })

        # 去除 NaN 值
        group_data = group_data.dropna(subset=["logfoldchanges", "pvals", "pvals_adj"])

        # 合并到主 DataFrame
        gene_df = pd.concat([gene_df, group_data], ignore_index=True)

    return gene_df


# 提取火山图所需数据
diff_genes = extract_rank_genes(adata_combined)

# 计算 -log10(p值)
diff_genes['-log10(pvals)'] = -np.log10(diff_genes['pvals'])

# 筛选显著基因
threshold_logfc = 1.0  # logFC 阈值
threshold_pval = 0.05  # p 值阈值
diff_genes['Significant'] = (
        (diff_genes['logfoldchanges'].abs() > threshold_logfc) &
        (diff_genes['pvals'] < threshold_pval)
)
diff_genes.to_csv("diff_genes_results.csv", index=False)
print(diff_genes)
# **7. 绘制火山图**
def plot_volcano_with_labels(diff_genes, save_path=None, top_n=10):
    """
    绘制火山图，并对显著上调和下调的基因名称进行标注。

    参数：
    - diff_genes: 差异基因数据表。
    - save_path: 保存路径（可选）。
    - top_n: 标注的显著基因数量。
    """
    # 移除无效数据
    diff_genes = diff_genes.dropna(subset=["logfoldchanges", "pvals"])

    # 计算 -log10(p值)
    diff_genes['-log10(pvals)'] = -np.log10(diff_genes['pvals'])

    # 筛选显著基因
    threshold_logfc = 1.0  # logFC 阈值
    threshold_pval = 0.05  # p 值阈值
    diff_genes['Significant'] = (
            (diff_genes['logfoldchanges'].abs() > threshold_logfc) &
            (diff_genes['pvals'] < threshold_pval)
    )

    # 筛选显著上调和下调基因
    upregulated = diff_genes[(diff_genes['Significant']) & (diff_genes['logfoldchanges'] > 0)]
    downregulated = diff_genes[(diff_genes['Significant']) & (diff_genes['logfoldchanges'] < 0)]

    # 按 -log10(pval) 排序，选择 top_n 基因进行标注
    top_up = upregulated.nlargest(top_n, '-log10(pvals)')
    top_down = downregulated.nlargest(top_n, '-log10(pvals)')

    # 绘制火山图
    plt.figure(figsize=(12, 10))

    # 所有基因散点
    plt.scatter(
        diff_genes['logfoldchanges'],
        diff_genes['-log10(pvals)'],
        c='lightgray',
        alpha=0.7,
        s=10,
        label="All Genes"
    )

    # 显著上调基因
    plt.scatter(
        upregulated['logfoldchanges'],
        upregulated['-log10(pvals)'],
        c='red',
        alpha=0.8,
        s=20,
        label="Significant Upregulated"
    )

    # 显著下调基因
    plt.scatter(
        downregulated['logfoldchanges'],
        downregulated['-log10(pvals)'],
        c='blue',
        alpha=0.8,
        s=20,
        label="Significant Downregulated"
    )

    # 标注显著上调基因名称
    for i, row in top_up.iterrows():
        plt.annotate(
            row['genes'],
            (row['logfoldchanges'], row['-log10(pvals)']),
            fontsize=10,
            color='darkred',
            ha='right',
            arrowprops=dict(arrowstyle='-', color='darkred', lw=0.5)
        )

    # 标注显著下调基因名称
    for i, row in top_down.iterrows():
        plt.annotate(
            row['genes'],
            (row['logfoldchanges'], row['-log10(pvals)']),
            fontsize=10,
            color='darkblue',
            ha='left',
            arrowprops=dict(arrowstyle='-', color='darkblue', lw=0.5)
        )

    # 添加轴标签和标题
    plt.axhline(y=-np.log10(threshold_pval), color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=threshold_logfc, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=-threshold_logfc, color='gray', linestyle='--', linewidth=1)

    plt.xlabel("Log Fold Change", fontsize=14)
    plt.ylabel("-Log10(p-value)", fontsize=14)
    plt.title("Volcano Plot: Original vs Imputed", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(False)

    # 保存或展示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# 绘制火山图
plot_volcano_with_labels(diff_genes, save_path="volcano_plot.png")
genes_to_plot = [ "ALK", "ATRX", "HIF1A", "NTRK1", "NTRK2", "CXCR4"]

# 提取基因表达矩阵（仅保留感兴趣的基因）
expression_data = adata_combined[:, [gene for gene in genes_to_plot if gene in adata_combined.var_names]].to_df()

# 添加分组信息（batch）
expression_data['batch'] = adata_combined.obs['batch'].values

# 将数据转换为长格式，方便绘图
long_data = expression_data.melt(id_vars=['batch'], var_name='Gene', value_name='Expression')

# **绘制对称小提琴图**
plt.figure(figsize=(8, 4))

sns.violinplot(
    x="Gene",
    y="Expression",
    hue="batch",
    data=long_data,
    split=True,  # 左右对称的小提琴图
    inner="quartile",  # 添加四分位信息
    scale="width",  # 调整宽度
    palette=["salmon", "skyblue"]  # 设置颜色
)

# 调整字体大小
plt.title("Gene Expression", fontsize=22, weight='bold')  # 标题字体
# plt.xlabel("Gene", fontsize=20, weight='bold')  # x轴标签字体
# plt.ylabel("Expression", fontsize=20, weight='bold')  # y轴标签字体
plt.xlabel("")
plt.ylabel("")
plt.xticks(fontsize=16, rotation=45)  # x轴刻度字体大小并旋转
plt.yticks(fontsize=16)  # y轴刻度字体大小
plt.legend(title="Batch", loc="upper right", fontsize=16, title_fontsize=18)  # 图例字体大小

plt.tight_layout()  # 自动调整布局避免标签重叠

# 保存图像或显示
plt.savefig("split_violin_plot_with_large_fonts.png", dpi=300)
plt.show()

# **绘制对比基因表达的 UMAP 散点图 (分两张图)**
# **绘制对比基因表达的 UMAP 散点图 (分两张图)**
for gene in genes_to_plot:
    if gene in adata_combined.var_names:  # 确保基因在数据集中存在
        # 提取原始数据和插补数据的 UMAP 坐标
        umap_original = adata_original.obsm['X_umap']
        umap_imputed = adata_imputed.obsm['X_umap']

        # 提取原始数据和插补数据中该基因的表达量
        expression_original = adata_original[:, gene].X.flatten()
        expression_imputed = adata_imputed[:, gene].X.flatten()

        # **绘制原始数据的 UMAP 图**
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(
            umap_original[:, 0], umap_original[:, 1],
            c=expression_original, cmap="Reds", s=40, alpha=0.8, edgecolor="k",marker='o'
        )
        plt.title(f"{gene} (Original)", fontsize=22, weight='bold')
        plt.xlabel("UMAP 1", fontsize=18)
        plt.ylabel("UMAP 2", fontsize=18)
        cbar = plt.colorbar(scatter)
        cbar.ax.tick_params(labelsize=18)  # 设置 colorbar 字体大小
        cbar.set_label("Expression Level", fontsize=18)  # 设置 colorbar 标签字体大小
        plt.xticks([])  # 移除 x 坐标轴的刻度值
        plt.yticks([])  # 移除 y 坐标轴的刻度值
        plt.tight_layout()
        #plt.savefig(f"{gene}_Original.png", dpi=300)
        plt.show()

        # **绘制插补数据的 UMAP 图**
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(
            umap_imputed[:, 0], umap_imputed[:, 1],
            c=expression_imputed, cmap="Reds", s=40, alpha=0.8, edgecolor="k", marker='o'
        )
        plt.title(f"{gene} (Imputed)", fontsize=22, weight='bold')
        plt.xlabel("UMAP 1", fontsize=18)
        plt.ylabel("UMAP 2", fontsize=18)
        cbar = plt.colorbar(scatter)
        cbar.ax.tick_params(labelsize=18)  # 设置 colorbar 字体大小
        cbar.set_label("Expression Level", fontsize=18)  # 设置 colorbar 标签字体大小
        plt.xticks([])  # 移除 x 坐标轴的刻度值
        plt.yticks([])  # 移除 y 坐标轴的刻度值
        plt.tight_layout()
        #plt.savefig(f"{gene}_Imputed.png", dpi=300)
        plt.show()

    else:
        print(f"Gene {gene} is not present in the dataset.")

