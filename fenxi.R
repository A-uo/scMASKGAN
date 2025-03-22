# ====================================================================
# 单细胞RNA-seq数据分析流程：Seurat完整脚本（基于特定CSV格式）
# ====================================================================

# ------------------------------
# 1. 安装和加载必要的R包
# ------------------------------
# 如果尚未安装所需的包，请取消以下代码的注释并运行
# install.packages("Seurat")
# install.packages(c("ggplot2", "dplyr", "patchwork"))

# 加载包
library(Seurat)
library(ggplot2)
library(dplyr)
library(patchwork)

# ------------------------------
# 2. 读取和转换数据
# ------------------------------
# 替换 "your_data.csv" 为你的数据文件路径
# 假设第一列是基因名称，第一行（除第一列）是细胞元数据，剩余部分是表达量

# 读取CSV文件
raw_data <- read.csv("D:/LIGImpute-main/LIGImpute-main/scIGANs-GSM5768752.csv", 
                     header = TRUE, 
                     stringsAsFactors = FALSE, 
                     check.names = FALSE)

# 查看数据结构
head(raw_data)

# 提取元数据（第一行，除第一列）
cell_metadata <- raw_data[1, -1]  # 去掉第一列（基因名）
cell_ids <- colnames(raw_data)[-1]  # 获取细胞ID

# 提取表达数据（去掉第一行）
expression_data <- raw_data[-1, ]

# 设置行名为基因名称
rownames(expression_data) <- expression_data[,1]  # 假设第一列是基因名称
expression_data <- expression_data[,-1]  # 去掉基因名称列

# 转换为数值矩阵
expression_matrix <- as.matrix(expression_data)
storage.mode(expression_matrix) <- "numeric"

# 查看表达矩阵
dim(expression_matrix)
head(expression_matrix)

# ------------------------------
# 3. 创建Seurat对象
# ------------------------------
# 创建Seurat对象，项目名称为 "SingleCellProject"
seurat_obj <- CreateSeuratObject(counts = expression_matrix, 
                                 project = "SingleCellProject", 
                                 min.cells = 3, 
                                 min.features = 200)

# ------------------------------
# 4. 添加元数据到Seurat对象
# ------------------------------
# 创建元数据数据框
# 假设cell_metadata包含单一属性（如CellType），如果有多个属性，请相应调整

# 例如，假设cell_metadata包含"CellType"信息
# 如果有多个元数据属性，请确保它们在cell_metadata中以不同的列存在

# 创建元数据数据框
metadata_df <- data.frame(CellType = as.character(cell_metadata), 
                          row.names = cell_ids)

# 添加元数据到Seurat对象
seurat_obj <- AddMetaData(seurat_obj, metadata = metadata_df)

# 查看Seurat对象的元数据
head(seurat_obj@meta.data)

# ------------------------------
# 5. 质量控制（QC）
# ------------------------------
# 计算线粒体基因比例
# 根据物种和注释文件，线粒体基因的前缀可能不同
# 这里以人类为例，线粒体基因以"MT-"开头
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")

# 可视化QC指标
VlnPlot(seurat_obj, 
        features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
        ncol = 3) + 
  NoLegend()

# 绘制特征-特征关系图
FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt") +
  FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")

# 根据QC指标过滤细胞
seurat_obj <- subset(seurat_obj, 
                     subset = nFeature_RNA > 200 & 
                       nFeature_RNA < 2500 & 
                       percent.mt < 5)

# 查看过滤后的Seurat对象
print(seurat_obj)

# ------------------------------
# 6. 数据归一化
# ------------------------------
seurat_obj <- NormalizeData(seurat_obj, 
                            normalization.method = "LogNormalize", 
                            scale.factor = 10000)

# 查看归一化后的数据（可选）
# head(seurat_obj@assays$RNA@data)

# ------------------------------
# 7. 识别高变基因
# ------------------------------
seurat_obj <- FindVariableFeatures(seurat_obj, 
                                   selection.method = "vst", 
                                   nfeatures = 2000)

# 查看高变基因
top10 <- head(VariableFeatures(seurat_obj), 10)
VariableFeaturePlot(seurat_obj) + LabelPoints(points = top10, repel = TRUE)

# ------------------------------
# 8. 数据尺度化（Scaling）
# ------------------------------
# 回归掉一些技术变量，如线粒体基因比例和细胞总RNA计数
seurat_obj <- ScaleData(seurat_obj, 
                        vars.to.regress = c("nCount_RNA", "percent.mt"))

# 查看尺度化后的数据（可选）
# head(seurat_obj@assays$RNA@scale.data)

# ------------------------------
# 9. 线性降维（PCA）
# ------------------------------
seurat_obj <- RunPCA(seurat_obj, 
                     features = VariableFeatures(object = seurat_obj))

# 可视化PCA结果
VizDimLoadings(seurat_obj, dims = 1:2, reduction = "pca")
DimPlot(seurat_obj, reduction = "pca")
DimHeatmap(seurat_obj, dims = 1:15, cells = 500, balanced = TRUE)

# 查看PCA的方差解释
print(seurat_obj[["pca"]], dims = 1:5, nfeatures = 5)

# 选择主成分数量
ElbowPlot(seurat_obj)

# ------------------------------
# 10. 非线性降维（UMAP）
# ------------------------------
# 通常选择前10-20个主成分，根据ElbowPlot结果调整
num_pcs <- 10  # 根据ElbowPlot调整

seurat_obj <- RunUMAP(seurat_obj, dims = 1:num_pcs)
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:num_pcs)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# 可视化 UMAP
DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()

# ------------------------------
# 11. 细胞类型注释
# ------------------------------
# 定义已知的标记基因，根据物种和研究对象调整
marker_genes <- c("CD3D", "CD14", "MS4A1", "FCER1A", "LYZ", "PPBP")

# 绘制特征基因的表达图
FeaturePlot(seurat_obj, features = marker_genes)

# 根据表达模式手动标注细胞类型
# 例如：
# 假设聚类0为T细胞，聚类1为B细胞，依此类推
new.cluster.ids <- c("T cells", "B cells", "Monocytes", "NK cells", "Dendritic cells", 
                     "Platelets", "Erythrocytes", "Endothelial", "Fibroblasts", "Unknown")

# 确保 new.cluster.ids 的长度与当前聚类数一致
if(length(new.cluster.ids) == length(levels(seurat_obj))){
  names(new.cluster.ids) <- levels(seurat_obj)
  seurat_obj <- RenameIdents(seurat_obj, new.cluster.ids)
} else {
  warning("new.cluster.ids 的长度与当前聚类数不匹配，请检查并调整。")
}

# 查看标注后的结果
DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()

# ------------------------------
# 12. 差异表达分析
# ------------------------------
# 以某个细胞群为例，寻找与其他群体的差异表达基因
cluster.markers <- FindMarkers(seurat_obj, ident.1 = "T cells", min.pct = 0.25)

# 查看差异表达基因
head(cluster.markers, n = 10)

# 或者寻找所有群体的标记基因
all.markers <- FindAllMarkers(seurat_obj, 
                              only.pos = TRUE, 
                              min.pct = 0.25, 
                              logfc.threshold = 0.25)

# 查看前几个标记基因
head(all.markers)

# ------------------------------
# 13. 保存和导出结果
# ------------------------------
# 保存Seurat对象
saveRDS(seurat_obj, file = "seurat_final.rds")

# 导出所有标记基因到CSV文件
write.csv(all.markers, file = "all_markers.csv")

# ------------------------------
# 完成
# ------------------------------
# 以上脚本涵盖了单细胞RNA-seq数据分析的主要步骤。根据具体的数据和研究需求，你可能需要进行额外的分析或调整参数。建议参考[Seurat官方文档](https://satijalab.org/seurat/)以获取更多详细信息和高级功能。
