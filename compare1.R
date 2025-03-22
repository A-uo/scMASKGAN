# 加载必要的R包
library(Seurat)
library(ggplot2)
library(patchwork)

# 读取数据（假设数据文件名为"Brainraw.csv"和"scNZ.csv"）
raw_data <- read.csv("D:\\scMASKGAN\\figure2-metric\\ERCC_csv\\ERCC.raw.csv", row.names = 1)
imp_data <- read.csv("D:\\scMASKGAN\\figure2-metric\\ERCC_csv\\ERCC_scZN.csv", row.names = 1)

# 创建Seurat对象，注意数据行为基因，列为barcode
raw_seurat <- CreateSeuratObject(counts = raw_data, project = "Dropout(33%)")
imp_seurat <- CreateSeuratObject(counts = imp_data, project = "ERCC")

celltype_vector <- read_csv("D:\\scMASKGAN\\figure2-metric\\label\\ERCC.label.csv", 
                            col_names = FALSE) %>% pull(1)

if(length(celltype_vector) != ncol(raw_data)) {
  stop("细胞类型标签数量与细胞数不匹配，请检查文件顺序是否一致！")
}

# 4. 将细胞类型信息添加到Seurat对象中
# 假定 celltype_vector 顺序与 raw_data/imp_data 列顺序一致
raw_seurat$celltype <- celltype_vector
imp_seurat$celltype <- celltype_vector

# 5. 计算零值比例
# raw_counts <- GetAssayData(raw_seurat, slot = "counts")
# imp_counts <- GetAssayData(imp_seurat, slot = "counts")
# 
# raw_zero_rate <- sum(raw_counts == 0) / length(raw_counts)
# imp_zero_rate <- sum(imp_counts < 0.1) / length(imp_counts)
# 
# print(paste("Raw zero rate:", raw_zero_rate))
# print(paste("Imputed zero rate:", imp_zero_rate))
# 
# # 6. 表达统计：计算每个基因的均值和方差
# raw_gene_mean <- apply(raw_counts, 1, mean)
# raw_gene_var  <- apply(raw_counts, 1, var)
# imp_gene_mean <- apply(imp_counts, 1, mean)
# imp_gene_var  <- apply(imp_counts, 1, var)
# 
# # 绘制均值和方差分布直方图
# p1 <- ggplot(data.frame(mean = raw_gene_mean), aes(x = mean)) + 
#   geom_histogram(bins = 50) + ggtitle("Raw Gene Mean Expression")
# p2 <- ggplot(data.frame(mean = imp_gene_mean), aes(x = mean)) + 
#   geom_histogram(bins = 50) + ggtitle("Imputed Gene Mean Expression")
# p3 <- ggplot(data.frame(var = raw_gene_var), aes(x = var)) + 
#   geom_histogram(bins = 50) + ggtitle("Raw Gene Variance")
# p4 <- ggplot(data.frame(var = imp_gene_var), aes(x = var)) + 
#   geom_histogram(bins = 50) + ggtitle("Imputed Gene Variance")
# 
# p1 + p2 + p3 + p4
# 
# # 7. 降维分析：归一化、标准化、PCA、t-SNE和UMAP
raw_seurat <- NormalizeData(raw_seurat)
raw_seurat <- ScaleData(raw_seurat)
raw_seurat <- RunPCA(raw_seurat, features = rownames(raw_seurat), verbose = FALSE)
raw_seurat <- RunTSNE(raw_seurat, dims = 1:10)
raw_seurat <- RunUMAP(raw_seurat, dims = 1:10)

imp_seurat <- NormalizeData(imp_seurat)
imp_seurat <- ScaleData(imp_seurat)
imp_seurat <- RunPCA(imp_seurat, features = rownames(imp_seurat), verbose = FALSE)
imp_seurat <- RunTSNE(imp_seurat, dims = 1:10)
imp_seurat <- RunUMAP(imp_seurat, dims = 1:10)

# 8. 绘制降维图，按 celltype 标签着色（无需聚类）
#p_raw_pca <- DimPlot(raw_seurat, reduction = "pca", group.by = "celltype", label = TRUE) + ggtitle("Raw PCA (celltype)")
#p_imp_pca <- DimPlot(imp_seurat, reduction = "pca", group.by = "celltype", label = TRUE) + ggtitle("Imputed PCA (celltype)")
p_raw_tsne <- DimPlot(raw_seurat, reduction = "tsne", group.by = "celltype", label = TRUE) + ggtitle("Raw t-SNE (celltype)")
p_imp_tsne <- DimPlot(imp_seurat, reduction = "tsne", group.by = "celltype", label = TRUE) + ggtitle("Imputed t-SNE (celltype)")
p_raw_umap <- DimPlot(raw_seurat, reduction = "umap", group.by = "celltype", label = TRUE) + ggtitle("Raw UMAP (celltype)")
p_imp_umap <- DimPlot(imp_seurat, reduction = "umap", group.by = "celltype", label = TRUE) + ggtitle("Imputed UMAP (celltype)")
# 
# # 并排展示降维结果
#(p_raw_pca + p_imp_pca) / 
(p_raw_tsne + p_imp_tsne)/(p_raw_umap+p_imp_umap)
# 
# # 聚类分析
# raw_seurat <- FindNeighbors(raw_seurat, dims = 1:10)
# raw_seurat <- FindClusters(raw_seurat, resolution = 0.8)
# imp_seurat <- FindNeighbors(imp_seurat, dims = 1:10)
# imp_seurat <- FindClusters(imp_seurat, resolution = 0.8)
# 
# # 聚类可视化
# p_raw_umap_cluster <- DimPlot(raw_seurat, reduction = "umap", label = TRUE, group.by = "seurat_clusters") + ggtitle("Raw UMAP Clusters")
# p_imp_umap_cluster <- DimPlot(imp_seurat, reduction = "umap", label = TRUE, group.by = "seurat_clusters") + ggtitle("Imputed UMAP Clusters")
# p_raw_umap_cluster + p_imp_umap_cluster

# 差异表达基因（DEG）分析（配对比较同一基因在两种数据中的表达）
# 这里假设可以根据细胞barcode匹配原始与插补数据，每个细胞一一对应
# 提取计数矩阵
raw_counts <- GetAssayData(object = raw_seurat, slot = "counts")
imp_counts <- GetAssayData(object = imp_seurat, slot = "counts")
common_cells <- intersect(colnames(raw_counts), colnames(imp_counts))
# 构建差异表达结果数据框
deg_results <- data.frame(
  gene = rownames(raw_counts),
  raw_mean = apply(raw_counts[, common_cells], 1, median),
  imp_mean = apply(imp_counts[, common_cells], 1, median)
)

deg_results$log2FC <- log2((deg_results$imp_mean+1) / (deg_results$raw_mean+1))

# 火山图示例
# remotes::install_github("kevinblighe/EnhancedVolcano")
# 
# library(EnhancedVolcano)
# EnhancedVolcano(deg_results,
#                 lab = deg_results$gene,
#                 x = 'log2FC',
#                 y = 'imp_mean', # 此处使用imp_mean的p值需要根据实际情况替换，这里仅为示例
#                 title = 'DEG Volcano Plot',
#                 pCutoff = 0.05)

# 批次效应评估：对比两种数据的整体表达特征
# 计算每个细胞总表达量
# 提取计数矩阵
# raw_counts <- GetAssayData(object = raw_seurat, slot = "counts")
# imp_counts <- GetAssayData(object = imp_seurat, slot = "counts")
# 
# # 计算每个细胞的总表达量
# raw_total_counts <- colSums(raw_counts)
# imp_total_counts <- colSums(imp_counts)
# 
# 
# df_total <- data.frame(Cell = names(raw_total_counts),
#                        Raw_Total = raw_total_counts,
#                        Imputed_Total = imp_total_counts[names(raw_total_counts)])
# 
# library(reshape2)
# df_total_m <- melt(df_total, id.vars = "Cell")
# ggplot(df_total_m, aes(x = variable, y = value)) + 
#   geom_boxplot() + ggtitle("Total Expression per Cell Comparison")
# 
# # 细胞表达基因数统计（非零基因数）
# # 提取计数矩阵
# raw_counts <- GetAssayData(object = raw_seurat, slot = "counts")
# imp_counts <- GetAssayData(object = imp_seurat, slot = "counts")
# 
# # 计算每个细胞检测到的基因数（非零计数）
# raw_nGenes <- apply(raw_counts, 2, function(x) sum(x > 0))
# imp_nGenes <- apply(imp_counts, 2, function(x) sum(x > 0.1))
# 
# df_ngenes <- data.frame(Cell = names(raw_nGenes),
#                         Raw_nGenes = raw_nGenes,
#                         Imputed_nGenes = imp_nGenes[names(raw_nGenes)])
# df_ngenes_m <- melt(df_ngenes, id.vars = "Cell")
# ggplot(df_ngenes_m, aes(x = variable, y = value)) + 
#   geom_boxplot() + ggtitle("Number of Detected Genes per Cell")

# 热图展示部分DEG在不同细胞中的表达模式
top_genes <- head(deg_results$gene[order(-abs(deg_results$log2FC))], 6)
# 提取计数矩阵
raw_counts <- GetAssayData(object = raw_seurat, slot = "counts")
imp_counts <- GetAssayData(object = imp_seurat, slot = "counts")

# 提取top_genes和共有细胞的数据子集
raw_mat <- raw_counts[top_genes, common_cells]
imp_mat <- imp_counts[top_genes, common_cells]
top_genes
# 使用Seurat内置的DoHeatmap函数
DoHeatmap(raw_seurat, features = top_genes, cells = common_cells) + ggtitle("Raw DEG Heatmap")
DoHeatmap(imp_seurat, features = top_genes, cells = common_cells) + ggtitle("Imputed DEG Heatmap")
p_heatmap_raw <- DoHeatmap(raw_seurat, features = top_genes, group.by = "celltype") +
  ggtitle("Raw DEG Heatmap")
p_heatmap_imp <- DoHeatmap(imp_seurat, features = top_genes, group.by = "celltype") +
  ggtitle("Imputed DEG Heatmap")
p_heatmap_raw
p_heatmap_imp

vln_raw <- VlnPlot(raw_seurat, features = top_genes, group.by = "celltype", pt.size = 0) 
vln_imp <- VlnPlot(imp_seurat, features = top_genes, group.by = "celltype", pt.size = 0)

# 显示原始数据和插补数据的小提琴图
vln_raw/vln_imp

# （2）绘制气泡图（Seurat中使用 DotPlot 实现）
dot_raw <- DotPlot(raw_seurat, features = top_genes, group.by = "celltype") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dot_imp <- DotPlot(imp_seurat, features = top_genes, group.by = "celltype") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 显示气泡图
print(dot_raw)
print(dot_imp)
p_feature_raw <- FeaturePlot(raw_seurat, 
                             features = top_genes, 
                             reduction = "umap", 
                             cols = c("lightgrey", "blue")) +
  ggtitle("Raw Data: UMAP Expression")

# 在插补数据的 UMAP 上展示这些基因的表达情况
p_feature_imp <- FeaturePlot(imp_seurat, 
                             features = top_genes, 
                             reduction = "umap", 
                             cols = c("lightgrey", "blue")) +
  ggtitle("Imputed Data: UMAP Expression")

# 并排显示两个 UMAP 图
p_feature_raw
p_feature_imp
# library(Seurat)
# library(ggplot2)
# library(patchwork)
# 
# # 2. 读取插补后的数据 (假设文件名为"scNZ.csv")
# #   数据格式：行为基因，列为细胞barcode
# imp_data <- read.csv("Brainraw_scNZ.csv", row.names = 1)
# 
# # 3. 创建Seurat对象
# imp_seurat <- CreateSeuratObject(counts = imp_data, project = "scNZ")
# 
# # 4. 将插补数据中表达值小于0.1的部分置为0
# #   4.1 提取计数矩阵
# imp_counts <- GetAssayData(imp_seurat, slot = "counts")
# #   4.2 替换小于0.1的值
# imp_counts[imp_counts < 0.1] <- 0
# #   4.3 将修改后的矩阵写回Seurat对象
# imp_seurat <- SetAssayData(object = imp_seurat, layer = "counts", new.data = imp_counts)
# 
# 
# # 5. (可选) 质控可视化：查看 nFeature_RNA、nCount_RNA 等
# #   - nFeature_RNA: 检测到的基因数
# #   - nCount_RNA:   总表达量（UMI/reads）
# # 这里仅作演示，可根据需要添加更多 QC 步骤或阈值过滤
# VlnPlot(imp_seurat, features = c("nFeature_RNA", "nCount_RNA"), ncol = 2) + 
#   ggtitle("Imputed data (after <0.1->0) QC metrics")
# 
# # 6. 下游分析流程
# #   6.1 归一化和标准化
# imp_seurat <- NormalizeData(imp_seurat)
# imp_seurat <- ScaleData(imp_seurat)
# 
# #   6.2 降维 (PCA, t-SNE, UMAP)
# #       若不想筛选高变基因，可指定全部基因:
# imp_seurat <- RunPCA(imp_seurat, features = rownames(imp_seurat), verbose = FALSE)
# imp_seurat <- RunTSNE(imp_seurat, dims = 1:10)
# imp_seurat <- RunUMAP(imp_seurat, dims = 1:10)
# 
# #   6.3 聚类 (可自行调节分辨率)
# imp_seurat <- FindNeighbors(imp_seurat, dims = 1:10)
# imp_seurat <- FindClusters(imp_seurat, resolution = 0.5)
# 
# #   6.4 可视化聚类结果
# DimPlot(imp_seurat, reduction = "umap", group.by = "seurat_clusters") + 
#   ggtitle("Imputed UMAP Clusters (after <0.1->0)")
