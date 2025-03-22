library(dplyr)
library(tidyr)
library(monocle3)
library(ggplot2)
library(clusterProfiler)
library(org.Hs.eg.db)
library(org.Mm.eg.db)  # 小鼠基因数据库
library(enrichplot)    # 用于可视化

# 加载原始数据
raw_data <- read.csv("D:\\shiyan\\Section_4.zip\\Section_4\\data/Timecourse.raw.tsv", sep = "\t", row.names = 1)

# 加载插补数据
imputed_data <- read.csv("D:\\shiyan\\Section_4.zip\\Section_4\\data/scMASKGAN-timecourse.csv", row.names = 1)

# 创建原始数据的元数据
raw_metadata <- data.frame(
  cell_id = colnames(raw_data), 
  time_point = as.numeric(gsub("H9\\.(\\d+)h.*", "\\1", colnames(raw_data))) # 提取时间点
)
rownames(raw_metadata) <- colnames(raw_data) # 设置元数据的行名与表达矩阵的列名一致

# 创建插补数据的元数据
imputed_metadata <- data.frame(
  cell_id = colnames(imputed_data), 
  time_point = as.numeric(gsub("H9\\.(\\d+)h.*", "\\1", colnames(imputed_data))) # 提取时间点
)
rownames(imputed_metadata) <- colnames(imputed_data) # 设置元数据的行名与表达矩阵的列名一致

# 创建基因元数据
gene_metadata <- data.frame(gene_id = rownames(raw_data))
rownames(gene_metadata) <- rownames(raw_data) # 设置基因元数据的行名与表达矩阵的行名一致

# 构建原始数据的 CellDataSet
raw_cds <- new_cell_data_set(
  expression_data = as.matrix(raw_data),
  cell_metadata = raw_metadata,
  gene_metadata = gene_metadata
)

# 构建插补数据的 CellDataSet
imputed_cds <- new_cell_data_set(
  expression_data = as.matrix(imputed_data),
  cell_metadata = imputed_metadata,
  gene_metadata = gene_metadata
)

# 原始数据降维与轨迹构建
raw_cds <- preprocess_cds(raw_cds)
raw_cds <- reduce_dimension(raw_cds, preprocess_method = "PCA")
raw_cds <- cluster_cells(raw_cds)
raw_cds <- learn_graph(raw_cds)

# 插补数据降维与轨迹构建
imputed_cds <- preprocess_cds(imputed_cds)
imputed_cds <- reduce_dimension(imputed_cds, preprocess_method = "PCA")
imputed_cds <- cluster_cells(imputed_cds)
imputed_cds <- learn_graph(imputed_cds)

# 原始数据设置根细胞
raw_cds <- order_cells(raw_cds, root_cells = raw_metadata[raw_metadata$time_point == min(raw_metadata$time_point), "cell_id"])

# 插补数据设置根细胞
imputed_cds <- order_cells(imputed_cds, root_cells = imputed_metadata[imputed_metadata$time_point == min(imputed_metadata$time_point), "cell_id"])

# 将 time_point 添加到 CellDataSet 的元数据中
colData(raw_cds)$time_point <- raw_metadata$time_point
colData(imputed_cds)$time_point <- imputed_metadata$time_point
colData(raw_cds)$dataset <- "Raw"
colData(imputed_cds)$dataset <- "Imputed"
combined_metadata <- rbind(colData(raw_cds), colData(imputed_cds))

dev.new()
# 绘制伪时间分布
ggplot(combined_metadata, aes(x = time_point, fill = dataset)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(x = "Timepoint", y = "Density") + # 只保留 X 和 Y 轴标签
  theme(
    axis.title.x = element_text(size = 14),               # 设置 X 轴标题字体大小
    axis.title.y = element_text(size = 14),               # 设置 Y 轴标题字体大小
    axis.text.x = element_text(size = 12),                # 设置 X 轴刻度字体大小
    axis.text.y = element_text(size = 12),                # 设置 Y 轴刻度字体大小
    legend.title = element_text(size = 14),               # 设置图例标题字体大小
    legend.text = element_text(size = 12)                 # 设置图例文本字体大小
  )

# 原始数据轨迹图
dev.new()
plot_cells(
  raw_cds,
  color_cells_by = "time_point", # 按时间点着色
  label_groups_by_cluster = FALSE,
  label_leaves = FALSE,
  label_branch_points = TRUE,
  show_trajectory_graph = TRUE,
  graph_label_size = 5,                   # 调整轨迹图标签字体大小
  cell_size = 1.5 
)

# 插补数据轨迹图
dev.new()
plot_cells(
  imputed_cds,
  color_cells_by = "time_point", # 按时间点着色
  label_groups_by_cluster = FALSE,
  label_leaves = FALSE,
  label_branch_points = FALSE,
  show_trajectory_graph = TRUE,
  graph_label_size = 5,                   # 调整轨迹图标签字体大小
  cell_size = 1.5 
)

# 差异表达分析
raw_graph_test <- graph_test(raw_cds, neighbor_graph = "principal_graph", cores = 4)
raw_top_genes <- rownames(subset(raw_graph_test, q_value < 0.05))

imputed_graph_test <- graph_test(imputed_cds, neighbor_graph = "principal_graph", cores = 4)
imputed_top_genes <- rownames(subset(imputed_graph_test, q_value < 0.05))

# 比较差异基因
shared_genes <- intersect(raw_top_genes, imputed_top_genes)

# 定义感兴趣的基因
# genes_of_interest <- c("A1CF","A1BG","A2ML1","AAAS","AACS","AADAC")
# genes_of_interest <- c("CD3D","CD14","MS4A1","FCER1A","LYZ","PPBP")
genes_of_interest <- c("FCER1A")
# genes_of_interest <- c("AADAC")
# 提取原始数据的基因表达量与时间点
raw_expr <- t(as.matrix(exprs(raw_cds[genes_of_interest, ]))) # 转置表达矩阵
raw_expr <- as.data.frame(raw_expr)
raw_expr$time_point <- raw_metadata$time_point # 使用时间点
raw_expr$dataset <- "Raw" # 添加数据集标识

# 提取插补数据的基因表达量与时间点
imputed_expr <- t(as.matrix(exprs(imputed_cds[genes_of_interest, ]))) # 转置表达矩阵
imputed_expr <- as.data.frame(imputed_expr)
imputed_expr$time_point <- imputed_metadata$time_point # 使用时间点
imputed_expr$dataset <- "Imputed" # 添加数据集标识

# 合并数据
combined_expr <- rbind(raw_expr, imputed_expr)

# 转换为长格式，便于 ggplot2 绘图
combined_expr_long <- pivot_longer(
  combined_expr,
  cols = genes_of_interest,
  names_to = "gene",
  values_to = "expression"
)

# 绘制时序变化曲线
ggplot(combined_expr_long, aes(x = time_point, y = expression, color = dataset)) +
  geom_line(stat = "smooth", method = "loess", span = 0.5, se = FALSE) + # 平滑曲线
  facet_wrap(~ gene, scales = "free_y", ncol = 3) +                     # 分面绘制每个基因
  theme_minimal() +
  labs(
    title = "Temporal Changes of Marker Genes",
    x = "Time Point",
    y = "Expression",
    color = "Dataset"
  ) +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )

# 基因 GO 富集分析
# 将基因符号转换为 Entrez ID
gene_entrez_ids <- bitr(genes_of_interest, fromType = "SYMBOL", 
                        toType = "ENTREZID", 
                        OrgDb = org.Hs.eg.db)

# 进行 GO 富集分析
go_enrichment <- enrichGO(
  gene          = gene_entrez_ids$ENTREZID,  # 使用 Entrez ID 进行分析
  OrgDb         = org.Hs.eg.db,             # 使用小鼠基因数据库
  keyType       = "ENTREZID",               # 基因 ID 类型
  ont           = "ALL",                    # 分析所有 GO 分支：BP、MF、CC
  pAdjustMethod = "BH",                     # P 值调整方法
  pvalueCutoff  = 0.05,                     # P 值阈值
  qvalueCutoff  = 0.2                       # Q 值阈值
)

# 查看富集结果
head(go_enrichment)
barplot(go_enrichment, showCategory = 4, title = "GO Enrichment Analysis")
dotplot(go_enrichment, showCategory = 4, title = "GO Enrichment Analysis")
emapplot(go_enrichment, showCategory = 10)
go_bp <- gofilter(go_enrichment, ont = "BP") # 仅选取 BP
dotplot(go_bp, showCategory = 10, title = "Biological Process")

# 保存 GO 富集结果到 CSV 文件
write.csv(as.data.frame(go_enrichment), "go_enrichment_results.csv")
