library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(enrichplot)
library(DOSE)
# 读取原始数据和插补数据
original_data <- read.csv("D:\\TCGA\\TCGAtraining\\GSM5768752_GN1_UMI_COUNTS_RAW.csv", row.names = 1)
imputed_data <- read.csv("D:\\TCGA\\TCGAtraining\\scIGANs-GSM5768752.csv", row.names = 1)

# 确保两组数据具有相同的基因顺序
original_data <- original_data[order(rownames(original_data)), ]
imputed_data <- imputed_data[order(rownames(imputed_data)), ]

# 查看数据维度
print(dim(original_data))
print(dim(imputed_data))

# 定义目标基因列表
genes_to_plot <- c("ALK", "ATRX", "HIF1A", "NTRK1", "NTRK2", "CXCR4")

# 提取目标基因的表达量
original_genes <- original_data[rownames(original_data) %in% genes_to_plot, ]
imputed_genes <- imputed_data[rownames(imputed_data) %in% genes_to_plot, ]

# 查看目标基因表达量
print(original_genes)
print(imputed_genes)
# 设置基因列表
# genes_to_plot <- c("ALK", "ATRX", "HIF1A", "NTRK1", "NTRK2", "CXCR4")
# 计算差异（以基因的平均表达值为例）
original_mean <- rowMeans(original_genes)
imputed_mean <- rowMeans(imputed_genes)

# 差异计算
gene_diff <- data.frame(
  Gene = genes_to_plot,
  Original = original_mean,
  Imputed = imputed_mean,
  Log2FC = log2(imputed_mean + 1) - log2(original_mean + 1)
)

print(gene_diff)

# 将目标基因转换为 ENTREZ ID
genes_entrez <- bitr(genes_to_plot, fromType = "SYMBOL", 
                     toType = "ENTREZID", 
                     OrgDb = org.Hs.eg.db)

# GO 富集分析
go_enrich <- enrichGO(gene = genes_entrez$ENTREZID,
                      OrgDb = org.Hs.eg.db,
                      ont = "BP",  # 生物过程
                      pvalueCutoff = 0.05,
                      qvalueCutoff = 0.05,
                      readable = TRUE)

# KEGG 富集分析
kegg_enrich <- enrichKEGG(gene = genes_entrez$ENTREZID,
                          organism = "hsa", 
                          pvalueCutoff = 0.05)
# 绘制 GO 气泡图
dotplot(go_enrich, showCategory = 10, title = "GO Enrichment Analysis") + 
  theme_minimal()

# 绘制 KEGG 气泡图
dotplot(kegg_enrich, showCategory = 10, title = "KEGG Enrichment Analysis") + 
  theme_minimal()
# 绘制 GO 网络图
cnetplot(go_enrich, showCategory = 5, 
         circular = FALSE, 
         colorEdge = TRUE) + 
  ggtitle("GO Enrichment Network")

# 绘制 KEGG 网络图
cnetplot(kegg_enrich, showCategory = 5, 
         circular = FALSE, 
         colorEdge = TRUE) + 
  ggtitle("KEGG Enrichment Network")
ggsave("GO_Bubble_Plot.png", width = 8, height = 6, dpi = 300)
ggsave("KEGG_Bubble_Plot.png", width = 8, height = 6, dpi = 300)
ggsave("GO_Network_Plot.png", width = 8, height = 6, dpi = 300)
ggsave("KEGG_Network_Plot.png", width = 8, height = 6, dpi = 300)

