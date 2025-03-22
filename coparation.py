import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, recall_score, \
    roc_auc_score, roc_curve, auc
import scanpy as sc
from scipy import sparse


# Preprocessing function
def preprocess(adata):
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # adata.var['mt'] = adata.var_names.str.startswith('MT-')
    # sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    # adata = adata[adata.obs['pct_counts_mt'] < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat")
    # adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    return adata


# Load CSV and create AnnData object
def load_csv_to_adata(file_path):
    df = pd.read_csv(file_path,
                     index_col=0)  # Assuming the first column contains gene names and the first row contains cell names
    df = df.T  # Transpose the dataframe to match Scanpy's expected format
    return sc.AnnData(df)


# Calculate mean gene expression
def calculate_gene_mean_expression(adata):
    if sparse.issparse(adata.X):
        mean_expression = adata.X.mean(axis=0).A1  # Convert matrix to 1D array
    else:
        mean_expression = adata.X.mean(axis=0)
    return pd.Series(mean_expression, index=adata.var_names)


# Define function to calculate evaluation metrics
def calculate_metrics(original_gene_mean, interpolated_gene_mean):
    print(f"Processing dataset: {interpolated_data_path}")
    print(f"Original genes: {len(adata_original.var_names)}, Interpolated genes: {len(adata_interpolated.var_names)}")

    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(original_gene_mean, interpolated_gene_mean)
    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(original_gene_mean, interpolated_gene_mean)
    # Mean squared error (MSE)
    mse = mean_squared_error(original_gene_mean, interpolated_gene_mean)
    # Mean absolute error (MAE)
    mae = mean_absolute_error(original_gene_mean, interpolated_gene_mean)
    # R² score
    r2 = r2_score(original_gene_mean, interpolated_gene_mean)

    return pearson_corr, pearson_pval, spearman_corr, spearman_pval, mse, mae, r2


# Calculate classification metrics
def calculate_classification_metrics(original_gene_mean, interpolated_gene_mean):
    # Use median as threshold
    threshold = np.median(original_gene_mean)

    binary_true_labels = (original_gene_mean > threshold).astype(int)
    binary_pred_labels = (interpolated_gene_mean > threshold).astype(int)

    accuracy = accuracy_score(binary_true_labels, binary_pred_labels)
    f1 = f1_score(binary_true_labels, binary_pred_labels)
    recall = recall_score(binary_true_labels, binary_pred_labels)
    auc_score = roc_auc_score(binary_true_labels, interpolated_gene_mean)

    return accuracy, f1, recall, auc_score
#dataset_names = ['AutoImpute', 'DCA', 'DeepImpute',"MAGIC","scImpute","scGAIN","scIGANs","scMASKGAN"]
dataset_names = ['AutoImpute', 'DCA', 'DeepImpute',"DrImpute","ENHANCE","MAGIC","SAVER","scImpute","SCRABBLE","VIPER","scGAIN","scIGANs","scNZ"]
# mERC
#dataset_names = ["scIGANs","scMASKGAN"]
# Plot gene expression comparison
def plot_gene_expression_comparison(original_gene_mean, interpolated_gene_mean, dataset_num):
    plt.figure(figsize=(4, 4))
    plt.scatter(original_gene_mean, interpolated_gene_mean, alpha=0.5, edgecolor='k',s=5)
    plt.plot([min(original_gene_mean.min(), interpolated_gene_mean.min()),
              max(original_gene_mean.max(), interpolated_gene_mean.max())],
             [min(original_gene_mean.min(), interpolated_gene_mean.min()),
              max(original_gene_mean.max(), interpolated_gene_mean.max())],
             'r--', label='Diagonal')
    plt.xlabel('Observed (log1p)', fontsize=22)
    plt.ylabel(f'Imputed (log1p)', fontsize=22)
    plt.title(f'{dataset_num}', fontsize=22)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend()
    #plt.grid(True)
    plt.savefig(f'huitu/{dataset_num}.png',dpi=300,bbox_inches='tight')
    plt.tight_layout()
    plt.show()




# Store evaluation metrics
metrics_dict = {
    'Dataset': [],
    'Pearson Correlation': [],
    'Pearson p-value': [],
    'Spearman Correlation': [],
    'Spearman p-value': [],
    'Mean Squared Error (MSE)': [],
    'Mean Absolute Error (MAE)': [],
    'R² Score': [],
    'Accuracy': [],
    'F1 Score': [],
    'Recall': [],
    'AUC': []
}
#
# # Assuming file paths for the original and 5 interpolated datasets
# # #1.brain
original_data_path = 'Humanbrain_csv/Brainraw.csv'
interpolated_data_paths = ['Humanbrain_csv/Brain_AutoImpute_impute.csv',
                           'Humanbrain_csv/Brain_DCA_impute.csv',
                           'Humanbrain_csv/Brain_DeepImpute_impute.csv',
                           'Humanbrain_csv/Brain_DrImpute_impute.csv',
                           'Humanbrain_csv/Brain_ENHANCE_impute.csv',
                           'Humanbrain_csv/Brain_MAGIC_impute.csv',
                           'Humanbrain_csv/Brain_SAVER_impute.csv',
                           'Humanbrain_csv/Brain_scGAIN_impute.csv',
                           'Humanbrain_csv/Brain_scImpute_impute.csv',
                           'Humanbrain_csv/Brain_SCRABBLE_impute.csv',
                           'Humanbrain_csv/Brain_VIPER_impute.csv',
                           'Humanbrain_csv/Brain_scIGANs_impute_label.csv',
                           "Humanbrain_csv/Brainraw_scNZ.csv"]
# #                          'mESC_csv/mESC_Autoimpute_impute.csv',
#                            'mESC_csv/mESC_DCA_impute.csv',
#                            'mESC_csv/mESC_Deepimpute_impute.csv',
#                            'mESC_csv/mESC_MAGIC_impute.csv',
#                            'mESC_csv/mESC_scImpute_impute.csv',
# original_data_path = 'mESC_csv/mESC.raw.csv'
# interpolated_data_paths = ['mESC_csv/mESC_Autoimpute_impute.csv',
#                            'mESC_csv/mESC_DCA_impute.csv',
#                            'mESC_csv/mESC_Deepimpute_impute.csv',
#                            'mESC_csv/mESC_MAGIC_impute.csv',
#                            'mESC_csv/mESC_scImpute_impute.csv',
#                            'mESC_csv/scGAIN_impute.csv',
#                            'mESC_csv/scIGANs_impute_label.csv',
#                            'mESC_csv/scMASKGAN-mESCraw.csv']
# original_data_path = 'timecourse_csv/Timecourse.raw.csv'
# interpolated_data_paths = ['timecourse_csv/Timecourse_Autoimpute_impute.csv',
#                            'timecourse_csv/Timecourse_DCA_impute.csv',
#                            'timecourse_csv/Timecourse_Deepimpute_impute.csv',
#                            'timecourse_csv/Timecourse_MAGIC_impute.csv',
#                            'timecourse_csv/Timecourse_scImpute_count.csv',
#                            'timecourse_csv/Timecourse_scGAIN_impute.csv',
#                            'timecourse_csv/Timecourse_scIGANs_impute_label.csv',
#                            'timecourse_csv/scMASKGAN-timecourse.csv']
# original_data_path = 'sc10x_csv/sc_10x.raw.csv'
# interpolated_data_paths = ['sc10x_csv/scIGANs_sc_10x.csv',
#                            'sc10x_csv/scMASKGAN-sc10x.csv']
#
# original_data_path = 'dropseq_csv/sc_dropseq.raw.csv'
# interpolated_data_paths = ['dropseq_csv/scIGANs_sc_dropseq.csv',
#                            'dropseq_csv/scMASKGAN-sc_dropseq.csv']
#
# original_data_path = 'celseq2_csv/sc_celseq2.raw.csv'
# interpolated_data_paths = ['celseq2_csv/scIGANs_sc_celseq2.csv',
#                            'celseq2_csv/scMASKGAN-celseq2.csv']
#
# Load original data
adata_original = load_csv_to_adata(original_data_path)
adata_original = preprocess(adata_original)

# Function to plot UMAP for the datasets with Louvain clustering
# Function to plot UMAP for the datasets with Louvain clustering
# def plot_umap(adata_original, adata_list, dataset_names):
#     # Create subplots (flattening axes to make indexing easier)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()  # Flatten the 2D axes array into a 1D array
#
#     # UMAP for original data
#     #sc.pp.neighbors(adata_original, n_neighbors=10)
#     sc.tl.umap(adata_original)
#     # Perform clustering (Louvain) on original data if not done yet
#     sc.tl.louvain(adata_original)
#     sc.pl.umap(adata_original, color='louvain', ax=axes[0], title='Original Data', show=False)
#
#     # UMAP for interpolated datasets
#     for i, adata in enumerate(adata_list):
#         sc.pp.neighbors(adata, n_neighbors=10)
#         sc.tl.umap(adata)
#         # Perform clustering (Louvain) on interpolated data if not done yet
#         sc.tl.louvain(adata)
#         sc.pl.umap(adata, color='louvain', ax=axes[i + 1], title=dataset_names[i], show=False)
#
#     plt.tight_layout()
#     plt.show()


# Generate UMAP for the original and interpolated datasets
adata_list = []
for interpolated_data_path in interpolated_data_paths:
    adata_interpolated = load_csv_to_adata(interpolated_data_path)
    adata_interpolated = preprocess(adata_interpolated)
    adata_list.append(adata_interpolated)

# Now call the plot_umap function
#plot_umap(adata_original, adata_list, dataset_names)

# For storing AUC values for plotting
auc_values = []

# Plot ROC curves for all datasets on a single plot
def plot_all_roc_curves(binary_true_labels, auc_values, fpr_list, tpr_list,dataset_names):
    plt.figure(figsize=(6, 6))

    # Plot ROC curve for each dataset
    for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
        roc_auc = auc(fpr, tpr)  # Calculate AUC score
        auc_values.append(roc_auc)  # Store AUC for each dataset
        plt.plot(fpr, tpr, label=f'{dataset_names[i]} AUC = {roc_auc:.4f}')

    # Plot diagonal (random classifier line)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')

    # Set plot labels and title
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)
    plt.title('ROC Curves for All Datasets', fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f'AUC.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Store FPR, TPR for all datasets
fpr_list = []
tpr_list = []

# Process each of the 5 interpolated datasets
for i, interpolated_data_path in enumerate(interpolated_data_paths, start=1):
    # Load interpolated data
    adata_interpolated = load_csv_to_adata(interpolated_data_path)
    adata_interpolated = preprocess(adata_interpolated)
    print(f"Processing dataset: {interpolated_data_path}")
    print(f"Original genes: {len(adata_original.var_names)}, Interpolated genes: {len(adata_interpolated.var_names)}")

    # Align genes
    common_genes = adata_original.var_names.intersection(adata_interpolated.var_names)
    print(f"Common genes: {len(common_genes)}")
    adata_original_aligned = adata_original[:, common_genes].copy()
    adata_interpolated_aligned = adata_interpolated[:, common_genes].copy()

    # Calculate mean gene expression
    original_gene_mean = calculate_gene_mean_expression(adata_original_aligned)
    interpolated_gene_mean = calculate_gene_mean_expression(adata_interpolated_aligned)

    # Calculate evaluation metrics
    pearson_corr, pearson_pval, spearman_corr, spearman_pval, mse, mae, r2 = calculate_metrics(
        original_gene_mean, interpolated_gene_mean)

    # Calculate classification metrics
    accuracy, f1, recall, auc_score = calculate_classification_metrics(
        original_gene_mean, interpolated_gene_mean)

    # Store evaluation results in dictionary
    metrics_dict['Dataset'].append(dataset_names[i - 1])  # Use custom names
    metrics_dict['Pearson Correlation'].append(pearson_corr)
    metrics_dict['Pearson p-value'].append(pearson_pval)
    metrics_dict['Spearman Correlation'].append(spearman_corr)
    metrics_dict['Spearman p-value'].append(spearman_pval)
    metrics_dict['Mean Squared Error (MSE)'].append(mse)
    metrics_dict['Mean Absolute Error (MAE)'].append(mae)
    metrics_dict['R² Score'].append(r2)
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['F1 Score'].append(f1)
    metrics_dict['Recall'].append(recall)
    metrics_dict['AUC'].append(auc_score)

    # Calculate ROC curve
    threshold = np.median(original_gene_mean)
    binary_true_labels = (original_gene_mean > threshold).astype(int)
    fpr, tpr, _ = roc_curve(binary_true_labels, interpolated_gene_mean)

    # Append fpr, tpr to lists
    fpr_list.append(fpr)
    tpr_list.append(tpr)

    # Plot gene expression comparison for each dataset
    plot_gene_expression_comparison(original_gene_mean, interpolated_gene_mean, dataset_names[i - 1])

# Ensure all lists in the dictionary have the same length (fill with None if needed)
max_len = len(interpolated_data_paths)
for key in metrics_dict:
    while len(metrics_dict[key]) < max_len:
        metrics_dict[key].append(None)

# Create DataFrame from the metrics dictionary
metrics_df = pd.DataFrame(metrics_dict)

# Save evaluation metrics to CSV
metrics_df.to_csv('ercc.csv', index=False)

#print("Evaluation metrics saved as 'gene_expression_comparison_metrics_50_datasets.csv'.")

# Plot ROC curves for all datasets
plot_all_roc_curves(binary_true_labels, auc_values, fpr_list, tpr_list, dataset_names)
