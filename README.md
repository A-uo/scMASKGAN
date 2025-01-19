# scMASKGAN
scMASKGAN A Novel GAN based Approach for scRNA-seq Data Imputation
The specific installation steps are as follows：

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install numpy
conda install pandas
conda install scikit-learn
```

Runing：

```python
python scMASKGAN.py --train --n_epochs 500 --batch_size 64 --file_d ./Data/GSM5768752_NB5_UMI_COUNTS_RAW.csv --ncls 10 --knn_k 10 --img_size 200 --latent_dim 200 --file_c louvain_labels_GSM5768752.csv --job_name GSM5768752 --outdir ./output
python scMASKGAN.py --impute --n_epochs 500 --batch_size 64 --file_d ./Data/GSM5768747_NB5_UMI_COUNTS_RAW.csv --ncls 10 --knn_k 10 --img_size 200 --latent_dim 200 --file_c louvain_labels_GSM5768752.csv --job_name GSM5768752 --outdir ./output
```

The sample data files are saved in Data.zip. You can run them by unzipping them in this directory. The other two .py files can also be run according to the above code. The "xxx_labels.csv" file represents the clustering label file of the original data.
ncls (number of classes) and knn_k (the k-value for KNN) need to be adjusted based on the number of labels.
img_size and latent_dim need to be adjusted based on the number of genes m, with the condition that m < \text{img_size}^2.
