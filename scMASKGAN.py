import argparse
import os
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.autograd import Variable
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsRegressor
import torch.nn.functional as F
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=170, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=170, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--dpt', type=str, default='', help='load discriminator model')
parser.add_argument('--gpt', type=str, default='', help='load generator model')
parser.add_argument('--train', help='train the network', action='store_true')
parser.add_argument('--impute', help='do imputation', action='store_true')
parser.add_argument('--sim_size', type=int, default=200, help='number of sim_imgs in each type')
parser.add_argument('--file_d', type=str, default='', help='path of data file')
parser.add_argument('--file_c', type=str, default='', help='path of cls file')
parser.add_argument('--ncls', type=int, default=9, help='number of clusters')
parser.add_argument('--knn_k', type=int, default=9, help='neighours used')
parser.add_argument('--lr_rate', type=int, default=9, help='rate for slow learning')
parser.add_argument('--threshold', type=float, default=0.01, help='the convergence threshold')
parser.add_argument('--job_name', type=str, default="", help='the user-defined job name, which will be used to name the output files.')
parser.add_argument('--outdir', type=str, default=".", help='the directory for output.')

opt = parser.parse_args()
max_ncls = opt.ncls

job_name = opt.job_name
GANs_models = opt.outdir + '/GANs_models'
if job_name == "":
    job_name = os.path.basename(opt.file_d) + "-" + os.path.basename(opt.file_c)
model_basename = job_name + "-" + str(opt.latent_dim) + "-" + str(opt.n_epochs) + "-" + str(opt.ncls)
if not os.path.isdir(GANs_models):
    os.makedirs(GANs_models)

img_shape = (opt.channels, opt.img_size, opt.img_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False

class MyDataset(Dataset):
    def __init__(self, d_file, cls_file, transform=None):
        self.data = pd.read_csv(d_file, header=0, index_col=0, delimiter="\t")
        # self.data = pd.read_csv(d_file, header=0, index_col=0)
        d = pd.read_csv(cls_file, header=None, index_col=False)
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes
        self.transform = transform
        self.fig_h = opt.img_size
        self.required_length = self.fig_h * self.fig_h  # 需要的长度

        # 记录被填充的样本索引（可选）
        self.padded_indices = []

        # 调试信息
        print(f"Data shape: {self.data.shape}")
        print(f"Data class length: {len(self.data_cls)}")

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        if idx >= self.data.shape[1]:
            raise IndexError("Index out of bounds for data columns.")

        # 获取第 idx 列的数据
        column_data = self.data.iloc[:, idx].values

        # 检查数据长度是否满足要求
        current_length = len(column_data)
        if current_length < self.required_length:
            # 需要填充
            pad_length = self.required_length - current_length
            padded_data = np.pad(column_data, (0, pad_length), 'constant', constant_values=0)
            mask = np.concatenate([np.ones(current_length), np.zeros(pad_length)])
            self.padded_indices.append(idx)  # 记录被填充的索引（可选）
        else:
            # 数据长度足够，截断多余的部分
            padded_data = column_data[:self.required_length]
            mask = np.ones(self.required_length)

        # 重塑为 (img_size, img_size, 1) 的数组
        data = padded_data.reshape(self.fig_h, self.fig_h, 1).astype('double')

        # 生成掩码，重塑为 (img_size, img_size, 1)
        mask = mask.reshape(self.fig_h, self.fig_h, 1).astype('float32')

        # 获取标签
        label = np.array(self.data_cls[idx]).astype('int32')

        sample = {'data': data, 'label': label, 'mask': mask}

        # 应用转换（如 ToTensor）
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        data, label, mask = sample['data'], sample['label'], sample['mask']
        # 将数据和掩码从 (H, W, C) 转置为 (C, H, W)
        data = data.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {
            'data': torch.from_numpy(data).float(),
            'label': torch.from_numpy(np.array(label)).long(),
            'mask': torch.from_numpy(mask).float()
        }


def one_hot(batch, depth):
    ones = torch.eye(depth)
    batch = batch.clamp(0, depth - 1)  # Ensure all batch values are within valid range
    return ones.index_select(0, batch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('SelfAttention') != -1:
        torch.nn.init.normal_(m.query_conv.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.key_conv.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.value_conv.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.gamma.data, 0.0)

class SelfAttention(nn.Module):
    def __init__(self, in_dim, block_size=16):
        super(SelfAttention, self).__init__()
        self.block_size = block_size
        self.chanel_in = in_dim

        # 生成 Query, Key, Value
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # 缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  # 对最后一个维度进行 softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N= W*H)
        """
        B, C, W, H = x.size()
        block = self.block_size

        # 计算需要填充的宽度和高度
        pad_w = (block - W % block) % block
        pad_h = (block - H % block) % block

        # 如果需要填充，执行填充
        if pad_w != 0 or pad_h != 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            x_padded = x

        W_padded, H_padded = x_padded.size(2), x_padded.size(3)

        # 创建掩码：标记有效像素（1）和填充像素（0）
        # 假设填充区域为0，且有效像素和填充像素的值不同
        mask = (x_padded.abs().sum(dim=1, keepdim=True) > 0).float()  # (B,1,W_padded,H_padded)

        # 划分块
        mask_blocks = mask.view(B, 1, W_padded // block, block, H_padded // block, block)
        mask_blocks = mask_blocks.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, 1, block * block)  # (B*num_blocks,1,N)

        x_reshaped = x_padded.view(B, C, W_padded // block, block, H_padded // block, block)
        x_reshaped = x_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, block, block)  # (B*num_blocks, C, block, block)

        # 生成 Query, Key, Value
        proj_query = self.query_conv(x_reshaped).view(-1, C // 8, block * block).permute(0, 2, 1)  # (B*num_blocks, N, C')
        proj_key = self.key_conv(x_reshaped).view(-1, C // 8, block * block)  # (B*num_blocks, C', N)
        energy = torch.bmm(proj_query, proj_key)  # (B*num_blocks, N, N)

        # 应用掩码：将填充像素的注意力分数设置为 -inf
        mask_keys = mask_blocks.squeeze(1).unsqueeze(1)  # (B*num_blocks,1,N)
        energy = energy.masked_fill(mask_keys == 0, -1e9)

        # 可选：如果查询也是填充像素，可以进一步掩码
        mask_queries = mask_blocks.squeeze(1).unsqueeze(2)  # (B*num_blocks,N,1)
        energy = energy.masked_fill(mask_queries == 0, -1e9)

        # 应用 softmax
        attention = self.softmax(energy)  # (B*num_blocks,N,N)

        # 生成 Value
        proj_value = self.value_conv(x_reshaped).view(-1, C, block * block)  # (B*num_blocks, C, N)

        # 计算注意力输出
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B*num_blocks, C, N)
        out = out.view(-1, C, block, block)  # (B*num_blocks, C, block, block)

        # 重组块
        out = out.view(B, W_padded // block, H_padded // block, C, block, block)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, W_padded, H_padded)  # (B, C, W_padded, H_padded)

        # 如果进行了填充，裁剪回原始尺寸
        if pad_w != 0 or pad_h != 0:
            out = out[:, :, :W, :H]

        # 残差连接
        out = self.gamma * out + x

        return out, attention



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.cn1 = 32
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, self.cn1 * (self.init_size ** 2)),
            nn.BatchNorm1d(self.cn1 * (self.init_size ** 2)),
            nn.ReLU(True)
        )

        self.l1p = nn.Sequential(
            nn.Linear(opt.latent_dim, self.cn1 * (opt.img_size ** 2)),
            nn.BatchNorm1d(self.cn1 * (opt.img_size ** 2)),
            nn.ReLU(True)
        )

        #卷积块1：针对所有通道进行卷积操作
        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )
        self.attn1 = SelfAttention(self.cn1)
        #卷积块2：上采样层，将特征尺寸放大scale_factor倍，使其与图像的尺寸匹配。
        self.conv_blocks_02p = nn.Sequential(
            nn.Upsample(scale_factor=opt.img_size),
            #将labels数映射到卷积通道中
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noise, label_oh):
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, opt.img_size, opt.img_size)
        out01 = self.conv_blocks_01p(out)
        out01, attn1 = self.attn1(out01)
        label_oh = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks_02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return out1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32
        self.pre = nn.Sequential(nn.Linear(opt.img_size ** 2, self.down_size0 ** 2))
        #下采样卷积块
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            #将特征图尺寸缩小一半
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            ResidualBlock(self.cn1, self.cn1),
            nn.Conv2d(self.cn1, self.cn1 // 2, 3, 1, 1),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(),
        )
        #标签处理卷积快
        self.conv_blocks02p = nn.Sequential(
            nn.Upsample(scale_factor=self.down_size),
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )
        self.attn1 = SelfAttention(self.cn1 // 2)
        down_dim = 24 * (self.down_size) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, label_oh):
        out00 = self.pre(img.view((img.size()[0], -1))).view((img.size()[0], 1, self.down_size0, self.down_size0))
        out01 = self.down(out00)
        out01, attn1 = self.attn1(out01)
        label_oh = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 24, self.down_size, self.down_size))
        return out

def my_dbscan_impute(data_imp_org_k, sim_out_k, eps=0.5, min_samples=10):
    sim_size = sim_out_k.shape[0]
    out = data_imp_org_k.copy()
    q1k = data_imp_org_k.reshape((opt.img_size * opt.img_size, 1))
    q1kl = np.int8(q1k > 0)
    q1kn = np.repeat(q1k * q1kl, repeats=sim_size, axis=1)
    sim_out_tmp = sim_out_k.reshape((sim_size, opt.img_size * opt.img_size)).T
    sim_outn = sim_out_tmp * np.repeat(q1kl, repeats=sim_size, axis=1)
    diff = q1kn - sim_outn
    diff = diff * diff
    rel = np.sum(diff, axis=0)
    locs = np.where(q1kl == 0)[0]
    sim_out_c = np.median(sim_out_tmp[:, rel.argsort()[0:min_samples]], axis=1)
    out[locs] = sim_out_c[locs]
    return out

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

transformed_dataset = MyDataset(d_file=opt.file_d, cls_file=opt.file_c, transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

gamma = opt.gamma
lambda_k = 0.001
k = opt.kt

import matplotlib.pyplot as plt


def save_fake_images(images, epoch, batch_idx):
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    np_grid = grid.cpu().numpy()
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    plt.title(f'Epoch {epoch}, Batch {batch_idx}')
    plt.axis('off')
    plt.savefig(f'fake_images_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

if opt.train:
    model_g_path = os.path.join(GANs_models, f"{model_basename}-g.pt")
    model_d_path = os.path.join(GANs_models, f"{model_basename}-d.pt")
    model_exists = os.path.isfile(model_g_path) and os.path.isfile(model_d_path)
    if model_exists:
        overwrite = input("WARNING: A trained model exists with the same settings for your data.\nDo you want to train and overwrite it?: (y/n)\n")
        if overwrite.lower() != "y":
            print("The training was deprecated since an existing model exists.")
            print("scIGANs continues imputation using the existing model...")
            sys.exit()
    print(f"The optimal model will be output in \"{os.getcwd()}/{GANs_models}\" with basename = {model_basename}")

    G_losses = []
    D_losses = []

    max_M = sys.float_info.max
    min_dM = 0.001
    dM = 1
    for epoch in range(opt.n_epochs):
        cur_M = 0
        cur_dM = 1
        epoch_G_loss = 0  # 当前epoch的生成器损失总和
        epoch_D_loss = 0  # 当前epoch的判别器损失总和

        for i, batch_sample in enumerate(dataloader):
            imgs = batch_sample['data'].type(Tensor).to(device)
            label = batch_sample['label'].to(device)
            mask = batch_sample['mask'].type(Tensor).to(device)
            label_oh = one_hot(label.type(torch.LongTensor), max_ncls).type(Tensor).to(device)

            real_imgs = Variable(imgs.type(Tensor))

            # 训练生成器
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
            gen_imgs = generator(z, label_oh)
            # 应用掩码
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs, label_oh) - gen_imgs) * mask)
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()
            d_real = discriminator(real_imgs, label_oh)
            d_fake = discriminator(gen_imgs.detach(), label_oh)
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs) * mask)
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()) * mask)
            d_loss = d_loss_real - k * d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # 更新 k
            diff = torch.mean(gamma * d_loss_real - d_loss_fake)
            k = k + lambda_k * diff.detach().cpu().numpy()
            k = min(max(k, 0), 1)
            M = (d_loss_real + torch.abs(diff)).item()
            cur_M += M

            # 累加当前batch的损失
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()

            # 输出训练状态
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f, k: %f" % (
                epoch + 1, opt.n_epochs, i + 1, len(dataloader),
                d_loss.item(), g_loss.item(), M, dM, k))
            sys.stdout.flush()

        # 计算当前 epoch 的平均 M
        cur_M = cur_M / len(dataloader)
        # 记录当前epoch的平均损失
        avg_G_loss = epoch_G_loss / len(dataloader)
        avg_D_loss = epoch_D_loss / len(dataloader)
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)

        if cur_M < max_M:
            # 保存模型
            torch.save(discriminator.state_dict(), model_d_path)
            torch.save(generator.state_dict(), model_g_path)
            dM = min(max_M - cur_M, cur_M)
            if dM < min_dM:
                print(f"\nTraining was stopped after {epoch + 1} epochs since the convergence threshold ({min_dM}) was reached: {dM}")
                break
            cur_dM = max_M - cur_M
            max_M = cur_M
        if epoch + 1 == opt.n_epochs and cur_dM > min_dM:
            print(f"\nTraining was stopped after {epoch + 1} epochs since the maximum epochs reached: {opt.n_epochs}.")
            print(f"WARNING: the convergence threshold ({min_dM}) was not met. Current value is: {cur_dM}")
            print("You may need more epochs to get the most optimal model!!!")

    # 绘制并保存损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(G_losses)+1), G_losses, label="Generator Loss")
    plt.plot(range(1, len(D_losses)+1), D_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss During Training")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(os.getcwd(), GANs_models, f"{model_basename}_loss_curve.pdf")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curves saved to {loss_plot_path}")

if opt.impute:
    if opt.gpt == '':
        model_g = os.path.join(GANs_models, f"{model_basename}-g.pt")
        model_exists = os.path.isfile(model_g)
        if not model_exists:
            print("ERROR: There is no model exists with the given settings for your data.")
            print("Please set --train instead of --impute to train a model first.")
            sys.exit("scIGANs stopped!!!")
    else:
        model_g = opt.gpt
    print(f"{model_g} is used for imputation.")
    if cuda:
        generator.load_state_dict(torch.load(model_g))
    else:
        generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))

    sim_size = opt.sim_size
    sim_out = []
    for i in range(opt.ncls):
        label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ncls).type(Tensor).to(
            device)
        z = Variable(Tensor(np.random.normal(0, 1, (sim_size, opt.latent_dim)))).to(device)
        fake_imgs = generator(z, label_oh).detach().cpu().numpy()
        sim_out.append(fake_imgs)

    mydataset = MyDataset(d_file=opt.file_d, cls_file=opt.file_c)
    data_imp_org = np.asarray(
        [mydataset[i]['data'].reshape((opt.img_size * opt.img_size)) for i in range(len(mydataset))])

    # 获取掩码
    masks = np.asarray(
        [mydataset[i]['mask'].reshape((opt.img_size * opt.img_size)) for i in range(len(mydataset))])

    # 调试：打印数据长度
    print(f"Data_imp_org shape: {data_imp_org.shape}")
    print(f"Mydataset length: {len(mydataset)}")

    data_imp = data_imp_org.copy()
    sim_out_org = sim_out
    rels = []

    # 使用孤立森林检测并移除异常值
    isolation_forest = IsolationForest(contamination=0.25)
    for k in range(len(mydataset)):
        label = int(mydataset[k]['label'])
        if label >= len(sim_out_org):
            raise IndexError(f"Label {label} is out of bounds for sim_out_org with length {len(sim_out_org)}")
        sample_data = data_imp_org[k,:].reshape(1, -1)
        outlier_pred = isolation_forest.fit_predict(sample_data)
        if outlier_pred[0] == -1:
            data_imp[k, :] = np.nan

    for k in range(len(mydataset)):
        label = int(mydataset[k]['label'])
        if label >= len(sim_out_org):
            raise IndexError(f"Label {label} is out of bounds for sim_out_org with length {len(sim_out_org)}")
        rel = my_dbscan_impute(data_imp_org[k, :], sim_out_org[label], eps=0.5, min_samples=opt.knn_k)
        # 应用掩码：只填补原始数据缺失的部分
        rel = rel * (masks[k, :])
        rels.append(rel)

    output_path = os.path.join(os.path.dirname(os.path.abspath(opt.file_d)), f'scIGANs-{job_name}.csv')
    pd.DataFrame(rels).to_csv(output_path,index=False)
    print(f"Imputed data saved to {output_path}")
    imputed_df = pd.read_csv(output_path)
    # 检查并删除全零行
    imputed_df = imputed_df[~(imputed_df == 0).all(axis=1)].T
    # 保存清理后的数据
    imputed_df.to_csv(output_path, index=False)
    print(f"Cleaned imputed data saved to {output_path}")

    # 释放内存
    del generator, discriminator, sim_out, sim_out_org, data_imp_org, data_imp, masks, rels, mydataset, isolation_forest
    torch.cuda.empty_cache()

