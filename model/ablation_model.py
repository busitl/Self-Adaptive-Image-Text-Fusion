import copy

import pandas as pd
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 读取CSV文件
from dataset import ImageDataset

def elementwise_cosine_similarity(A, B):
    # 计算每个元素的范数
    A_norm = torch.norm(A, dim=0, keepdim=True)
    B_norm = torch.norm(B, dim=0, keepdim=True)

    # 避免除零错误，添加一个非常小的值
    epsilon = 1e-8
    A_norm = A_norm + epsilon
    B_norm = B_norm + epsilon

    # 归一化
    A_normalized = A / A_norm
    B_normalized = B / B_norm

    # 逐元素计算余弦相似度
    cosine_sim = A_normalized * B_normalized
    return cosine_sim

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape 回原来的形状
        out = self.fc(context)   # 全连接
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class ConfigTrans(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 13
        self.num_epochs = 100
        self.batch_size = 128
        self.pad_size = 340
        self.learning_rate = 0.001
        self.embed = 340
        self.dim_model = 340
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
config = ConfigTrans()
     
class TransformerRS_200_b2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = x1 + x2
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x1, 1)
        x_r = torch.flatten(x2, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2c(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2c, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)  
        out = self.fc1(out)
        x_d = torch.flatten(x1, 1)
        x_r = torch.flatten(x2, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2cks(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2cks, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        C = torch.matmul(x_d.view(d0, d1 * 7 * 7, 1) , x_r.view(d0, d1 * 7 * 7, 1).transpose(1, 2))
        C1 = torch.sum(C,dim = 1)
        C2 = torch.sum(C,dim = 2)
        C = torch.sigmoid(C1 + C2)
        x_d = x_d * C
        x_r = x_r * C
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcos(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcos, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cos(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cos, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cosw0(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cosw0, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01coswt(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01coswt, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_i(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcosw0(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcosw0, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcoswt(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcoswt, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_i(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckw0(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckw0, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckwt(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckwt, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_i(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cos_2w0(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cos_2w0, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.1)

    def make_layers(self, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = out_i.view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = x_d * C
        x2 = out_i
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cosw2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cosw2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.linear_t = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        #x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_d = self.linear_t(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cosw(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cosw, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        #x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_d = torch.flatten(out_t, 2).view(d0, d1 * 7 * 7, 1)
        x_r = torch.flatten(out_i, 2).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cosw1(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cosw1, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.linear_t = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        #x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_d = self.linear_i(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cos2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cos2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = x_d * C
        x_r = x_r * C
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cos3(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cos3, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = x_d * C
        x_r = x_r
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r * C , 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck01cos4(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck01cos4, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        C = (C >= 0.5).float()
        x_d = x_d
        x_r = x_r * C
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r , 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcos2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcos2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = x_d * C
        x_r = x_r * C
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcos3(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcos3, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = x_d * C
        x_r = x_r 
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r * C, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ckcos4(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ckcos4, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.sigmoid(elementwise_cosine_similarity(x_d,x_r))
        x_d = x_d
        x_r = x_r * C
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d * C, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2cks2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2cks2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = out_t.view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        C = torch.matmul(x_d.view(d0, d1 * 7 * 7, 1) , x_r.view(d0, d1 * 7 * 7, 1).transpose(1, 2))
        C1 = torch.sum(C,dim = 1)
        C2 = torch.sum(C,dim = 2)
        C = torch.sigmoid(C1 + C2)
        x_d = (x_d.view(x_d.size(0), -1) * C).view(d0, d1 * 7 * 7, 1)
        x_r = (x_r.view(x_r.size(0), -1) * C).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck1(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck1, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_i(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.linear_t = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_t(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_b2ck3(nn.Module):
    def __init__(self):
        super(TransformerRS_200_b2ck3, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(6272, 6272)
        self.linear_t = nn.Linear(6272, 6272)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_t(torch.flatten(out_t, 1)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 1)).view(d0, d1 * 7 * 7, 1)
        x2 = x_r
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out
        
class TransformerRS_200_c2ck2(nn.Module):
    def __init__(self):
        super(TransformerRS_200_c2ck2, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.features = self.make_layers()
        
        self.fc1 = nn.Sequential(
            nn.Linear(6272 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(True),
        )
        self.avgpool_t = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool_i = nn.AdaptiveAvgPool2d((7, 7))
        self.conv = nn.Conv2d(1, 128, kernel_size=1, stride=1)
        self.linear_i = nn.Linear(49, 49)
        self.linear_t = nn.Linear(49, 49)
        self.dropout = nn.Dropout(0.1)

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x_i,x_t):
        out_t = self.postion_embedding(x_t)
        for encoder in self.encoders:
            out_t = encoder(out_t)
        #print(out.size())
        out_t = out_t.unsqueeze(1)
        out_t = self.conv(out_t)
        out_t = self.avgpool_t(out_t)
        out_i = self.features(x_i)
        out_i = self.avgpool_i(out_i)
        d0 = out_t.size(0)
        d1 = out_t.size(1)
        d = out_t.size(1) #* out_t.size(0)
        d = d ** 0.5
        x_d = self.linear_t(torch.flatten(out_t, 2)).view(d0, d1 * 7 * 7, 1)
        x_r = self.linear_i(torch.flatten(out_i, 2)).view(d0, d1 * 7 * 7, 1)
        x2 = torch.matmul(x_r / d, x_d.transpose(1, 2))
        x2 = self.dropout(F.softmax(x2, dim=-1))
        x2 = torch.matmul(x2, x_d)
        x1 = torch.matmul(x_d / d, x_r.transpose(1, 2))
        x1 = self.dropout(F.softmax(x1, dim=-1))
        x1 = torch.matmul(x1, x_r)
        x = torch.cat((x1.view(x1.size(0), -1),x2.view(x2.size(0), -1)),dim=-1)
        out = x.view(x.size(0), -1) 
        out = self.fc1(out)
        x_d = torch.flatten(x_d, 1)
        x_r = torch.flatten(x_r, 1)
        return x_r,x_d,out