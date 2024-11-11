import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import utils
import torch.nn.functional as F
from torch.autograd import Variable
from model.multimodal_model import TransformerRS_200_b2ck01cos
from model.txt_model import Transformer_CA
from model.img_model import Net_R
import time
import os

batch_size = 8
device = utils.get_device()
epochs = 200
T = 25
a = 0.99

def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()
    clss_n = torch.zeros((13)).to(device)
    clss_m = torch.zeros((13)).to(device)
    total_loss, total_correct_1, total_clss, total_num, subset, data_bar = 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data_i, data_t ,target in data_bar:
            data_i, data_t ,target = data_i.to(device), data_t.to(device) ,target.to(device)
            xr,xd,out=net(data_i,data_t)
            
            ce_loss = loss_criterion(out, target)
            kl_loss = F.kl_div(F.log_softmax(xr/T, dim=1),F.softmax(xd/T, dim=1))*T*T
            loss = (1-a) * kl_loss + a*ce_loss
            ce_loss = loss_criterion(out, target)
            kl_loss = F.kl_div(F.log_softmax(xd/T, dim=1),F.softmax(xr/T, dim=1))*T*T
            loss += (1-a) * kl_loss + a*ce_loss
            loss = 0.5 * loss
            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += target.size(0)
            total_clss += target.size(0) * target.size(1)
            total_loss += loss.item() * data_i.size(0)
            model_output = torch.sigmoid(out)
            predicted_labels = torch.round(model_output)
            target = target.float()
            total_correct_1 += (predicted_labels == target).sum().item()
            clss_n += (predicted_labels == target).sum(dim=0)
            clss_m += target.size(0)
            subset += torch.sum(torch.all(predicted_labels == target, dim=1)).item()
            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_clss * 100))

    return total_loss / total_num, total_correct_1 / total_clss * 100



loss_criterion = F.binary_cross_entropy_with_logits

train_set = ..

test_set = ..

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=1)
                      
                      
model_path1, model_path2 = '/path/to/text/model.pth', '/path/to/image/model.pth'
model1= Transformer_CA()
model1.load_state_dict(torch.load(model_path1))
model2 = Net_R()
model2.load_state_dict(torch.load(model_path2))
model3 = TransformerRS_200_b2ck01cos()
model3.features = model2.features
model3.postion_embedding = model1.postion_embedding
model3.encoder = model1.encoder
model3.encoders = model1.encoders   
model3.conv = model1.conv
model3.avgpool_i = model2.avgpool
model3.avgpool_t = model1.avgpool
optimizer = optim.Adam(model3.parameters(), lr=1e-4)
net = nn.DataParallel(model3)
net.to(device)

net = model3
net.to(device)

def freeze_params(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any([layer_name in name for layer_name in layers_to_freeze]):
            param.requires_grad = False

freeze_params(model3, ['features', 'postion_embedding','encoder','encoders','conv','avgpool_i','avgpool_t'])

results = {'train_loss': [], 'train_acc@1': [],
           'test_loss': [], 'test_acc@1': []}

for epoch in range(1, epochs + 1):
    train_loss, train_acc_1 = train_val(net, train_loader, optimizer)
    results['train_loss'].append(train_loss)
    results['train_acc@1'].append(train_acc_1)
    test_loss, test_acc_1 = train_val(net, test_loader, None)
    results['test_loss'].append(test_loss)
    results['test_acc@1'].append(test_acc_1)

    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv('/path/to/save/model.csv', index_label='epoch')
    torch.save(model3.state_dict(), '/path/to/save/model.pth')
