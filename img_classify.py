import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from model.img_model import Net_R
import torch.nn.functional as F

batch_size = 8
device = utils.get_device()
epochs = 200


def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()
    clss_n = torch.zeros((13)).to(device)
    clss_m = torch.zeros((13)).to(device)
    total_loss, total_correct_1, total_clss, total_num, subset, data_bar = 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = net(data)
            loss = loss_criterion(out, target)

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


train_set = ..

test_set = ..

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=1)

model = Net_R()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
net = nn.DataParallel(model)
net.to(device)

loss_criterion = F.binary_cross_entropy_with_logits
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
    data_frame.to_csv('/path/to/save/image/model.csv', index_label='epoch')
    torch.save(model.state_dict(), '/path/to/image/model.pth')
