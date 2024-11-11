import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import resnet50

class Net_R(nn.Module):

    def __init__(self):
        super(Net_R, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 13),
            nn.BatchNorm1d(13),
            nn.ReLU(True),
        )

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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    '''
    定义普通残差模块
    resnet34为普通残差块，resnet50为瓶颈结构
    '''
    def __init__(self, inchannel, outchannel, stride=1, padding=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        #resblock的首层，首层如果跨维度，卷积stride=2，shortcut需要1*1卷积扩维
        if inchannel != outchannel:
            stride= 2
            shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )

        # 定义残差块的左部分
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, 3, 1, padding, bias=False),
            nn.BatchNorm2d(outchannel),

        )

        #定义右部分
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return F.relu(out)

class BottleNeckBlock(nn.Module):
    '''
    定义resnet50的瓶颈结构
    '''
    def __init__(self,inchannel,outchannel, pre_channel=None, stride=1,shortcut=None):
        super(BottleNeckBlock, self).__init__()
        #首个bottleneck需要承接上一批blocks的输出channel
        if pre_channel is None:     #为空则表示不是首个bottleneck，
            pre_channel = outchannel    #正常通道转换


        else:   # 传递了pre_channel,表示为首个block，需要shortcut
            shortcut = nn.Sequential(
                nn.Conv2d(pre_channel,outchannel,1,stride,0,bias=False),
                nn.BatchNorm2d(outchannel)
            )

        self.left = nn.Sequential(
            #1*1,inchannel
            nn.Conv2d(pre_channel, inchannel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            #3*3,inchannel
            nn.Conv2d(inchannel,inchannel,3,1,1,bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            #1*1,outchannel
            nn.Conv2d(inchannel,outchannel,1,1,0,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        self.right = shortcut

    def forward(self,x):
        
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        return F.relu(out+residual)

class ClassModel(nn.Module):
    """
    实现通用的ResNet模块，可根据需要定义
    """
    def __init__(self, layer_num=[2,2,2,2],bottleneck = False):
        super(ClassModel, self).__init__()

        #conv1
        self.pre = nn.Sequential(
            #in 224*224*3
            nn.Conv2d(3,64,7,2,3,bias=False),   #输入通道3，输出通道64，卷积核7*7*64，步长2,根据以上计算出padding=3
            #out 112*112*64
            nn.BatchNorm2d(64),     #输入通道C = 64

            nn.ReLU(inplace=True),   #inplace=True, 进行覆盖操作
            # out 112*112*64
            nn.MaxPool2d(3,2,1),    #池化核3*3，步长2,计算得出padding=1;
            # out 56*56*64
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 13),
            nn.BatchNorm1d(13),
            nn.ReLU(True),
        )

        if bottleneck:  #resnet50以上使用BottleNeckBlock
            self.residualBlocks1 = self.add_layers(64, 256, layer_num[0], 64, bottleneck=bottleneck)
            self.residualBlocks2 = self.add_layers(128, 512, layer_num[1], 256, 2,bottleneck)
            self.residualBlocks3 = self.add_layers(256, 1024, layer_num[2], 512, 2,bottleneck)
            self.residualBlocks4 = self.add_layers(512, 2048, layer_num[3], 1024, 2,bottleneck)

        else:   #resnet34使用普通ResidualBlock
            self.residualBlocks1 = self.add_layers(64,64,layer_num[0])
            self.residualBlocks2 = self.add_layers(64,128,layer_num[1])
            self.residualBlocks3 = self.add_layers(128,256,layer_num[2])
            self.residualBlocks4 = self.add_layers(256,128,layer_num[3])

    def add_layers(self, inchannel, outchannel, nums, pre_channel=64, stride=1, bottleneck=False):
        layers = []
        if bottleneck is False:

            #添加大模块首层, 首层需要判断inchannel == outchannel ?
            #跨维度需要stride=2，shortcut也需要1*1卷积扩维

            layers.append(ResidualBlock(inchannel,outchannel))

            #添加剩余nums-1层
            for i in range(1,nums):

                layers.append(ResidualBlock(outchannel,outchannel))
            return nn.Sequential(*layers)
        else:   #resnet50使用bottleneck
            #传递每个block的shortcut，shortcut可以根据是否传递pre_channel进行推断

            #添加首层,首层需要传递上一批blocks的channel
            layers.append(BottleNeckBlock(inchannel,outchannel,pre_channel,stride))
            for i in range(1,nums): #添加n-1个剩余blocks，正常通道转换，不传递pre_channel
                layers.append(BottleNeckBlock(inchannel,outchannel))
            return nn.Sequential(*layers)

    def forward(self, x):

        x = self.pre(x)

        x = self.residualBlocks1(x)

        x = self.residualBlocks2(x)

        x = self.residualBlocks3(x)

        x = self.residualBlocks4(x)

        x = self.avgpool(x)
        #print(x.size())

        x = x.view(x.size(0), -1)
        x = x
        #print(x.size())

        return self.classifier(x)
