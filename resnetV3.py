import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,inchannel ,outchannel,stride = 1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                      padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1,
                  padding=1, bias=False),
            nn.BatchNorm2d(outchannel))


        self.shoetcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shoetcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )


    def forward(self,x):
        out1 = self.left(x)
        out2 = self.shoetcut(x)
        out = out1+out2
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_class=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock,64,2,stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512,num_class)




    def make_layer(self,block,channel,num_blocks,stride):
        strides = [stride]+[1]*(num_blocks-1)
        layer=[]
        for s in strides:
            layer.append(block(self.inchannel,channel,s))
            self.inchannel = channel
        return nn.Sequential(*layer)


    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = F.log_softmax(out,dim=1)
        return out



def Resnet18():
    return ResNet()
