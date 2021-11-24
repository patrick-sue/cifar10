from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob
from pathlib import Path

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

# # 用于训练的数据争抢，一般用compose进行拼接操作
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28,28)),
#     transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),
#     # transforms.ColorJitter(0.2,0.2,0.2),
#     # 要转化为tensor回传到数据里面(网络需要是数据是tensor类型的)
#     transforms.ToTensor()
#
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize(28),
#     transforms.ToTensor()
# ])
#
# def default_loader(path):
#     return Image.open(path).convert("RGB")
# class MyDataset(Dataset):
#     # 完成对数据的读取，对数据进行简单的处理放到列表中
#     def __init__(self,im_list, transform =None, loader= default_loader ):
#         # 初始化这个类
#         super(MyDataset, self).__init__()
#         # 定义数据列表
#         imgs= []
#         for im_item in im_list:
#             # "D:\deepfake\练习\练习源码\Pytorch_code-master_免费IT课程加微信2268731\pytorch_code\06\cifar10\TRAIN\airplane\2321322.png"
#             im_label_name = im_item.split(r"/")[-2]
#             imgs.append([im_item,label_dict[im_label_name]])
#
#         # 图片元素
#         self.imgs = imgs
#         #两个方法
#         self.transform = transform
#         self.loader = loader
#
#     # 定义图片的读取和图片的增强，读取图片
#     def __getitem__(self, index):
#         im_path, im_label = self.imgs[index]
#         #用loader读取图片
#         im_data = self.loader(im_path)
#         # 判断是否需要进行数据增强，一般在训练的时候进行数据增强，测试不需要
#         if self.transform is not None:
#             im_data = self.transform(im_data)
#         return im_data,im_label
#
#     # 返回样本的总数
#     def __len__(self):
#         return len(self.imgs)
#
# # def file_name(path):
# #     l = []
# #     for p in Path(path).iterdir():
# #         for s in p.rglob('*.png'):
# #             l.append(s)
# #     return l
#
# # 数据dataset
# train_list0 = glob.glob('cifar10/TRAIN/*/*.png')
# train_list = []
# for item in train_list0:
#     item = item.replace('\\','/')
#     train_list.append(item)
# # print(train_list)
# test_list0 = glob.glob('cifar10/TEST/*/*.png')
# test_list = []
# for item in test_list0:
#     item = item.replace('\\','/')
#     test_list.append(item)
# # print(train_list)
# train_dataset = MyDataset(train_list,transform=train_transform)
# test_dataset = MyDataset(test_list,transform=test_transform)
# train_loader = DataLoader(train_dataset,batch_size=6,shuffle=True,num_workers=4)
# test_loader = DataLoader(test_dataset,batch_size=6,shuffle=False,num_workers=4)

# print("num",len(train_dataset))
# print("num",len(test_dataset))
# print(test_loader)
# s=0
# if __name__ == '__main__':
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         s += i
#         print(labels)

# print(s)



def default_loader(path):
    return Image.open(path).convert("RGB")

# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(90),
#     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
#     transforms.RandomGrayscale(0.2),
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor()
# ])

class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader = default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            #"/home/kuan/dataset/CIFAR10/TRAIN/" \
            #"airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split("/")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)

im_train_list = glob.glob("cifar10/TRAIN/*/*.png")
im_test_list = glob.glob("cifar10/TEST/*/*.png")

train_dataset = MyDataset(im_train_list,
                         transform=train_transform)
test_dataset = MyDataset(im_test_list,
                        transform =test_transform)

train_loader = DataLoader(dataset=train_dataset,
                               batch_size=128,
                               shuffle=True,
                               num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                               batch_size=128,
                               shuffle=False,
                               num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))









