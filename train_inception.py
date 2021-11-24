import torch
import torch.nn as nn

from inception import InceptionNetSmall

from load_cifar10 import train_loader,test_loader
import os
import tensorboardX
import torchvision

# 判断是否使用gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_num =200
lr = 0.01
batch_size = 128

net = InceptionNetSmall()
net = nn.DataParallel(net)
net = net.to(device)

#loss
loss_func = nn.CrossEntropyLoss()
#optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
#scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,
                                            gamma=0.9)
# 创建文件夹
model_path = "models/pytorch_inception"
log_path = "logs/pytorch_inception"

if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)
writer = tensorboardX.SummaryWriter(log_path)
step_n = 0

if __name__ == '__main__':
    for epoch in range(epoch_num):
        print("epoch is",epoch)
        # net 在train阶段是进行BN和dropout
        net.train()

        for i,data in enumerate(train_loader):
            input,labels = data
            input,labels = input.to(device),labels.to(device)

            outputs = net(input)
            loss = loss_func(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs.data,dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            # print("train step:",i,"loss is:",loss.item(),
            #       "min-batch correct is :",100.0*correct/batch_size)
            writer.add_scalar("train loss",loss.item(),global_step=step_n)
            writer.add_scalar("train correct",100.0*correct/batch_size,global_step=step_n)
            im = torchvision.utils.make_grid(input)
            writer.add_image("train im",im,global_step=step_n)

            step_n += 1



        torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))

        scheduler.step()
        print("lr is :", optimizer.state_dict()["param_groups"][0]["lr"])

        sum_loss = 0
        sum_correct = 0

        for i,data in enumerate(test_loader):
            net.eval()
            input,labels = data
            input,labels = input.to(device),labels.to(device)

            outputs = net(input)
            loss = loss_func(outputs,labels)


            _, pred = torch.max(outputs.data,dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            sum_loss+=loss.item()
            sum_correct+=correct.item()
            im = torchvision.utils.make_grid(input)
            writer.add_image("test im ",im,global_step=step_n)


        test_loss = 1.0*sum_loss / len(test_loader)
        test_correct = 100.00*sum_correct/len(test_loader)/batch_size

        writer.add_scalar("test loss",test_loss,global_step=epoch+1)
        writer.add_scalar("test correct",test_correct,global_step=epoch+1)

        print("epoch:",epoch,"test_loss is:",test_loss,
              "test_corrext :",test_correct)

writer.close()






