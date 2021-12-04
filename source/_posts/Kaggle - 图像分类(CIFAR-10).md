---
title: Kaggle--图像分类(CIFAR-10) 基于Pytorch的实现
date: 2021-08-29 17:26:15
tags: Pytorch
---

> CIFAR-10 是计算机视觉领域中一个非常重要的数据集. 它由Hinton的学生Alex Krizhevsky 和 Ilya Sutskever整理的小型数据集. 其中包括 10 个不同类别的RGB 3通道 32 * 32 的图像: 飞机 (airplane)、汽车 (automobile)、鸟类 (bird)、猫 (cat)、鹿(deer)、狗(dog)、蛙类(frog)、马(horse)、船(ship) 和卡车 (truck). 我们可以直接从Kaggle官网获得数据集 https://www.kaggle.com/c/cifar-10. 其中包含一个test文件和train文件.  test 文件中有60000张图像, 每个类各有6000张. train文件中包含300000万张图像, 其中只有10000张作为测试, 省下的是Kaggle为了防止人工标记数据集的额外数据. 

## 1. 整理数据集

首先我们先导入一些必要的库

```python
import os
import math
import time
import torch
import shutil
import collections
import torchvision
import pandas as pd
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

data_dir = "C:\\Users\\***\\OneDrive\\桌面\\kaggle\\CIFAR-10"
```

我们需要对数据集进行整理以一遍训练和测试使用. 函数**read_labels**用来读取训练集的标签文件并以 **name: label** 的字典形式储存.

```python
def read_labels(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]	#从1开始读是为了排除文件的title行
    tokens = [l.rstrip().split(',') for l in lines]
    return dict((name, label) for name, label in tokens)
```

我们使用一种非常常规的数据处理方法将文件每个标签作为一个文件夹储存对应的图片, 但这种方法并不高效, 我们相当于把所有图片copy了一次, 当数据量非常大到时候这个方法可能会过于耗时.

``` python
def copyfile(filename, target_dir):	#将图片复制到对应文件夹, 如果文件夹不存在则创建文件夹
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
```

下面我们对所有图片进行reorganize, 其中valid_ratio是验证集样本和原始数据集样本的比 (我们需要把数据急分成两部分一部分用作验证一部分用作训练, 由于数据量并不是很小所以不需要做k折交叉验证). 让我们以 valid_ratio=0.1 为例，由于原始的训练集有 50000 张图像，因此 train_valid_test/train 路径中将有 45000 张图像⽤ 于训练，而剩下 5000 张图像将作为路径 train_valid_test/valid 中的验证集。组织数据集后，同类别的图像将被放置在同⼀⽂件夹下。

```python
def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]	#每一个标签的数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio))	#验证集中每个标签的数量
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        filename = os.path.join(data_dir, 'train', train_file)
        copyfile(filename, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(filename, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(filename, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label
```

然后我们再定义一个函数对测试集进行整理

```python
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))
      
def reorg_CIFAR10(data_dir, valid_ratio):
    labels = read_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

```python
batch_size = 128
valid_ratio = 0.1	# 10% 的训练集数据作为验证集
reorg_CIFAR10(data_dir, valid_ratio)
```

## 2. 图像增广

为了防止过拟合我们对图像进行增强, 由于测试集只用作测试所以我们字对其做标准化

```python
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),# 这里吧图像放大到 40*40 后在按比例取32*32是为了取局部特征
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),	# 垂直翻转
    torchvision.transforms.RandomVerticalFlip(),	# 水平翻转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) # 百分之五十的概率对曝光, 对比度, 饱和度, 色调进行变换
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
```

## 3. 读取数据集

接下来问他们用torchvision.datasets.ImageFolder实例来读取不同的数据集包括每张图片的标签.

```python
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train
    ) for folder in ['train', 'train_valid']
]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test
    ) for folder in ['valid', 'test']
]
```

使用DataLoader对数据进行图像增广的操作

```python
train_iter, train_valid_iter = [
    DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True	# drop last现在可有可无
    ) for dataset in (train_ds, train_valid_ds)
]

valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True, num_workers=4)
test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False, num_workers=4)
# num_workers 使用多线程运行
```

## 4.定义模型

这里我们使用models 模块中的resnet50模型, 对于CIAFAR-10我们从头训练不使用迁移学习. 但我们需要讲最后一层全连接层的输出改成我们需要的输出类别个数. 损失函数这里使用交叉熵误差.

```python
def get_net():
    num_classes = 10
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(models.resnet50().fc.in_features, num_classes)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## 5. 定义训练函数

### 辅助函数

对象Accumulator 用于计算所有变量之和

```python 
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

accuracy 用于计算预测正确的数量

```python
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))
```

evaluation_acc 用于计算验证准确率

```python
def evaluate_acc(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]
```

train_batch 用于对每个batch进行训练

```python
def train_batch(net, feature, label, loss, optimizer, device):
    if isinstance(feature, list):
        feature = [x.to(device) for x in feature]
    else:
        feature = feature.to(device)
    loss = loss.to(device)
    net.train()
    net.cuda()
    label = label.cuda(0)
    optimizer.zero_grad()
    pred = net(feature)
    l = loss(pred, label)
    l.sum().backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_acc_sum = (pred.argmax(dim=1) == label).sum().item()
    return train_loss_sum, train_acc_sum
```

### 定义训练函数

接下来我们定义train函数

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay):
    output_params = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches = len(train_iter)
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    net.cuda()
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        metric = Accumulator(3)
        for i, (feature, label) in enumerate(train_iter):
            l, acc = train_batch(net, feature, label, loss, optimizer, device)
            metric.add(l, acc, label.shape[0])
        if valid_iter is not None:
            valid_acc = evaluate_acc(net, valid_iter, device)
        scheduler.step()
        print(f'{epoch+1}' + f'train loss {metric[0] / metric[2]:.3f}' + f'train acc {metric[1] / metric[2]:.3f}' + f' time: {time.time() - start} sec')
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f' examples/sec on {str(device)}')
```

## 6. 训练模型

我们开始对模型进行训练, 这里使用GPU训练, 并且lr_period, lr_decay我们设置为4, 0.9, 意味着每4个周期学习率自乘0.9

```python
device, num_epochs, lr, wd = 'cuda', 100, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay)
```

## 7. 提交结果

但我们训练出了满意的结果后我们使用设计好的超参数和训练好的模型对测试集重新训练并且对测试集进行分类提交.

```python
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, device, lr_period, lr_decay)
for X, _ in test_iter:
    y_hat = net(X.to(device))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds}) # 此为Kaggle要求格式
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('C:\\Users\\***\\OneDrive\\桌面\\kaggle\\CIFAR-10\\submission.csv', index=False)
```

最终我们会得到一个submission.csv文件我们就可以把他上传到Kaggle啦.

## References

https://tangshusen.me/Dive-into-DL-PyTorch/#/

https://zh-v2.d2l.ai/

https://github.com/d2l-ai/d2l-zh

https://pytorch.org/docs/stable/index.html

https://blog.csdn.net/mao_hui_fei/article/details/89477938
