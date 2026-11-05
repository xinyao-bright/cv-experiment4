# 实验四 多层感知机手写数字识别

姓名：薛昕瑶　学号：202410315005

## 一、实验目的

用 PyTorch 搭建一个简单的多层感知机，完成 MNIST 手写数字分类，熟悉数据处理、前向传播、反向传播和模型测试的过程，再观察学习率对训练结果的影响。

## 二、数据与实验设置

MNIST 训练集有 60000 张图像，测试集有 10000 张，每张都是 28 × 28 的灰度图，对应数字 0 到 9。输入模型前，将像素除以 255，再把每张图展平成 784 维向量。训练时打乱样本，测试时保持原有顺序。

![MNIST 训练样本](results/samples.png)

基础实验采用参考 PDF 中的设置。两组实验只改变学习率，网络结构、初始权重、训练数据顺序和训练轮数保持一致，随机种子设为 42。两组都在训练结束后测试一次，没有根据测试结果选择训练轮数。

| 项目 | 设置 |
| --- | --- |
| 网络结构 | 784 → 128 → 10 |
| 隐藏层激活函数 | ReLU |
| 损失函数 | 交叉熵 |
| 优化器 | SGD |
| 批量大小 | 2048 |
| 训练轮数 | 30 |
| 基础学习率 | 0.01 |
| 对比学习率 | 0.1 |
| 运行环境 | Python 3.13.7、PyTorch 2.11.0、CPU |

## 三、主要实现

### 1. 数据处理

下面直接对数据集中的灰度像素进行处理，效果与参考代码的 ToTensor 一致。训练集和测试集分开读取，测试数据不参与参数更新。

```python
train_data = MNIST("data", train=True, download=True)
test_data = MNIST("data", train=False, download=True)
x_train = train_data.data.reshape(-1, 784).float() / 255.0
x_test = test_data.data.reshape(-1, 784).float() / 255.0
y_train, y_test = train_data.targets, test_data.targets
```

### 2. 定义模型

模型包含两层全连接层，中间用 ReLU 引入非线性，一共有 101770 个可训练参数。输出层给出 10 个类别的得分，得分最大的类别就是预测数字。交叉熵函数内部已经处理了归一化计算，因此输出层不再额外加 Softmax。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)
```

### 3. 模型训练

每个批次先清空梯度，再计算预测结果和损失，随后反向传播并更新参数。每轮的训练损失按样本数加权平均，避免最后一个较小批次影响统计结果。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(1, epochs + 1):
    model.train()
    loss_sum = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
    train_loss = loss_sum / len(train_set)
```

### 4. 模型测试

测试时切换到评估模式，并关闭梯度计算。准确率用预测正确的总数除以 10000，测试损失也按全部样本平均。

```python
model.eval()
loss_sum, correct = 0.0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_sum += F.cross_entropy(output, y, reduction="sum").item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
test_loss = loss_sum / len(test_loader.dataset)
accuracy = correct / len(test_loader.dataset)
```

## 四、实验结果与分析

### 1. 损失变化与测试结果

![两种学习率的训练损失](results/loss_curve.png)

| 实验 | 最后一轮训练损失 | 测试损失 | 测试准确率 | 错分数量 |
| --- | ---: | ---: | ---: | ---: |
| 学习率 0.01 | 0.6295 | 0.5995 | 86.26% | 1374 |
| 学习率 0.1 | 0.2621 | 0.2548 | 92.68% | 732 |

基础模型的损失从 2.2703 降到 0.6295，能够学到数字的主要特征，但下降较慢。学习率改为 0.1 后，第 5 轮损失已经降到 0.4964，低于基础模型第 30 轮的损失。训练结束时，测试准确率提高了 6.42 个百分点，少错分了 642 张图像。

这说明在本次批量大小和训练轮数下，0.01 的学习率偏小，参数更新较慢。0.1 在这次实验中下降平稳、效果更好，但不能由此认为学习率越大越好，也不能把基础模型的结果完全归因于网络层数少。本次只使用一个随机种子，结果还不能代表不同初始化下的平均表现。

### 2. 混淆矩阵

![两组实验的混淆矩阵](results/confusion_matrix.png)

矩阵的行表示真实数字，列表示预测数字，对角线上的数值是预测正确的数量。调整学习率后，数字 5 的正确识别数量从 614 增加到 787，改善比较明显。数字 4 和 9 仍容易混淆，其中 40 张数字 4 被识别为 9，34 张数字 9 被识别为 4。两种数字的部分写法接近，仅靠这个简单网络还不能完全分开。

### 3. 预测示例与错误分析

![测试集前十张图像的预测结果](results/predictions.png)

上图展示测试集前 10 张图像的预测，绿色表示正确，红色表示错误。其中一张真实标签为 5 的图像被识别为 6，其下半部分有比较明显的闭合形状。

![前十张错分样本](results/errors.png)

错分图按测试集顺序取前 10 张，序号从 0 开始。有些数字存在倾斜、笔画粘连或写法不常见的情况，例如序号 92 的数字 9 被识别为 4。全连接网络直接使用展开后的像素，对笔画位置和形状变化的适应能力有限。后续可以尝试增加训练轮数，或使用卷积网络进一步比较，本次没有加入这些额外实验。

## 五、实验小结

本次完成了一个 784 → 128 → 10 的多层感知机，基础设置在测试集上的准确率为 86.26%，只调整学习率后达到 92.68%。同样的网络结构也会因为训练参数不同产生明显差别，因此分析结果时要同时看训练曲线和错分样本，不能只看最后的准确率。

## 六、代码与运行方法

[完整 Notebook](mlp_mnist.ipynb) 包含全部代码和本次运行输出，也可以运行 [Python 程序](mlp_mnist.py)。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python mlp_mnist.py
```

首次运行会下载 MNIST，之后可直接使用本地数据。运行后会更新 results 目录中的图片和数据。原始数据不上传到仓库。

逐轮训练损失见 [history.csv](results/history.csv)，最终指标见 [summary.csv](results/summary.csv)，环境信息见 [environment.json](results/environment.json)。每张测试图的标签和两组预测保存在 [predictions.npz](results/predictions.npz)。

参考资料：教师提供的实验四参考代码 PDF、[往年实验报告](https://github.com/Guyudong1/CV_exp/tree/main/4th.exp)。代码按参考模型实现，图像和数值均来自本次实际运行。
