"""
    @Author : Gilbert
    @Creation Time : 2025/7/23 
    @Description : 
"""
import random
import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l

def synetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成一个形状为(num_examples, len(w))的正态分布随机数矩阵X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算X与w的矩阵乘积，并加上偏置b
    y = torch.matmul(X, w) + b
    # 在y上加上一个形状为y的均值为0，标准差为0.01的正态分布随机数
    y += torch.normal(0, 0.01, y.shape)
    # 返回X和y的reshape后的结果
    return X, y.reshape((-1, 1))

# 生成真实参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成数据
features, labels = synetic_data(true_w, true_b, 1000)

# 打印生成的数据
print('features:', features[0], '\nlabels:', labels[0])

# 绘制散点图
plt.scatter(features[:,1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

def data_iter(batch_size, features, labels):
    # 获取特征和标签的长度
    num_example = len(features)
    # 生成一个包含所有索引的列表
    indices = list(range(num_example))
    # 这些脚本书随机读取的，没有特定的顺序
    # 打乱索引
    random.shuffle(indices)
    # 遍历所有样本
    for i in range(0, num_example, batch_size):
        # 获取当前批次的索引
        batch_indices = torch.tensor(indices[i:min(i + batch_size,num_example)])
        # 生成当前批次的特征和标签
        yield features[batch_indices], labels[batch_indices]

# 设置批量大小
batch_size = 10

# 打印第一个批量的数据
for X, y in data_iter(batch_size, features, labels):
    print('X:',X, '\ny:', y)
    break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    # 不计算梯度
    with torch.no_grad():
        # 遍历所有参数
        for param in params:
            # 更新参数
            param -= lr * param.grad / batch_size
            # 清空梯度
            param.grad.zero_()

# 设置学习率和训练轮数
lr = 0.003
num_epochs = 20
# 设置模型和损失函数
net = linreg
loss = squared_loss

# 开始训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 'X'和'y'的小批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使⽤参数的梯度更新参数
    # 在不计算梯度的情况下进行训练
    with torch.no_grad():
        # 计算训练损失
        train_l = loss(net(features, w, b), labels)
        # 打印训练损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
