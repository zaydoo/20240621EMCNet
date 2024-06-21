'''
'''
'''
        训练教师网络
'''
'''
'''
from utils import generate_training_set, load_dataset, get_dataloader, func_gtImshow, detect_changes_batch, draw_curve, func_GTSave, save, path_exists_make, generate_training_set_both, load_training_set
import numpy as np
import torch
from dataset import ChangeDetectionDataset
from torch.utils.data import DataLoader
from model import VGGNet, ResNet50
from torch import nn
from train import train, test
import scipy.io as sio
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")
print('-------------------------------')

# hyper-parameter setting
dataset = "hermiston" # 'hermiston' / 'yancheng' / 'bayarea' / 'river'
mode = 'BCD'
patch_size = 3
training_ratio = 0.9 # ratio of training samples to total samples
data_path = os.path.join('dataset', dataset) # save data
batch_size = 64 # 'hermiston' and 'yancheng' -- 64
num_workers = 0
epochs = 100
shuffle = True
manner = 'random' # random -- random upsampling
KD_mode = 'NKD' # NKD -- not in knowledge distillation mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = True # whether to use pca when create the samples (memory limit)
numComponents = 64 # pca dimensions
model_name = 'resnet' # 'resnet', 'vgg'
is_standard_cva = True # whether data needed to be standardized in cva (bayarea true, otherwise false)

print('backbone is: {}'.format(model_name))
print('-------------------------------')

#检测path是否存在，如果不存在则创建
path_exists_make(data_path)
#保存路径
save_path = os.path.join('model', dataset)
path_exists_make(save_path) #model/hermiston
#结果路径
result_path = os.path.join('result', dataset)
path_exists_make(result_path) #result/hermiston

#权重保存路径
#model/hermiston/resnet
save_path = os.path.join(save_path, model_name) # save model weights
path_exists_make(save_path)
result_path = os.path.join(result_path, model_name) # save result
path_exists_make(result_path)

save_path = os.path.join(save_path, 'model.pth')
#检查datasets路径下数据集是否存在
is_exists = os.path.exists(os.path.join(data_path, 'samples.npy'))

# generate training samples
#hsi_gt_m：多变化检测gt，这里应该拿到的是None
#hermiston.mat数据已经处理好了
hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m = load_dataset(dataset, mode)
#波段数量B
print(hsi_t1.shape) #390x200x242
bands = hsi_t1.shape[2]

if is_exists:
    #加载数据集
    training_samples, training_labels, test_samples, test_labels = load_training_set(data_path)
else:
    #如果不存在samples.npy，则根据初始化生成
    #CVA和CAD，欧式距离和余弦距离，第三章初始化部分
    training_samples, training_labels, test_samples, test_labels = generate_training_set_both(hsi_t1, hsi_t2, hsi_gt_b, mode, patch_size, training_ratio, data_path, hsi_gt_m, manner, pca, numComponents, is_standard_cva)
#数据预处理，get_dataloader返回data_loader的封装
train_loader, test_loader = get_dataloader(training_samples, training_labels.astype(np.int64), test_samples, test_labels.astype(np.int64), batch_size, num_workers, shuffle)

# 模型的建立
# 如果降维了，bands=pca降维后的维度
if pca:
    if model_name == 'resnet':
        model = ResNet50(bands=numComponents).to(device)
    elif model_name == 'vgg':
        model = VGGNet(bands=numComponents).to(device)
else:
    if model_name == 'resnet':
        model = ResNet50(bands=bands).to(device)
    elif model_name == 'vgg':
        model = VGGNet(bands=bands).to(device)

# print(model)

#交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
#随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
#milestones: 这是一个列表，其中包含了在训练的不同阶段（milestones）时将会降低学习率的时刻，设置为[40.70.90]
#学习率调度器，动态调整学习率
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# training
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
max_acc = 0

for t in range(epochs):
    print(f"epoch{t+1}\n-------------------------")
    print("Learning rate is set as:{}".format(optimizer.param_groups[0]['lr']))
    train_loss, train_acc=train(train_loader, model, loss_fn, optimizer, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    test_loss, test_acc=test(test_loader, model, loss_fn, device)
    val_loss_list.append(test_loss)
    val_acc_list.append(test_acc)

    stepLR.step()

    draw_curve(train_loss_list, 'Training loss', 'Epochs', 'Loss', 'b')
    draw_curve(val_loss_list, 'Validation loss', 'Epochs', 'Loss', 'r', result_path, 'loss.png', False)
    draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)
    #accuracy score, kappa score, mask
    oa, kc, mask = detect_changes_batch(model, data_path, hsi_gt_b, device, batch_size, 'NKD')
    
    if oa > max_acc:
        max_acc = oa
        #保存模型的参数
        torch.save(model.state_dict(), save_path)
print('Done!')
print('-------------------------------')




# Change Detection
#打印输出推理时间
model.load_state_dict(torch.load(save_path))
since = time.time()
#读取模型参数，进行判断推理
acc, kc, mask = detect_changes_batch(model, data_path, hsi_gt_b, device, batch_size, KD_mode)
end = time.time()
print('Inference time is: {}'.format(end-since))

# save experimental results
sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'val_loss': val_loss_list})
sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
#save mask
func_GTSave(mask, os.path.join(result_path, 'mask.png'))



#与八种无监督变化检测方法进行比较
#与三种知识蒸馏方法进行比较