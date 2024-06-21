'''
    训练学生网络
'''
from model import ResNet18, StudentVGGNet, ResNet50, VGGNet, low_rank_loss,shufflenetv2, MobileNetv2
import torch
import torch.nn as nn
import os
import time
import numpy as np
import warnings
from utils import func_GTSave, draw_curve, load_dataset, get_dataloader, load_training_set, draw_curve_student, detect_changes_batch, path_exists_make
from train import train_student_igkd, test_student_igkd
import scipy.io as sio
warnings.filterwarnings('ignore')


#Get cpu or gpu device for training
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")
print("--------------------------------")

# 超参数设置
dataset = 'zy3' # 'hermiston' / 'yancheng' / 'bayarea'
data_path = os.path.join('dataset', dataset) # save data
model_name = 'resnet' # 'resnet', 'vgg'
epochs = 100
temp = 4 # temperature of the softmax
batch_size = 128
num_workers = 0
shuffle = True
reg_lr = 0.01 # regularization coefficient of the low rank loss
KD_mode = 'KD'
mode = 'BCD' # 'BCD'--binary change detection mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = False # whether to use pca when create the samples (memory limit)
numComponents = 64 # pca dimensions
#model/hermiston/resnet/model.pth
save_teacher_path = os.path.join(os.path.join(os.path.join('model', dataset), model_name),  'model.pth') # save model weights
#result/hermiston/resnet/student
result_path = os.path.join(os.path.join(os.path.join('result', dataset), model_name), 'student') # save result
#model/hermiston/resnet/student
save_student_path = os.path.join("EMCNet",os.path.join(os.path.join(os.path.join('model', dataset), model_name), 'student'))

print('backbone is:{}'.format(model_name))
print('--------------------------------')
path_exists_make(save_student_path)
path_exists_make(result_path)
save_student_path= os.path.join(save_student_path, 'model.pth')

hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m = load_dataset(dataset, mode)
bands=hsi_t1.shape[2]

training_samples, training_labels, test_samples, test_labels = load_training_set(data_path)
train_loader, test_loader = get_dataloader(training_samples, training_labels.astype(np.int64), test_samples, test_labels.astype(np.int64), batch_size, num_workers, shuffle)

if pca:
    if model_name == "resnet":
        model_stu = ResNet18(bands=numComponents, mode=KD_mode).to(device)
        model_tea = ResNet50(bands=numComponents, mode=KD_mode).to(device)
    elif model_name == "vgg":
        model_stu = StudentVGGNet(bands=numComponents, mode=KD_mode).to(device)
        model_tea = VGGNet(bands=numComponents, mode=KD_mode).to(device)
else:
    if model_name == "resnet":
        model_stu = ResNet18(bands=bands, mode=KD_mode).to(device)
        model_tea = ResNet50(bands=bands, mode=KD_mode).to(device)
    elif model_name == "vgg":
        model_stu = StudentVGGNet(bands=bands, mode=KD_mode).to(device)
        model_tea = VGGNet(bands=bands, mode=KD_mode).to(device)

# =================================================================测试部分！
# model_tea.load_state_dict(torch.load("model/zy3/resnet/model.pth"))
# since = time.time()
# acc, kc, mask = detect_changes_batch(model_tea, data_path, hsi_gt_b, device, batch_size, KD_mode)
# end = time.time()
# print('Inference time is: {}'.format(end-since))
# time.sleep(1000)
# =================================================================测试部分！

#加载教师数据集
# student = "shufflenet"
# if student == "shufflenet":
#     model_stu = shufflenetv2(num_class=2, size = 1).to(device)
#     model_tea = ResNet50(bands=3, mode=KD_mode).to(device)
# if student == "mobilenet":
#     model_stu = MobileNetv2(bands=3, num_class=2, mode='KD').to(device)
#     model_tea = VGGNet(bands=3, mode=KD_mode).to(device)
# model_tea.load_state_dict(torch.load(save_teacher_path))

loss_fn=nn.CrossEntropyLoss()
#低秩损失，低秩正则化
loss_lr = low_rank_loss()
optimizer=torch.optim.SGD(model_stu.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
val_acc_list=[]
max_acc=0

#相比教师网络添加了
#Cross Entropy的硬损失
train_hard_loss_list = []
val_hard_loss_list = []
#KL散度的软损失
train_soft_loss_list = []
val_soft_loss_list = []
#低秩正则化项的损失
train_low_rank_loss_list = []
val_low_rank_loss_list = []


#学生网络的loss：网络输出概率和标签Y之间的Cross Entropy+带有温度的教师和学生网络的logits的KL散度+低秩正则化项
for t in range(epochs):
    print(f"epoch{t+1}\n-----------------------")
    print("Learning rate is set as:{}".format(learning_rate))
    #train_hard_loss
    train_loss, train_hard_loss, train_soft_loss, train_low_rank_loss, train_acc = train_student_igkd(train_loader, model_tea, model_stu, loss_fn, optimizer, device, temp, KD_mode, loss_lr, reg_lr)
    train_loss_list.append(train_loss)
    train_hard_loss_list.append(train_hard_loss)
    train_soft_loss_list.append(train_soft_loss)
    train_low_rank_loss_list.append(train_low_rank_loss)
    train_acc_list.append(train_acc)

    test_loss, test_hard_loss, test_soft_loss, test_low_rank_loss, test_acc = test_student_igkd(test_loader, model_tea, model_stu, loss_fn, device, temp, KD_mode, loss_lr, reg_lr)
    val_loss_list.append(test_loss)
    val_hard_loss_list.append(test_hard_loss)
    val_soft_loss_list.append(test_soft_loss)
    val_low_rank_loss_list.append(test_low_rank_loss)
    val_acc_list.append(test_acc)

    stepLR.step()
    draw_curve(train_loss_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    draw_curve(val_loss_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)

    since = time.time()
    oa, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, 'KD')
    end = time.time()

    print('Inference time is: {}'.format(end - since))
    torch.save(model_stu.state_dict(), os.path.join(os.path.join(os.path.join(os.path.join('IGKD/model', dataset), model_name), 'student'), 'model_epoch_{}_oa_{}_kappa_{}.pth'.format(t, oa, kc)))
    if kc > max_acc:
        max_acc = kc
        #保存模型的参数
        torch.save(model_stu.state_dict(), save_student_path)

print('Done!')
print('-------------------------------')

# Change Detection
# 打印输出推理时间
model_stu.load_state_dict(torch.load(save_student_path))
since = time.time()
acc, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, KD_mode)
end = time.time()
print('Inference time is: {}'.format(end-since))

# 保存实验结果
sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'train_hard_loss': train_hard_loss_list, 'train_soft_loss': train_soft_loss_list,
                                                    'train_low_rank_loss': train_low_rank_loss_list, 'val_loss': val_loss_list, 'val_hard_loss': val_hard_loss_list,
                                                    'val_soft_loss': val_soft_loss_list, 'val_low_rank_loss': val_low_rank_loss_list})
sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
func_GTSave(mask, os.path.join(result_path, 'mask.png'))