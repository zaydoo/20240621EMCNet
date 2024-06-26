'''
    main function to train the student network (SemCKD-AAAI-2021)
'''
import torch
from utils import load_training_set, load_dataset, get_dataloader, draw_curve, func_GTSave, path_exists_make, detect_changes_batch
from model import StudentVGGNet, VGGNet, ResNet50, ResNet18, SemCKDLoss, SelfA, shufflenetv2, MobileNetv2
from train import train_student_SemCKD, test_student_SemCKD
import numpy as np
from torch import nn
import scipy.io as sio
import os
import warnings
import time
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from torch.nn import Softmax
import random

# 设置随机数种子保证可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")
print('-------------------------------')

# hyper-parameter setting
dataset = 'lasvegas' # 'hermiston' / 'yancheng' / 'bayarea'/ 'zy3
data_path = os.path.join('dataset', dataset) # save data
model_name = 'resnet' # 'resnet', 'vgg'
save_teacher_path = os.path.join(os.path.join(os.path.join('model', dataset), model_name),  'model.pth') # save model weights
result_path = os.path.join(os.path.join(os.path.join('SemCKD\\result', dataset), model_name), 'student') # save result
save_student_path = os.path.join(os.path.join(os.path.join('SemCKD\\model', dataset), model_name), 'student')
epochs = 100
batch_size = 256
num_workers = 0
soft = 1.0
shuffle = True
KD_mode = 'KD'
mode = 'BCD' # 'BCD'--binary change detection mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = False # whether to use pca when create the samples (memory limit)
numComponents = 64 # pca dimensions]
ret = 'all'

print('backbone is: {}'.format(model_name))
print('-------------------------------')

# create URL
path_exists_make(save_student_path)
path_exists_make(result_path)
save_student_path = os.path.join(save_student_path, 'model.pth')

# load training samples
hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m = load_dataset(dataset, mode)
bands = hsi_t1.shape[2]
training_samples, training_labels, test_samples, test_labels = load_training_set(data_path)
train_loader, test_loader = get_dataloader(training_samples, training_labels.astype(np.int64), test_samples, test_labels.astype(np.int64), batch_size, num_workers, shuffle)

# build model
if pca:
    if model_name == 'resnet':
        model_stu = ResNet18(bands=numComponents, mode=KD_mode, ret=ret).to(device)
        model_tea = ResNet50(bands=numComponents, mode=KD_mode, ret=ret).to(device)
    elif model_name == 'vgg':
        model_stu = StudentVGGNet(bands=numComponents, mode=KD_mode, ret=ret).to(device)
        model_tea = VGGNet(bands=numComponents, mode=KD_mode, ret=ret).to(device)
else:
    if model_name == 'resnet':
        model_stu = ResNet18(bands=bands, mode=KD_mode, ret=ret).to(device)
        model_tea = ResNet50(bands=bands, mode=KD_mode, ret=ret).to(device)
    elif model_name == 'vgg':
        model_stu = StudentVGGNet(bands=bands, mode=KD_mode, ret=ret).to(device)
        model_tea = VGGNet(bands=bands, mode=KD_mode, ret=ret).to(device)

# student = "mobilenet"
# if student == "shufflenet":
#     model_stu = shufflenetv2(num_class=2, size = 0.5, ret="all").to(device)
#     model_tea = ResNet50(bands=3, mode=KD_mode, ret=ret).to(device)
# if student == "mobilenet":
#     model_stu = MobileNetv2(bands=64, num_class=2, mode='KD', ret='all').to(device)
#     model_tea = VGGNet(bands=64, mode=KD_mode, ret= ret).to(device)
    
model_tea.load_state_dict(torch.load(save_teacher_path))
# model_stu.load_state_dict(torch.load("SemCKD/model/zy3/resnet/student/model_epoch_8_oa_0.850774542812727_kappa_0.4563087399545891.pth"))
if pca:
    data = torch.randn(2, numComponents, 5, 5).type(torch.cuda.FloatTensor)
else:
    data = torch.randn(2, bands, 5, 5).type(torch.cuda.FloatTensor)

model_tea.eval()
model_stu.eval()
__, feat_t = model_tea(data)
__, feat_s = model_stu(data)
print(len(feat_t)) #2
print(len(feat_s)) #2
print(feat_t[0].shape) #torch.Size([2, 256, 4, 4])
print(feat_s[0].shape) #torch.Size([2, 256, 4, 4])
print("processing")
#time.sleep(1000)
module_list = nn.ModuleList([])
module_list.append(model_stu)
trainable_list = nn.ModuleList([])
trainable_list.append(model_stu)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        soft_logits_tea = Softmax()(y_t * 1.0/self.T)
        soft_logits_stu = Softmax()(y_s * 1.0/self.T)
        loss = torch.nn.BCEWithLogitsLoss().cuda()(soft_logits_stu, soft_logits_tea)
        return loss

criterion_cls = nn.CrossEntropyLoss()
kd_t = 4
criterion_div = DistillKL(kd_t)

s_n = [f.shape[1] for f in feat_s[0:]]
t_n = [f.shape[1] for f in feat_t[0:]]
criterion_kd = SemCKDLoss()
self_attention = SelfA(batch_size, s_n, t_n, soft)
module_list.append(self_attention)
trainable_list.append(self_attention)
module_list.append(model_tea)

# define the optimizer and the loss function
optimizer = torch.optim.SGD(trainable_list.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
criterion_list = nn.ModuleList([])
criterion_list.append(criterion_kd)
criterion = nn.CrossEntropyLoss()
criterion_list.append(criterion_cls)
criterion_list.append(criterion_div)

criterion_list.cuda()
module_list.cuda()

# training
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
max_acc = 0
# start_epoch = 9
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss, train_acc = train_student_SemCKD(train_loader, module_list, criterion_list, optimizer, device, batch_size, KD_mode, model_name)
    test_loss, test_acc = test_student_SemCKD(test_loader, module_list, criterion, device, KD_mode, model_name)
    stepLR.step()
    train_loss_list.append(train_loss)
    val_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(test_acc)
    draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)
    draw_curve(train_loss_list, 'Training loss', 'Epochs', 'MSE loss', 'b', result_path, 'train_loss.png', True)
    draw_curve(val_loss_list, 'Validation loss', 'Epochs', 'CE loss', 'r', result_path, 'validation_loss.png', True)
    if test_acc > max_acc:
        max_acc = test_acc
        torch.save(model_stu.state_dict(), save_student_path)
    oa, kc, mask = detect_changes_batch(module_list[0], data_path, hsi_gt_b, device, batch_size, KD_mode)
    sio.savemat(os.path.join(result_path, 'mask_epoch_{}_oa_{}_kappa_{}.mat'.format(t, oa, kc)), {'mask': mask})
    torch.save(model_stu.state_dict(), os.path.join(os.path.join(os.path.join(os.path.join('SemCKD\\model', dataset), model_name), 'student'), 'model_epoch_{}_oa_{}_kappa_{}.pth'.format(t, oa, kc)))
    func_GTSave(mask, os.path.join(result_path, 'mask_epoch_{}_oa_{}_kappa_{}.png'.format(t, oa, kc)))
print('Done!')
print('-------------------------------')

# Change Detection
model_stu.load_state_dict(torch.load(save_student_path))
since = time.time()
oa, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, KD_mode)
end = time.time()
print('Inference time is: {}'.format(end-since))

# save experimental results
sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'val_loss': val_loss_list})
sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
func_GTSave(mask, os.path.join(result_path, 'mask.png'))
