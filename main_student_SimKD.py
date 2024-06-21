'''
    main function to train the student network (SimKD)
'''
from sympy import false
import torch
from utils import load_training_set, load_dataset, get_dataloader, draw_curve, func_GTSave, path_exists_make, detect_changes_batch_SimKD
from model import StudentVGGNet, VGGNet, ResNet50, ResNet18, SimKD, shufflenetv2, MobileNetv2
from train import train_student_SimKD, test_student_SimKD
import numpy as np
from torch import nn
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
dataset = 'hermiston' # 'hermiston' / 'yancheng' / 'bayarea'
data_path = os.path.join('dataset', dataset) # save data
model_name = 'vgg' # 'resnet', 'vgg'
save_teacher_path = os.path.join(os.path.join(os.path.join('model', dataset), model_name),  'model.pth') # save model weights
result_path = os.path.join(os.path.join(os.path.join('SimKD\\result', dataset), model_name), 'student') # save result
save_student_path = os.path.join(os.path.join(os.path.join('SimKD\\model', dataset), model_name), 'student')
epochs = 100
batch_size = 256
num_workers = 0
shuffle = True
KD_mode = 'KD'
mode = 'BCD' # 'BCD'--binary change detection mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = True # whether to use pca when create the samples (memory limit)
numComponents = 64 # pca dimensions

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
        model_stu = ResNet18(bands=numComponents, mode=KD_mode).to(device)
        model_tea = ResNet50(bands=numComponents, mode=KD_mode).to(device)
    elif model_name == 'vgg':
        model_stu = StudentVGGNet(bands=numComponents, mode=KD_mode).to(device)
        model_tea = VGGNet(bands=numComponents, mode=KD_mode).to(device)
else:
    if model_name == 'resnet':
        model_stu = ResNet18(bands=bands, mode=KD_mode).to(device)
        model_tea = ResNet50(bands=bands, mode=KD_mode).to(device)
    elif model_name == 'vgg':
        model_stu = StudentVGGNet(bands=bands, mode=KD_mode).to(device)
        model_tea = VGGNet(bands=bands, mode=KD_mode).to(device)

student = "mobilenet"
if student == "shufflenet":
    model_stu = shufflenetv2(num_class=2, size = 0.5).to(device)
    model_tea = ResNet50(bands=3, mode=KD_mode).to(device)
if student == "mobilenet":
    model_stu = MobileNetv2(bands=64, num_class=2, mode='KD', ret='single').to(device)
    model_tea = VGGNet(bands=64, mode=KD_mode).to(device)
model_tea.load_state_dict(torch.load(save_teacher_path))

if pca:
    data = torch.randn(2, numComponents, 3, 3).type(torch.cuda.FloatTensor)
else:
    data = torch.randn(2, bands, 3, 3).type(torch.cuda.FloatTensor)

model_tea.eval()
model_stu.eval()
__, feat_t = model_tea(data)
__, feat_s = model_stu(data)
#print(feat_t.shape) #torch.Size([2, 2048, 3, 3]) 
#print(feat_s.shape) #torch.Size([2, 512, 3, 3])
module_list = nn.ModuleList([])
module_list.append(model_stu)
trainable_list = nn.ModuleList([])
trainable_list.append(model_stu)

s_n = feat_s[-2].shape[0]
t_n = feat_t[-2].shape[0]
model_simkd = SimKD(s_n= s_n, t_n=t_n)
module_list.append(model_simkd)
trainable_list.append(model_simkd)
module_list.append(model_tea)

# define the optimizer and the loss function
loss_mse = nn.MSELoss()
optimizer = torch.optim.SGD(trainable_list.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
criterion_list = nn.ModuleList([])
criterion_list.append(loss_mse)
criterion = nn.CrossEntropyLoss()

criterion_list.cuda()
module_list.cuda()

# training
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
max_acc = 0
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss, train_acc = train_student_SimKD(train_loader, module_list, criterion_list, optimizer, device, KD_mode, model_name)
    test_loss, test_acc = test_student_SimKD(test_loader, module_list, criterion, device, KD_mode, model_name)
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
    oa, kc, mask = detect_changes_batch_SimKD(module_list, data_path, hsi_gt_b, device, batch_size, KD_mode, model_name)
    sio.savemat(os.path.join(result_path, 'mask_epoch_{}_oa_{}_kappa_{}.mat'.format(t, oa, kc)), {'mask': mask})
    torch.save(model_stu.state_dict(), os.path.join(os.path.join(os.path.join(os.path.join('SimKD\\model', dataset), model_name), 'student'), 'model_epoch_{}_oa_{}_kappa_{}.pth'.format(t, oa, kc)))
    func_GTSave(mask, os.path.join(result_path, 'mask_epoch_{}_oa_{}_kappa_{}.png'.format(t, oa, kc)))
print('Done!')
print('-------------------------------')

# Change Detection
model_stu.load_state_dict(torch.load(save_student_path))
module_list[0] = model_stu
since = time.time()
oa, kc, mask = detect_changes_batch_SimKD(module_list, data_path, hsi_gt_b, device, batch_size, KD_mode, model_name)
end = time.time()
print('Inference time is: {}'.format(end-since))

# save experimental results
sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'val_loss': val_loss_list})
sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
func_GTSave(mask, os.path.join(result_path, 'mask.png'))
