'''
    DKD训练学生网络
'''
from model import ResNet18, StudentVGGNet, ResNet50, VGGNet, low_rank_loss, shufflenetv2, MobileNetv2
import torch
import torch.nn as nn
import os
import time
import numpy as np
import warnings
from utils import func_GTSave, draw_curve, load_dataset, get_dataloader, load_training_set, draw_curve_student, detect_changes_batch, path_exists_make
from train import train_student, test_student
import scipy.io as sio
from dkd_loss import DKD
warnings.filterwarnings('ignore')


#Get cpu or gpu device for training
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")
print("--------------------------------")

# 超参数设置
dataset = 'lasvegas' # 'hermiston' / 'yancheng' / 'bayarea'
data_path = os.path.join('dataset', dataset) # save data
model_name = 'resnet' # 'resnet', 'vgg'
epochs = 100
temp = 4 # temperature of the softmax
batch_size = 64
num_workers = 0
shuffle = True
reg_lr = 0.01 # regularization coefficient of the low rank loss
KD_mode = 'KD'
mode = 'BCD' # 'BCD'--binary change detection mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = False # whether to use pca when create the samples (memory limit)
numComponents = 13 # pca dimensions
#model/hermiston/resnet/model.pth
save_teacher_path = "model/{}/{}/teacher/model.pth".format(dataset,model_name) # save model weights
#result/hermiston/resnet/student
result_path = ".temp/" # save result
#model/hermiston/student
save_student_path = os.path.join(os.path.join(os.path.join(os.path.join('DKD\\model', dataset), model_name), 'student'))

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
        model_stu = ResNet18(bands=13, mode=KD_mode).to(device)
        model_tea = ResNet50(bands=13, mode=KD_mode).to(device)
    elif model_name == "vgg":
        model_stu = StudentVGGNet(bands=bands, mode=KD_mode).to(device)
        model_tea = VGGNet(bands=bands, mode=KD_mode).to(device)

# student = "shufflenet"
# if student == "shufflenet": #size = 1 for hermiston IN_CHAN = 3 FOR HERMISTON
#     model_stu = shufflenetv2(bands=3, num_class=2, mode='KD', ret='single', srrl=False).to(device)
#     model_tea = ResNet50(bands=3, mode=KD_mode).to(device)
# if student == "mobilenet":
#     model_stu = MobileNetv2(bands=64, num_class=2, mode='KD', ret='single').to(device)
#     model_tea = VGGNet(bands=64, mode=KD_mode).to(device)
model_tea.load_state_dict(torch.load(save_teacher_path))
# model_stu.load_state_dict(torch.load("DKD/model/zy3/resnet/student/model_epoch_37_oa_0.9161712665317824_kappa_0.5576497508893845.pth"))
# =================================================================测试部分！
# model_stu.load_state_dict(torch.load(".temp/model.pth"))
# since = time.time()
# acc, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, KD_mode)
# end = time.time()
# print('Inference time is: {}'.format(end-since))
# time.sleep(1000)
# =================================================================测试部分！

#加载教师数据集
#model_tea.load_state_dict(torch.load(save_teacher_path))

# loss_fn=nn.CrossEntropyLoss()
#低秩损失，低秩正则化
# loss_lr = low_rank_loss()
loss_fn = DKD()
optimizer = torch.optim.SGD(model_stu.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
val_acc_list=[]
max_kc=0

#相比教师网络添加了
#DKD损失
train_DKD_list = []
val_DKD_list = []

#低秩正则化项的损失
train_low_rank_loss_list = []
val_low_rank_loss_list = []

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, ".temp/" + filename)
        
# train_student返回total_loss, total_DKD_loss, total_low_rank_loss, total_acc

#学生网络的loss：网络输出概率和标签Y之间的Cross Entropy+带有温度的教师和学生网络的logits的KL散度+低秩正则化项
for t in range(epochs):
    print(f"epoch{t+1}\n-----------------------")
    print("Learning rate is set as:{}".format(learning_rate))
    #train_hard_loss
    train_loss, train_DKD_loss, train_low_rank_loss, train_acc = train_student(train_loader, model_tea, model_stu, loss_fn, optimizer, device, temp, KD_mode, reg_lr)
    train_loss_list.append(train_loss) 
    train_DKD_list.append(train_DKD_loss)
    train_low_rank_loss_list.append(train_low_rank_loss)
    train_acc_list.append(train_acc)

    test_loss, test_DKD_loss, test_low_rank_loss, test_acc = test_student(test_loader, model_tea, model_stu, loss_fn, device, temp, KD_mode, reg_lr)
    val_loss_list.append(test_loss)
    val_DKD_list.append(test_DKD_loss)
    val_low_rank_loss_list.append(test_low_rank_loss)
    val_acc_list.append(test_acc)

    stepLR.step()

    since = time.time()
    oa, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, 'KD')
    end = time.time()

    print('Inference time is: {}'.format(end - since))

    # save_checkpoint({
    #             'epoch': t + 1,
    #             'state_dict': model_stu.state_dict(),
    #             'optimizer' : optimizer.state_dict(),
    #         }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(t))
    torch.save(model_stu.state_dict(), os.path.join(os.path.join(os.path.join(os.path.join('DKD\\model', dataset), model_name), 'student'), 'model_epoch_{}_oa_{}_kappa_{}.pth'.format(t, oa, kc)))
    if kc > max_kc:
        max_kc = kc
        #保存模型的参数
        torch.save(model_stu.state_dict(), save_student_path)

    # np.save(".temp/train_loss.npy", train_loss_list)
    # np.save(".temp/train_DKD_list.npy", train_DKD_list)
    # np.save(".temp/train_acc_list.npy", train_acc_list)
    # np.save(".temp/val_acc_list.npy", val_acc_list)
    # draw_curve_student(train_loss_list, train_DKD_list, train_low_rank_loss_list, result_path, 'train loss.png')
    # draw_curve_student(val_loss_list, val_DKD_list, val_low_rank_loss_list, result_path, 'validation loss.png')
    # draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    # draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)
# if acc > max_acc:
#     max_acc = acc
#     torch.save(model_stu.state_dict(), save_student_path)

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
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'train_DKD_loss': train_DKD_list, 
                                                    'train_low_rank_loss': train_low_rank_loss_list, 'val_loss': val_loss_list, 'train_DKD_loss': train_DKD_list,
                                                     'val_low_rank_loss': val_low_rank_loss_list})

sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
func_GTSave(mask, os.path.join(result_path, 'mask.png'))