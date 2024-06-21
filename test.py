'''
    main function to train the student network
'''

import torch
from utils import load_training_set, load_dataset, get_dataloader, detect_changes_batch, func_gtImshow, draw_curve, func_GTSave, draw_curve_student, path_exists_make
from model import StudentVGGNet, VGGNet, low_rank_loss, ResNet50, ResNet18
from train import train_student, test_student
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

oa_list = []
kappa_list = []
times = 10
for i in range(times):
    print('time = {}'.format(i))
    print('-------------------------------')
    # hyper-parameter setting
    dataset = 'bayarea' # 'hermiston' / 'yancheng' / 'bayarea'
    data_path = os.path.join('dataset', dataset) # save data
    model_name = 'vgg' # 'resnet', 'vgg'
    save_teacher_path = os.path.join(os.path.join(os.path.join('model', dataset), model_name),  'model.pth') # save model weights
    result_path = os.path.join(os.path.join(os.path.join(os.path.join('result', dataset), model_name), 'student'), str(i)) # save result
    save_student_path = os.path.join(os.path.join(os.path.join(os.path.join('model', dataset), model_name), 'student'), str(i))
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
    model_tea.load_state_dict(torch.load(save_teacher_path))

    # define the optimizer and the loss function
    loss_fn = nn.CrossEntropyLoss()
    loss_lr = low_rank_loss()
    # optimizer = torch.optim.Adam(model_stu.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model_stu.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # training
    train_loss_list = []
    val_loss_list = []
    train_hard_loss_list = []
    val_hard_loss_list = []
    train_soft_loss_list = []
    val_soft_loss_list = []
    train_low_rank_loss_list = []
    val_low_rank_loss_list = []
    train_acc_list = []
    val_acc_list = []
    max_acc = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss, train_hard_loss, train_soft_loss, train_low_rank_loss, train_acc = train_student(train_loader, model_tea, model_stu, loss_fn, optimizer, device, temp, KD_mode, loss_lr, reg_lr)
        test_loss, test_hard_loss, test_soft_loss, test_low_rank_loss, test_acc = test_student(test_loader, model_tea, model_stu, loss_fn, device, temp, KD_mode, loss_lr, reg_lr)
        stepLR.step()
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)
        train_hard_loss_list.append(train_hard_loss)
        val_hard_loss_list.append(test_hard_loss)
        train_soft_loss_list.append(train_soft_loss)
        val_soft_loss_list.append(test_soft_loss)
        train_low_rank_loss_list.append(train_low_rank_loss)
        val_low_rank_loss_list.append(test_low_rank_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(test_acc)
        draw_curve_student(train_loss_list, train_hard_loss_list, train_soft_loss_list, train_low_rank_loss_list, result_path, 'train loss.png')
        draw_curve_student(val_loss_list, val_hard_loss_list, val_soft_loss_list, val_low_rank_loss_list, result_path, 'validation loss.png')
        draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
        draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)
        # if acc > max_acc:
        #     max_acc = acc
        #     torch.save(model_stu.state_dict(), save_student_path)
        since = time.time()
        oa, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, 'KD')
        end = time.time()
        print('Inference time is: {}'.format(end - since))
        if oa > max_acc:
            max_acc = oa
            torch.save(model_stu.state_dict(), save_student_path)
    print('Done!')
    print('-------------------------------')

    # Change Detection
    model_stu.load_state_dict(torch.load(save_student_path))
    since = time.time()
    oa, kc, mask = detect_changes_batch(model_stu, data_path, hsi_gt_b, device, batch_size, 'KD')
    oa_list.append(oa)
    kappa_list.append(kc)
    end = time.time()
    print('Inference time is: {}'.format(end-since))

    # save experimental results
    sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
    sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'train_hard_loss': train_hard_loss_list, 'train_soft_loss': train_soft_loss_list,
                                                        'train_low_rank_loss': train_low_rank_loss_list, 'val_loss': val_loss_list, 'val_hard_loss': val_hard_loss_list,
                                                        'val_soft_loss': val_soft_loss_list, 'val_low_rank_loss': val_low_rank_loss_list})
    sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
    func_GTSave(mask, os.path.join(result_path, 'mask.png'))

for i in range(times):
    print('i = {}, oa = {}, kappa = {}'.format(i, oa_list[i], kappa_list[i]))
    print('-------------------------------')



















