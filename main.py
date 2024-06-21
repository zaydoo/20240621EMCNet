'''
    main function to train the teacher network
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
dataset = "zy3" # 'hermiston' / 'yancheng' / 'bayarea' / 'river'
mode = 'BCD'
patch_size = 5
training_ratio = 0.9 # ratio of training samples to total samples
data_path = os.path.join('dataset', dataset) # save data
batch_size = 128 # 'hermiston' and 'yancheng' -- 64
num_workers = 0
epochs = 100
shuffle = True
manner = 'random' # random -- random upsampling
KD_mode = 'NKD' # NKD -- not in knowledge distillation mode
learning_rate = 0.01 # initial learning rate
milestones = [40, 70, 90] # learning rate decay 10 times at the epoch in the list
pca = False # whether to use pca when create the samples (memory limit)
numComponents = 4 # pca dimensions
model_name = 'resnet' # 'resnet', 'vgg'
is_standard_cva = False # whether data needed to be standardized in cva (bayarea true, otherwise false)

print('backbone is: {}'.format(model_name))
print('-------------------------------')

# create URL
path_exists_make(data_path)
save_path = os.path.join('model', dataset)
result_path = os.path.join('result', dataset)
path_exists_make(save_path)
path_exists_make(result_path)
save_path = os.path.join(save_path, model_name) # save model weights
result_path = os.path.join(result_path, model_name) # save result
path_exists_make(save_path)
path_exists_make(result_path)
save_path = os.path.join(save_path, 'model.pth')
is_exists = os.path.exists(os.path.join(data_path, 'samples.npy'))

# generate training samples
hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m = load_dataset(dataset, mode)
bands = hsi_t1.shape[2]
if is_exists:
    training_samples, training_labels, test_samples, test_labels = load_training_set(data_path)
else:
    training_samples, training_labels, test_samples, test_labels = generate_training_set_both(hsi_t1, hsi_t2, hsi_gt_b, mode, patch_size, training_ratio, data_path, hsi_gt_m, manner, pca, numComponents, is_standard_cva)
train_loader, test_loader = get_dataloader(training_samples, training_labels.astype(np.int64), test_samples, test_labels.astype(np.int64), batch_size, num_workers, shuffle)

# build model
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

# define the optimizer and the loss function
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# training
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
max_acc = 0
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    print("learning rate is set as: {}".format(optimizer.param_groups[0]['lr']))
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = test(test_loader, model, loss_fn, device)
    val_loss_list.append(test_loss)
    val_acc_list.append(test_acc)
    stepLR.step()
    draw_curve(train_loss_list, 'Training loss', 'Epochs', 'Loss', 'b')
    draw_curve(val_loss_list, 'Validation loss', 'Epochs', 'Loss', 'r', result_path, 'loss.png', False)
    draw_curve(train_acc_list, 'Training accuracy', 'Epochs', 'Accuracy', 'b')
    draw_curve(val_acc_list, 'Validation accuracy', 'Epochs', 'Accuracy', 'r', result_path, 'accuracy.png', False)
    if test_acc > max_acc:
        max_acc = test_acc
        torch.save(model.state_dict(), save_path)
        print("BEST MODEL UPDATED!!!!!")
    oa, kc, mask = detect_changes_batch(model, data_path, hsi_gt_b, device, batch_size, 'NKD')
    # if kc > max_kc:
    #     max_kc = kc
    #     torch.save(model.state_dict(), save_path)
    #     print("BEST MODEL UPDATED!!!!!")
print('Done!')
print('-------------------------------')

# Change Detection
model.load_state_dict(torch.load(save_path))
since = time.time()
acc, kc, mask = detect_changes_batch(model, data_path, hsi_gt_b, device, batch_size, KD_mode)
end = time.time()
print('Inference time is: {}'.format(end-since))

# save experimental results
sio.savemat(os.path.join(result_path, 'acc.mat'), {'train_acc': train_acc_list, 'val_acc': val_acc_list})
sio.savemat(os.path.join(result_path, 'loss.mat'), {'train_loss': train_loss_list, 'val_loss': val_loss_list})
sio.savemat(os.path.join(result_path, 'mask.mat'), {'mask': mask})
func_GTSave(mask, os.path.join(result_path, 'mask.png'))




