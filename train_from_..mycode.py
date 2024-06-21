
import os
import time
from torch.utils.data import DataLoader
# from train_options import Args
from model import ResNet50,ResNet18, VGGNet, StudentVGGNet
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import tqdm
import numpy as np
from tqdm import tqdm
from utils import Detect_Changes, save_loss_curve
import scipy.io as sio
from torch.nn import Softmax
import warnings
import scipy.io as sio
warnings.filterwarnings('ignore')

"""
 _|_ . _  _ 
| | ||| |(_)                                                                                                                                                                                                                       
"""
"""
            every dataset should use the DOUBLE data (theortically)!! 
"""

class Args:
    def __init__(self) -> None:
        self.backbone = "resnet"
        self.dataset = "zy3"
        self.batch_size = 256
        self.lr = 0.01
        self.epochs = 100
        self.patch_size = 5
        self.training_ratio = 0.9
        self.manner = 'random'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.milestones = [40, 70, 90]
        self.temp = 4 # student network's temperature of the softmax
        self.reg_lr = 0.01 # regularization coefficient of the low rank loss
        self.patch_size_for_simsiam = 224
        self.step_for_simsiam_patches = 50


args = Args()

def train():
    print(f"using {args.dataset} dataset")
    print(f"====================================")
    '''定义Dataloader'''
    train_dataset = Dataset_MSI(flag = 'train', dataset = args.dataset)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    val_dataset = Dataset_MSI(flag ='val', dataset = args.dataset)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = True)
    '''定义模型'''
    print(f"backbone is {args.backbone}")
    print(f"the picture has {train_dataset.msi_t1.shape[2]} bands")
    bands = train_dataset.msi_t1.shape[2]
    model = VGGNet(bands = bands).to(args.device) if args.backbone == "vgg" else ResNet50(bands = bands).to(args.device)
    """定义损失"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    Max_Kappa = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        print("learning rate is set as: {}".format(optimizer.param_groups[0]['lr']))
        model.train()
        train_epoch_loss = []
        acc , nums = 0, 0
        # ============================== train ==============================
        for idx, (inputs, label) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(args.device)
            label = label.to(args.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(Softmax()(outputs), label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(Softmax()(outputs).max(axis = 1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}, loss = {}".format(100 * acc /nums, np.average(train_epoch_loss)))
        # ============================== val ==============================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            for idx,(inputs, label) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = model(inputs)
                loss = criterion(Softmax()(outputs), label)
                val_epoch_loss.append(loss.item())

                acc += sum(Softmax()(outputs).max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)
            print("valid acc = {:.2f}%, loss = {}".format(100 * acc / nums, np.average(val_epoch_loss)))
        stepLR.step()
        overall_accuracy, kappa, mask, time = Detect_Changes(model, args.device, args.batch_size, KD_mode = 'NKD')
        result_path = os.path.join("result",args.dataset, args.backbone,"teacher")
        model_path = os.path.join("model",args.dataset, args.backbone,"teacher", "model.pth")
        if kappa > Max_Kappa:
            Max_Kappa = kappa
            torch.save(model.state_dict(), model_path)
            print("BEST MODEL UPDATED (KAPPA) !!!!!")
            sio.savemat(result_path +'/mask.mat', {'mask':mask})
            plt.imsave(result_path + '/mask.png', mask, cmap='gray')
        # ============================== plot and save==============================
        save_loss_curve(train_epochs_loss, valid_epochs_loss, result_path)

def pred():
    model = ResNet18(bands = 64).to(args.device)
    #model = ResNet50(bands = 3).to(args.device)
    #model = VGGNet(bands= 3).to(args.device)
    #model = StudentVGGNet(bands= 3).to(args.device)
    model.load_state_dict(torch.load("DKD/hermiston/resnet/model2024年5月11日010020.pth"))
    model.eval()
    overall_accuracy, kappa, mask, time = Detect_Changes(model, args.device, batch_size=args.batch_size, KD_mode='NKD')
    print(f"oa = {overall_accuracy}, kappa = {kappa}")
    print(f"Inference time is {time}")

if __name__ == "__main__":
    #train()
    pred()