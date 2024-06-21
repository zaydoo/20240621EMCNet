import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
import torch
import scipy.io as sio
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import random
from PIL import ImageFilter
import torchvision.transforms as transforms
#from train_options import Args
import os
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class Args:
    def __init__(self) -> None:
        self.batch_size = 256
        self.lr = 0.01
        self.epochs = 100
        self.patch_size = 5
        self.training_ratio = 0.9
        self.manner = 'random'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.milestones = [40, 70, 90]
        self.temp = 4 # temperature of the softmax
        self.reg_lr = 0.01 # regularization coefficient of the low rank loss
        self.patch_size_for_simsiam = 100 #224
        self.step_for_simsiam_patches = 50 # 50
        self.backbone = "ResNet"
        self.dataset = "zy3"
args = Args()

def applyPCA(X, numComponents=64):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components = numComponents, whiten = True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x, y):
        q = self.base_transform(x)
        k = self.base_transform(y)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

#标准化，均值为0方差为1
def Standartize_Data(X):
    newX = np.reshape(X,(-1, X.shape[2]))
    scaler = StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX

#自适应边界填充(win_size--卷积核大小3*3、5*5,输入为三维)
def Adaptive_Boundary_Filling(X, win_size=3):
    rows = X.shape[0]
    cols = X.shape[1]
    bands = X.shape[2]
    t = (win_size - 1) // 2
    newX = np.zeros(shape=(rows+2*t, cols+2*t, bands))
    newX[t:rows+t, t:cols+t, :] = X[:, :, :]
    newX[t:rows+t, 0:t, :] = X[:, t-1::-1, :]#切片[起始:终止:步长],起始闭,终止开
    newX[t:rows+t, t+cols:cols+2*t, :] = X[:, cols-1:cols-t-1:-1, :]
    newX[0:t, :, :] = newX[2*t-1:t-1:-1, :, :]
    newX[t+rows:rows+2*t, :, :] = newX[t+rows-1:rows-1:-1, :, :]
    return newX

class Dataset_num (Dataset):
    def __init__(self, flag = "train") -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'

        if self.flag == 'train':
            self.data = args.data_train
        else:
            self.data = args.data_val
    def __getitem__(self, index: int):
        val = self.data[index]
        if val > 8:
            label = 1
        else:
            label = 0
        return torch.tensor(label, dtype=torch.long),torch.tensor([val], dtype=torch.float32)
    def __len__(self) -> int:
        return len(self.data)


class Dataset_MSI(Dataset):
    def __init__(self, flag, dataset)-> None:
        super(Dataset_MSI, self).__init__()
        """
            伪标签
        """
        pseudo_label_path =  "zy3_pseudo_label_by_PCAKmeans.png" if dataset == "zy3" else "lasvegas_pseudo_label_by_CVA.png"
        self.pseudo_labels = Image.open(pseudo_label_path)
        self.pseudo_labels = np.array(self.pseudo_labels)
        self.pseudo_labels[self.pseudo_labels == 255] = 1
        #print(self.pseudo_labels)
        #print("pseduo_label的数值分布:", np.unique(self.pseudo_labels, return_counts=True))
        """
            .mat图像文件
        """
        mat_path = "dataset/zy3_double.mat" if dataset == "zy3" else "dataset/lasvegas_double.mat"
        if dataset == "hermiston":
            mat_path = "dataset/hermiston.mat"
        elif dataset == "bayarea":
            mat_path = "dataset/bayarea.mat"
        elif dataset == "yancheng":
            mat_path = "dataset/yancheng.mat"
        msi = sio.loadmat(mat_path) 
        self.msi_t1 = msi["msi_t1"]
        self.msi_t2 = msi["msi_t2"]
        self.msi_gt_b = msi["msi_gt_b"]
        #print("msi_gt_b的数据分布:",np.unique(self.msi_gt_b,return_counts=True))
        from utils import calculate_metrics
        """屏蔽函数的print输出"""
        #with suppress_stdout_stderr():
        #acc, kappa, rc, f1, false_alarm_rate = calculate_metrics(self.pseudo_labels,self.msi_gt_b)
        #print(f"伪标签的OA:{acc}, KAPPA:{kappa}")
        self.patch_size = args.patch_size
        #print("patch_size is set as:", self.patch_size)
        #=========================generating samples========================
        # print('generating samples')
        """获取diff图"""
        [rows, cols, bands] = self.msi_t1.shape
        #data_t1 = self.msi_t1.copy().astype(np.uint8) #定義、統一數據格式尤其重要!python語言會溢出自動取模運算!!!!!!!
        #data_t2 = self.msi_t2.copy().astype(np.uint8) #這裡之前實驗存在問題，所以做相應的修改，diff是無符號類型，不會存在負數，數據特征有問題，但影響應該不大
        data_t1 = self.msi_t1.copy().astype(np.float32) # zy3_oa0.9054,kappa0.5491==patchsize5.pth这个傻逼文件是用uint8数据训练的
        data_t2 = self.msi_t2.copy().astype(np.float32)
        #print("图像數據的類型是：",type(data_t1[0][0][0]))
        diff = data_t2 - data_t1
        diff = np.absolute(diff)
        if dataset == "hermiston":
            numComponents = 64
            diff = applyPCA(diff, numComponents)
            bands = numComponents
        diff = Standartize_Data(diff) #标准化
        diff = Adaptive_Boundary_Filling(diff, win_size= args.patch_size) #自适应边界填充,填的不是0
        t = (self.patch_size - 1) // 2
        self.samples = np.zeros([rows * cols, self.patch_size, self.patch_size, bands])
        for i in range(t, t+rows):
            for j in range(t, t+cols):
                patch = diff[i-t:i+t+1, j-t:j+t+1,:] #滑动窗口得到3 x 3的patch,作为samples
                index = (i-t)* cols + (j-t)
                self.samples[index,:,:,:] = patch
        self.samples = np.transpose(self.samples,(0,3,1,2)) #变成rows*cols, bands, patchsize, patchsize输入网络
        self.labels = np.zeros([rows * cols]).astype(np.int64)
        for i in range(0, rows):
            for j in range(0, cols):
                self.labels[i*cols+j] = self.pseudo_labels[i,j]
        cnt = np.sum(self.labels == self.msi_gt_b.reshape(-1))
        allpixels = rows * cols
        acc = cnt / allpixels
        # print(f"伪标签的Overall Accuracy为{acc}")
        num_per_label = np.zeros([2]).astype(np.int64)
        training_num = 0
        """统计psuedo_label为0和1的个数"""
        for i in range(2):
            num_per_label[i] = np.sum(self.labels == i)
            num_per_label[i] = math.ceil(num_per_label[i] * args.training_ratio)
            training_num = training_num + num_per_label[i]
        # print('num of training samples before upsampling: class 1 -- {}, class 2 -- {}'.format(num_per_label[0], num_per_label[1]))
        self.training_samples = np.zeros([training_num, bands, self.patch_size, self.patch_size])
        self.training_labels = np.zeros([training_num])

        self.test_samples = np.zeros([self.labels.shape[0]-training_num, bands, self.patch_size, self.patch_size])
        self.test_labels = np.zeros([self.labels.shape[0]-training_num])
        train_idx = 0
        test_idx = 0
        arr = np.arange(1, self.labels.shape[0]+1, 1)
        # 创建training_samples training_labels test_samples test_labels
        """把所有的samples随机拆分为train_samples和test_samples"""
        for i in range(self.labels.shape[0]): #这里labels已经是展平的1D tensor
            index = arr[i] - 1
            label = self.labels[index]
            if num_per_label[label] > 0:
                self.training_samples[train_idx, :, :, :] = self.samples[index, :, :, :]
                self.training_labels[train_idx] = label
                train_idx = train_idx+1
            else:
                self.test_samples[test_idx, :, :, :] = self.samples[index, :, :, :]
                self.test_labels[test_idx] = label
                test_idx = test_idx+1
            num_per_label[label] = num_per_label[label]-1
        """过采样，平衡样本数"""
        if args.manner == 'random':
            ros = RandomOverSampler()
            #ros = RandomUnderSampler()
            self.training_samples = self.training_samples.reshape([self.training_samples.shape[0], -1])
            self.training_samples, self.training_labels = ros.fit_resample(self.training_samples, self.training_labels)
            self.training_samples = self.training_samples.reshape([self.training_samples.shape[0], bands, args.patch_size, args.patch_size])
        if flag == "train":
            self.data = self.training_samples
            self.label = self.training_labels
        elif flag == "val":
            self.data = self.test_samples
            self.label = self.test_labels
        elif flag == "bug_test":
            self.data = self.training_samples[0:100,:,:,:]
            self.label = self.training_labels[0:100]
    def __getitem__(self, index) -> None:
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype = torch.long)
    def __len__(self):
        return len(self.data)

class Dataset_for_simsiam(Dataset):
    def __init__(self, dataset)-> None:
        super(Dataset_for_simsiam, self).__init__()
        # msi = sio.loadmat("dataset/zy3_double.mat") if dataset == "zy3" else sio.loadmat("dataset/lasvegas_double.mat")
        # self.msi_t1 = msi["msi_t1"]
        # self.msi_t2 = msi["msi_t2"]
        # self.msi_gt_b = msi["msi_gt_b"]
        if dataset == "hermiston":
            hsi = sio.loadmat("dataset/hermiston.mat")
            self.msi_t1 = hsi["hsi_t1"]
            self.msi_t2 = hsi["hsi_t2"]
            self.msi_gt_b = hsi["hsi_gt_b"]
        elif dataset == "bayarea":
            hsi = sio.loadmat("dataset/bayarea.mat")
            self.msi_t1 = hsi["hsi_t1"]
            self.msi_t2 = hsi["hsi_t2"]
            self.msi_gt_b = hsi["hsi_gt_2"]
        elif dataset == "yancheng":
            hsi = sio.loadmat("dataset/yancheng_new.mat") #hsi_gt
            self.msi_t1 = hsi["hsi_t1"]
            self.msi_t2 = hsi["hsi_t2"]
            self.msi_gt_b = hsi["hsi_gt"]
        elif dataset == "zy3":
            hsi = sio.loadmat("dataset/zy3_double.mat")
            self.msi_t1 = hsi["msi_t1"]
            self.msi_t2 = hsi["msi_t2"]
            self.msi_gt_b = hsi["msi_gt_b"]
        elif dataset == "lasvegas":
            hsi = sio.loadmat("dataset/lasvegas_double.mat")
            self.msi_t1 = hsi["msi_t1"]
            self.msi_t2 = hsi["msi_t2"]
            #self.msi_t1 = np.array(Image.open("C:/Users/iamzy/Desktop/get_parameter_analysis_result_of_zy3/dataset/visualized img/lasvegas_t1.png"))
            #self.msi_t2 = np.array(Image.open("C:/Users/iamzy/Desktop/get_parameter_analysis_result_of_zy3/dataset/visualized img/lasvegas_t2.png"))
            self.msi_gt_b = hsi["msi_gt_b"]
        self.patch_size = args.patch_size_for_simsiam
        # diff = np.absolute(self.msi_t1 - self.msi_t2)
        # diff = Standartize_Data(diff)
        # overlapping patches!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        step = args.step_for_simsiam_patches
        height = (self.msi_gt_b.shape[0]-self.patch_size)// step
        height += 1 
        #print(height) #112
        width = (self.msi_gt_b.shape[1]-self.patch_size)// step
        width += 1
        #print(width) #79
        self.samples_t1 = np.zeros((height * width ,self.patch_size,self.patch_size, self.msi_t1.shape[2]))
        self.samples_t2 = np.zeros((height * width ,self.patch_size,self.patch_size, self.msi_t1.shape[2]))
        index = 0
        for i in range(0, self.msi_gt_b.shape[0]-self.patch_size+1, step):
            for j in range(0, self.msi_gt_b.shape[1]-self.patch_size+1, step):
                patch_t1 = self.msi_t1[i:i+self.patch_size,j:j+self.patch_size,:] #滑动窗口得到3 x 3的patch,作为samples
                patch_t2 = self.msi_t2[i:i+self.patch_size,j:j+self.patch_size,:] 
                self.samples_t1[index,:,:,:] = patch_t1
                self.samples_t2[index,:,:,:] = patch_t2
                index += 1
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    # std=[0.229, 0.224, 0.225])
        self.augmentation = [
            #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),      
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8), 
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ]
        self.samples_t1 = self.samples_t1.transpose((0,3,1,2))
        self.samples_t2 = self.samples_t2.transpose((0,3,1,2))
    def __getitem__(self, index) -> None:
        # transform = TwoCropsTransform(transforms.Compose(self.augmentation))
        # q = Image.fromarray(self.samples_t1[index].astype('uint8')).convert('RGB')
        # k = Image.fromarray(self.samples_t2[index].astype('uint8')).convert('RGB')
        # q, k = transform(q, k)
        q = self.samples_t1[index]
        k = self.samples_t2[index]
        # q = Image.fromarray(.astype('uint8')).convert('RGB')
        # k = Image.fromarray(.astype('uint8')).convert('RGB')
        """diff.shape = (224, 224, 3)"""
        # q,k = transform(diff)
        # print(q1.shape)
        """原先的数据是numpy数组,一定要转换为torch.tensor!!!"""
        #t1.to(torch.uint8).squeeze()
        return torch.tensor(q, dtype = torch.float32), torch.tensor(k, dtype = torch.float32)
    def __len__(self):
        return len(self.samples_t1)


class test_dataset_for_zy3_simsiam(Dataset):
    def __init__(self)-> None:
        super(test_dataset_for_zy3_simsiam, self).__init__()
        msi = sio.loadmat("dataset/zy3_double.mat")
        self.msi_t1 = msi["msi_t1"]
        self.msi_t2 = msi["msi_t2"]
        self.msi_gt_b = msi["msi_gt_b"]
        self.patch_size = args.patch_size
        diff = np.absolute(self.msi_t1-self.msi_t2)
        diff = Standartize_Data(diff)
        #overlapping patches!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        step = 10
        height = (self.msi_gt_b.shape[0]-self.patch_size)// step
        height += 1 
        #print(height) #112
        width = (self.msi_gt_b.shape[1]-self.patch_size)// step
        width += 1
        #print(width) #79
        # 112 * 79
        self.samples = np.zeros((height * width ,self.patch_size,self.patch_size, diff.shape[2]))
        index = 0
        for i in range(0, self.msi_gt_b.shape[0]-self.patch_size+1, step):
            for j in range(0, self.msi_gt_b.shape[1]-self.patch_size+1, step):
                patch = diff[i:i+self.patch_size,j:j+self.patch_size,:] #滑动窗口得到3 x 3的patch,作为samples
                self.samples[index,:,:,:] = patch
                index += 1

        [rows, cols, bands] = diff.shape

        self.samples = np.transpose(self.samples, [0,3,1,2])
        self.labels = np.zeros([rows * cols]).astype(np.int64)
        for i in range(0, rows):
            for j in range(0, cols):
                self.labels[i*cols+j] = self.msi_gt_b[i,j]
        
        self.pseudo_labels = Image.open("zy3_pseudo_label_by_PCAKmeans.png")
        self.pseudo_labels = np.array(self.pseudo_labels)
        self.pseudo_labels[self.pseudo_labels == 255] = 1
        print(self.pseudo_labels)
        self.pseudo_labels = np.reshape(self.pseudo_labels,(rows*cols, -1))
    def __getitem__(self,index):
        return torch.Tensor(self.samples[index]), torch.Tensor(self.labels[index]), torch.Tensor(self.pseudo_labels[index])
    def __len__(self):
        return len(self.samples)
    

class dataset_for_CLS(Dataset):
    def __init__(self, device,dataset):
        super(dataset_for_CLS, self).__init__()
        if dataset =="zy3":
            msi = sio.loadmat("dataset/zy3_double.mat")
            self.msi_t1 = msi["msi_t1"]
            self.msi_t2 = msi["msi_t2"]
            self.msi_gt_b = msi["msi_gt_b"]
        elif dataset == "lasvegas":
            msi = sio.loadmat("dataset/lasvegas_double.mat")
            self.msi_t1 = msi["msi_t1"]
            self.msi_t2 = msi["msi_t2"]
            self.msi_gt_b = msi["msi_gt_b"]
        elif dataset == "hermiston":
            msi = sio.loadmat("dataset/hermiston.mat")
            self.msi_t1 = msi["hsi_t1"]
            self.msi_t2 = msi["hsi_t2"]
            self.msi_gt_b = msi["hsi_gt_b"]
        elif dataset == "bayarea":
            msi = sio.loadmat("dataset/bayarea.mat") #hsi_gt_2
            self.msi_t1 = msi["hsi_t1"]
            self.msi_t2 = msi["hsi_t2"]
            self.msi_gt_b = msi["hsi_gt_2"]
        elif dataset == "yancheng":
            msi = sio.loadmat("dataset/yancheng_new.mat") #hsi_gt
            self.msi_t1 = msi["hsi_t1"]
            self.msi_t2 = msi["hsi_t2"]
            self.msi_gt_b = msi["hsi_gt"]

        # print("msi_gt_b的数据分布:",np.unique(self.msi_gt_b,return_counts=True))

        self.patch_size = args.patch_size
        # print("patch_size is set as:", self.patch_size)
        #=========================generating samples========================
        # print('generating samples')
        """获取diff图"""
        [rows, cols, bands] = self.msi_t1.shape
        #data_t1 = self.msi_t1.copy().astype(np.uint8) #定義、統一數據格式尤其重要!python語言會溢出自動取模運算!!!!!!!
        #data_t2 = self.msi_t2.copy().astype(np.uint8) #這裡之前實驗存在問題，所以做相應的修改，diff是無符號類型，不會存在負數，數據特征有問題，但影響應該不大
        data_t1 = self.msi_t1.copy().astype(np.float32)
        data_t2 = self.msi_t2.copy().astype(np.float32)
        print("數據的類型是：",type(data_t1[0][0][0]))
        # diff = data_t2 - data_t1
        # diff = np.absolute(diff)
        # diff = Standartize_Data(diff) #标准化
        data_t1 = Adaptive_Boundary_Filling(data_t1, win_size= args.patch_size) #自适应边界填充,填的不是0
        data_t2 = Adaptive_Boundary_Filling(data_t2, win_size= args.patch_size)
        t = (self.patch_size - 1) // 2
        self.samples_t1 = np.zeros([rows * cols, self.patch_size, self.patch_size, bands])
        self.samples_t2 = np.zeros([rows * cols, self.patch_size, self.patch_size, bands])
        for i in range(t, t+rows):
            for j in range(t, t+cols):
                patch_t1 = data_t1[i-t:i+t+1, j-t:j+t+1,:] #滑动窗口得到3 x 3的patch,作为samples
                patch_t2 = data_t2[i-t:i+t+1, j-t:j+t+1,:]
                index = (i-t)* cols + (j-t)
                self.samples_t1[index,:,:,:] = patch_t1
                self.samples_t2[index,:,:,:] = patch_t2
        self.samples_t1 = np.transpose(self.samples_t1,(0,3,1,2)) #变成rows*cols, bands, patchsize, patchsize输入网络
        self.samples_t2 = np.transpose(self.samples_t2,(0,3,1,2))
        # self.samples_t1 = torch.tensor(self.samples_t1).to(device)  
        # self.samples_t2 = torch.tensor(self.samples_t2).to(device)
        # self.samples_t1 = self.samples_t1.type(torch.cuda.FloatTensor)
        # self.samples_t2 = self.samples_t2.type(torch.cuda.FloatTensor)
        if device == "cpu":
            self.samples_t1 = self.samples_t1.type(torch.FloatTensor)
            self.samples_t2 = self.samples_t2.type(torch.FloatTensor)

        self.labels = np.zeros([rows * cols, 1]).astype(np.int64)
        for i in range(0, rows):
            for j in range(0, cols):
                self.labels[i*cols+j] = self.msi_gt_b[i,j]
        # print(self.samples)
        # print(self.samples.shape)

        # self.pseudo_labels = Image.open("zy3_pseudo_label_by_PCAKmeans.png")
        # self.pseudo_labels = np.array(self.pseudo_labels)
        # self.pseudo_labels[self.pseudo_labels == 255] = 1

        # #print(self.pseudo_labels)

        # self.pseudo = np.zeros((rows * cols,2))
        # for i in range(0, rows):
        #     for j in range(0, cols):
        #         if self.pseudo_labels[i, j] == 1: #伪标签认为变化
        #             self.pseudo[i*cols+j,1] = 1
        #         else:
        #            self.pseudo[i*cols+j,0] = 1 #伪标签认为不变

        #self.model = model.to(device)       

        # self.feature_array = np.zeros((rows * cols, 2048))

        # for i in range(0, rows * cols, 64):
        #     feature,_,_,_ = self.model(self.samples[i:i+64,:,:,:], self.samples[i:i+64,:,:,:])
        #     self.feature_array[i:i+64,:] = feature.cpu().detach().numpy()

        # np.save("feature_array.npy",self.feature_array)

        #self.feature_array = np.load('feature_array.npy')
        # self.feature_map = np.zeros((rows * cols // 64, 64, 2048))
        # cnt = 0
        # for i in range(0, rows * cols):
        #     feature,_,_,_ = self.model(self.samples[cnt:cnt+64,:,:,:], self.samples[cnt:cnt+64,:,:,:])
        #     self.feature_map[i] = feature 
        #     cnt += 64
        #     if cnt > rows * cols:
        #         cnt -= 64
        #         remain = rows * cols - cnt
        #         feature,_,_,_ = self.model(self.samples[remain:rows * cols,:,:,:], self.samples[remain:rows * cols,:,:,:])
        #         self.feature_map[i] = feature 
        #         break
        # print(self.feature_map)
        # print(self.feature_map.shape)
        # time.sleep(1000)
    def __getitem__(self, index):
        # print(next(self.model.parameters()).device)
        # print(self.samples.device)
        # squeezed_tensor = [index].unsqueeze(0)  # 在最前面增加一个维度
        #print(squeezed_tensor.shape)
        
        #print(feature.shape)
        #print(self.feature)
        #elf.feature = np.array(self.feature.cpu())
        #print(self.feature.shape)
        # print(self.feature.shape)
        # print(feature.shape)
        # print(self.labels[index].shape)
        # print(self.pseudo[index].shape)

        return torch.Tensor(self.samples_t1[index]), torch.Tensor(self.samples_t2[index]), torch.Tensor(self.labels[index])
    def __len__(self):
        return len(self.samples_t1)



if __name__ == '__main__':
    # from model import Simsiam
    # from torchvision import models
    # model = Simsiam(models.__dict__['resnet50'], dim = 2048, pred_dim= 512).to("cuda:0")
    # checkpoint = torch.load('checkpoint_0100.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    #dataset = dataset_for_SIMPLE_CLSNET(model=model, device="cuda:0")
    dataset = Dataset_zy3_for_simsiam()
    # dataset = dataset_for_CLS(device="cuda:0")
    train_data = DataLoader(dataset, batch_size= 1, shuffle=False)
    for i, (t1, t2) in enumerate(train_data):
        print(i, t1.shape)
        plt.subplot(121)
        plt.imshow(t1.to(torch.uint8).squeeze())
        plt.subplot(122)
        plt.imshow(t2.to(torch.uint8).squeeze())
        plt.show()
        #print(pseudo.shape)


    # feature_array = np.load('feature_array.npy') #(256022, 2048)
    # print(feature_array.shape)


""" if __name__ == '__main__':
    dataset = Dataset_zy3_for_simsiam()   
    # 创建数据加载器
    batch_size = 32  # 设置批量大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历数据加载器，测试数据加载
    for q1, q2 in dataloader:
        # 这里可以对加载的批量数据进行一些测试操作，比如检查数据格式、打印数据的形状等
        print("Loaded batch shape:", q1.shape)
        # 进行其他测试操作...

    # 测试数据集长度
    print("Dataset length:", len(dataset))

    # print(a.pseudo_labels)
    # unique_values = np.unique(a.pseudo_labels)
    # print(unique_values) """
