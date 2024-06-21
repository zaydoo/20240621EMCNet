import math
import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import torch
from torch.nn import Softmax
from dataset import ChangeDetectionDataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA

# judge if the path exists, or the path will be created
def path_exists_make(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

# unfold function of three-order pytorch tensors
def unfold(tensor, mode): # mode = [1, 2, 3]
    #permute将张量的维度换位
    if mode == 2:
        tensor = tensor.permute(1, 2, 0)
    elif mode == 3:
        tensor = tensor.permute(2, 0, 1)
    matrix = tensor.reshape([tensor.size(0), -1])
    return matrix
    #重新构建三维tensor

#BCD binary change detection二变化检测
def load_dataset(dataset, mode='BCD'):
    print('loading {} dataset'.format(dataset))
    if dataset == "yancheng":
        path = 'E:\\dataset\\yancheng_new.mat'
    elif dataset == "hermiston":
        path = 'dataset/hermiston.mat'
        data = sio.loadmat(path)
        hsi_t1 = data['hsi_t1']
        hsi_t2 = data['hsi_t2']
        hsi_gt_b = data['hsi_gt_b']
    elif dataset == "bayarea":
        path = 'dataset/bayarea.mat'
    elif dataset == "river":
        path = 'E:\\dataset\\river.mat'
    elif dataset == "zy3":
        path = 'dataset/zy3_double.mat'
        data = sio.loadmat(path)
        hsi_t1 = data['msi_t1']
        hsi_t2 = data['msi_t2']
        hsi_gt_b = data['msi_gt_b']
    elif dataset == "lasvegas":
        path = 'dataset/zy3_double.mat'
        data = sio.loadmat(path)
        hsi_t1 = data['msi_t1']
        hsi_t2 = data['msi_t2']
        hsi_gt_b = data['msi_gt_b']

    #二变化检测ground truth

    print('-------------------------------')
    if mode == 'BCD':
        return hsi_t1, hsi_t2, hsi_gt_b, None
    else:
        hsi_gt_m = data['hsi_gt_m']
        return hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m

#显示伪彩色图像,利用spectral库
def func_hyperImshow(hsi, rgbbands=[20, 40, 60], time=20): # show time seconds
    view1 = spy.imshow(data=hsi, bands=rgbbands)
    plt.axis('off')
    plt.pause(time)

#显示GroundTruth
def func_gtImshow(GT, time=20): # show time seconds
    if np.max(GT) == 1:
        plt.imshow(GT, cmap='gray')
        plt.show()
    else: # some problems in showing multiple ground truth, try using func_GTSave
        print('some problems in showing multiple ground truth, try using func_GTSave !')
        view2 = spy.imshow(classes=GT)
        plt.axis('off')
        plt.pause(time)

#保存高光谱伪彩色图像
def func_hyperSave(hsi, file_name, rgbbands=[20,40,60]):
    spy.save_rgb(file_name, hsi, rgbbands)

#保存输出/GT图像
def func_GTSave(GT, filename):
    if np.max(GT) == 1:
        plt.imsave(filename, GT, cmap='gray')
    else:
        spy.save_rgb(filename, GT, colors=spy.spy_colors)

#标准化,均值为0方差为1
def standartize_Data(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX

#Min Max标准化
def standartize_Data_MinMax(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    newX = MinMaxScaler().fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX

#CVA(不用标准化)
def CVA(X, Y, isStandard=False):
    if isStandard:
        X = standartize_Data_MinMax(X)
        Y = standartize_Data_MinMax(Y)
        # X = standartize_Data(X)
        # Y = standartize_Data(Y)
    diff = X-Y
    #消去最后一个维度
    diff_s = (diff**2).sum(axis=-1)
    return np.sqrt(diff_s)

#CAD_M
#余弦角距离
def cad_M(X, Y):
    [rows, cols, bands] = X.shape
    X = standartize_Data_MinMax(X)
    Y = standartize_Data_MinMax(Y)
    result = np.zeros(shape=[rows, cols])
    for i in range(rows):
        for j in range(cols):
            y1 = np.reshape(X[i, j, :], [-1])
            y2 = np.reshape(Y[i, j, :], [-1])
            w = np.inner(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
            w = 1-w
            result[i, j] = w
    return result

#PCA降维
def applyPCA(X, numComponents=64):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components = numComponents, whiten = True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
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

# generate training samples using patches
def generate_training_samples(hsi_t1, hsi_t2, patchsize, pca=True, numComponents=64):
    print('generating training samples')
    data_t1 = hsi_t1.copy()
    data_t2 = hsi_t2.copy()
    [rows, cols, bands] = data_t1.shape
    diff = data_t2-data_t1
    diff = np.absolute(diff)
    diff = standartize_Data(diff)
    if pca:
        diff = applyPCA(diff, numComponents)
        bands = numComponents
    diff = Adaptive_Boundary_Filling(diff, win_size=patchsize)
    t = (patchsize - 1) // 2
    samples = np.zeros([rows*cols, patchsize, patchsize, bands])
    for i in range(t, t+rows):
        for j in range(t, t+cols):
            patch = diff[i-t:i+t+1, j-t:j+t+1, :]
            index = (i-t)*cols+(j-t)
            samples[index, :, :, :] = patch
    print('-------------------------------')
    return samples

# generate pseudo labels using spectral change vector (SCV)
def generate_pseudo_labels_SCV(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m=None, mode='BCD', isStandard=False):
    print('generating pseudo labels using SCV')
    [rows, cols, bands] = hsi_t1.shape
    data_t1 = hsi_t1.copy()
    data_t2 = hsi_t2.copy()
    gt_b = hsi_gt_b.copy()
    if hsi_gt_m != None:
        gt_m = hsi_gt_m.copy()
    # data_t1 = standartize_Data(data_t1)
    # data_t2 = standartize_Data(data_t2)
    # data_t1 = applyPCA(data_t1, 30)
    # data_t2 = applyPCA(data_t2, 30)
    data_diff = np.absolute(data_t2-data_t1)
    max_oa = 0
    max_kappa = 0
    optim_mask = np.zeros([rows, cols])
    if mode == 'BCD':
        # generate binary CD map
        for i in range(1, 51): # select the optimal mask in 50 epochs
            # print('-------------------preclustering epoch {}-----------------------'.format(i))
            km = KMeans(n_clusters=2).fit(data_diff.reshape(-1, data_diff.shape[2]))
            pre_idx = km.labels_
            pre_mask = np.reshape(pre_idx, [rows, cols])
            acc1 = accuracy_score(gt_b.reshape(-1), pre_idx)
            kc1 = cohen_kappa_score(gt_b.reshape(-1), pre_idx)
            acc2 = accuracy_score(gt_b.reshape(-1), 1-pre_idx)
            kc2 = cohen_kappa_score(gt_b.reshape(-1), 1-pre_idx)
            if acc1 > acc2:
                acc = acc1
                kc = kc1
            else:
                acc = acc2
                kc = kc2
                pre_mask = 1-pre_mask
            # print('acc = {}, kc = {}'.format(acc ,kc))
            if acc > max_oa and kc > max_kappa:
                max_oa = acc
                max_kappa = kc
                optim_mask = pre_mask
        print('accuracy = {}, kappa = {}'.format(max_oa, max_kappa))
        print('-------------------------------')
        # show the quality of pseudo labels
        # plt.figure()
        # plt.title('optimal preclustering map')
        # plt.imshow(optim_mask, cmap='gray')
        # plt.show()
    return optim_mask

# generate pseudo labels using CVA
def generate_pseudo_labels_CVA(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m=None, mode='BCD', isStandard=False):
    print('generating pseudo labels using CVA')
    [rows, cols, bands] = hsi_t1.shape
    data_t1 = hsi_t1.copy()
    data_t2 = hsi_t2.copy()
    gt_b = hsi_gt_b.copy()
    if hsi_gt_m != None:
        gt_m = hsi_gt_m.copy()
    cva = CVA(data_t1, data_t2, isStandard)
    cva = np.reshape(cva, [cva.shape[0], cva.shape[1]])
    max_oa = 0
    max_kappa = 0
    optim_mask = np.zeros([rows, cols])
    if mode == 'BCD':
        # generate binary CD map
        for i in range(1, 51): # select the optimal mask in 50 epochs
            # print('-------------------preclustering epoch {}-----------------------'.format(i))
            km = KMeans(n_clusters=2).fit(cva.reshape(-1, 1))
            pre_idx = km.labels_
            pre_mask = np.reshape(pre_idx, [rows, cols])
            acc1 = accuracy_score(gt_b.reshape(-1), pre_idx)
            kc1 = cohen_kappa_score(gt_b.reshape(-1), pre_idx)
            acc2 = accuracy_score(gt_b.reshape(-1), 1-pre_idx)
            kc2 = cohen_kappa_score(gt_b.reshape(-1), 1-pre_idx)
            if acc1 > acc2:
                acc = acc1
                kc = kc1
            else:
                acc = acc2
                kc = kc2
                pre_mask = 1-pre_mask
            # print('acc = {}, kc = {}'.format(acc ,kc))
            if acc > max_oa and kc > max_kappa:
                max_oa = acc
                max_kappa = kc
                optim_mask = pre_mask
        print('accuracy = {}, kappa = {}'.format(max_oa, max_kappa))
        print('-------------------------------')
        # show the quality of pseudo labels
        # plt.figure()
        # plt.title('optimal preclustering map')
        # plt.imshow(optim_mask, cmap='gray')
        # plt.show()
    return optim_mask


# generate pseudo labels using CAD
def generate_pseudo_labels_CAD(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m=None, mode='BCD'):
    print('generating pseudo labels using CAD')
    [rows, cols, bands] = hsi_t1.shape
    data_t1 = hsi_t1.copy()
    data_t2 = hsi_t2.copy()
    gt_b = hsi_gt_b.copy()
    if hsi_gt_m != None:
        gt_m = hsi_gt_m.copy()
    cad = cad_M(data_t1, data_t2)
    max_oa = 0
    max_kappa = 0
    optim_mask = np.zeros([rows, cols])
    if mode == 'BCD':
        # generate binary CD map
        for i in range(1, 51): # select the optimal mask in 50 epochs
            # print('-------------------preclustering epoch {}-----------------------'.format(i))
            km = KMeans(n_clusters=2).fit(cad.reshape(-1, 1))
            pre_idx = km.labels_
            pre_mask = np.reshape(pre_idx, [rows, cols])
            acc1 = accuracy_score(gt_b.reshape(-1), pre_idx)
            kc1 = cohen_kappa_score(gt_b.reshape(-1), pre_idx)
            acc2 = accuracy_score(gt_b.reshape(-1), 1-pre_idx)
            kc2 = cohen_kappa_score(gt_b.reshape(-1), 1-pre_idx)
            if acc1 > acc2:
                acc = acc1
                kc = kc1
            else:
                acc = acc2
                kc = kc2
                pre_mask = 1-pre_mask
            # print('acc = {}, kc = {}'.format(acc ,kc))
            if acc > max_oa and kc > max_kappa:
                max_oa = acc
                max_kappa = kc
                optim_mask = pre_mask
        print('accuracy = {}, kappa = {}'.format(max_oa, max_kappa))
        print('-------------------------------')
        # show the quality of pseudo labels
        # plt.figure()
        # plt.title('optimal preclustering map')
        # plt.imshow(optim_mask, cmap='gray')
        # plt.show()
    return optim_mask

def generate_pseudo_labels_both(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m=None, mode='BCD', isStandard=False):
    print('generating pseudo labels using both CVA and CAD')
    print('-------------------------------')
    [rows, cols, bands] = hsi_t1.shape
    data_t1 = hsi_t1.copy()
    data_t2 = hsi_t2.copy()
    gt_b = hsi_gt_b.copy()
    if hsi_gt_m != None:
        gt_m = hsi_gt_m.copy()
    else:
        gt_m = hsi_gt_m
    optim_mask_cva = generate_pseudo_labels_CVA(data_t1, data_t2, gt_b, gt_m, mode, isStandard)
    # optim_mask_cva = generate_pseudo_labels_SCV(data_t1, data_t2, gt_b, gt_m, mode, isStandard)
    optim_mask_cad = generate_pseudo_labels_CAD(data_t1, data_t2, gt_b, gt_m, mode)
    optim_mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            if optim_mask_cva[i, j] == optim_mask_cad[i, j]:
                optim_mask[i, j] = optim_mask_cva[i, j]
            else:
                optim_mask[i, j] = -1
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.set_title("GT")
    plt.imshow(hsi_gt_b, cmap='gray')
    ax = plt.subplot(1, 3, 2)
    ax.set_title("CVA")
    plt.imshow(optim_mask_cva, cmap='gray')
    ax = plt.subplot(1, 3, 3)
    ax.set_title("CAD")
    plt.imshow(optim_mask_cad, cmap='gray')
    plt.show()
    return optim_mask

# patch_size should be controlled due to memory limit
# solve the category imbalance problem by selecting the same ratio samples from all categories
# form = 'same' --- samples in each category is the same --- good as unchanged samples is far more than changed samples
def generate_training_set_no_upsampling(hsi_t1, hsi_t2, hsi_gt_b, mode, patch_size, training_ratio, data_path, hsi_gt_m=None, form='same'):
    print('generating training set')
    print('-------------------------------')
    [rows, cols, bands] = hsi_t1.shape
    pseudo_labels = generate_pseudo_labels_CVA(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m, mode)
    samples = generate_training_samples(hsi_t1, hsi_t2, patch_size)
    labels = np.zeros([rows*cols]).astype(np.int64)
    for i in range(0, rows):
        for j in range(0, cols):
            index = i*cols+j
            labels[index] = pseudo_labels[i, j]
    # count num of samples in each category
    max_label = np.max(labels).astype(np.int64)
    num_per_label = np.zeros([max_label+1]).astype(np.int64)
    temp_num_per_label = np.zeros([max_label+1]).astype(np.int64)
    training_num = 0
    if form == 'same':
        for i in range(max_label+1):
            num_per_label[i] = np.sum(labels==i)
            num_per_label[i] = math.ceil(num_per_label[i]*training_ratio)
            if i == 0:
                min_val = num_per_label[i]
            else:
                min_val = min(min_val, num_per_label[i])
        training_num = min_val*(max_label+1)
        for i in range(max_label+1):
            num_per_label[i] = min_val
    else:
        for i in range(max_label+1):
            num_per_label[i] = np.sum(labels==i)
            num_per_label[i] = math.ceil(num_per_label[i]*training_ratio)
            training_num = training_num + num_per_label[i]
    test_num = rows*cols-training_num
    # generate training samples according to training_ratio*number_of_each_category
    training_num = training_num.astype(np.int64)
    test_num = test_num.astype(np.int64)
    train = np.zeros([training_num])
    test = np.zeros([test_num])
    training_samples = np.zeros([training_num, patch_size, patch_size, bands])
    training_labels = np.zeros([training_num])
    test_samples = np.zeros([test_num, patch_size, patch_size, bands])
    test_labels = np.zeros([test_num])
    train_idx = 0
    test_idx = 0
    for i in range(rows*cols):
        if temp_num_per_label[labels[i]] < num_per_label[labels[i]]:
            train[train_idx] = i+1
            training_samples[train_idx, :, :, :] = samples[i, :, :, :]
            training_labels[train_idx] = labels[i]
            temp_num_per_label[labels[i]] = temp_num_per_label[labels[i]]+1
            train_idx = train_idx+1
        else:
            test[test_idx] = i+1
            test_samples[test_idx, :, :, :] = samples[i, :, :, :]
            test_labels[test_idx] = labels[i]
            test_idx = test_idx+1
    training_samples = np.transpose(training_samples, (0, 3, 1, 2))
    test_samples = np.transpose(test_samples, (0, 3, 1, 2))
    with open(os.path.join(data_path, 'training_samples.npy'), 'bw') as outfile:
        np.save(outfile, training_samples)
    with open(os.path.join(data_path, 'training_labels.npy'), 'bw') as outfile:
        np.save(outfile, training_labels)
    with open(os.path.join(data_path, 'test_samples.npy'), 'bw') as outfile:
        np.save(outfile, test_samples)
    with open(os.path.join(data_path, 'test_labels.npy'), 'bw') as outfile:
        np.save(outfile, test_labels)
    with open(os.path.join(data_path, 'train_indexes.npy'), 'bw') as outfile:
        np.save(outfile, train)
    with open(os.path.join(data_path, 'test_indexes.npy'), 'bw') as outfile:
        np.save(outfile, test)
    return training_samples, training_labels, test_samples, test_labels

#上采样
def up_sampling(samples, labels, training_ratio, manner='random'):
    patch_size = samples.shape[2]
    bands = samples.shape[1]
    num = labels.shape[0]
    arr = np.arange(1, num+1, 1)
    np.random.shuffle(arr)
    # count num of samples in each category
    max_label = np.max(labels).astype(np.int64)
    num_per_label = np.zeros([max_label+1]).astype(np.int64)
    training_num = 0
    for i in range(max_label+1):
        num_per_label[i] = np.sum(labels==i)
        num_per_label[i] = math.ceil(num_per_label[i]*training_ratio)
        training_num = training_num + num_per_label[i]
    # print('num of training samples before upsampling: class 1 -- {}, class 2 -- {}'.format(num_per_label[0], num_per_label[1]))
    test_num = num-training_num
    training_samples = np.zeros([training_num, bands, patch_size, patch_size])
    training_labels = np.zeros([training_num])
    test_samples = np.zeros([test_num, bands, patch_size, patch_size])
    test_labels = np.zeros([test_num])
    train_idx = 0
    test_idx = 0
    for i in range(num):
        index = arr[i]-1
        label = labels[index]
        if num_per_label[label] > 0:
            training_samples[train_idx, :, :, :] = samples[index, :, :, :]
            training_labels[train_idx] = label
            train_idx = train_idx+1
        else:
            test_samples[test_idx, :, :, :] = samples[index, :, :, :]
            test_labels[test_idx] = label
            test_idx = test_idx+1
        num_per_label[label] = num_per_label[label]-1
    if manner == 'random':
        ros = RandomOverSampler()
        # ros = RandomUnderSampler()
        training_samples = training_samples.reshape([training_samples.shape[0], -1])
        training_samples, training_labels = ros.fit_resample(training_samples, training_labels)
        training_samples = training_samples.reshape([training_samples.shape[0], bands, patch_size, patch_size])
    else:
        pass # do nothing
    print('num of training samples: unchanged -- {}, change -- {}'.format(np.sum(training_labels==0), np.sum(training_labels==1)))
    return training_samples, training_labels, test_samples, test_labels


def generate_training_set(hsi_t1, hsi_t2, hsi_gt_b, mode, patch_size, training_ratio, data_path, hsi_gt_m=None, manner='random', pca=True, numComponents=64):
    print('generating training set')
    print('-------------------------------')
    [rows, cols, bands] = hsi_t1.shape
    pseudo_labels = generate_pseudo_labels_CVA(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m, mode)
    samples = generate_training_samples(hsi_t1, hsi_t2, patch_size, pca, numComponents)
    samples = np.transpose(samples, (0, 3, 1, 2))
    labels = np.zeros([rows*cols]).astype(np.int64)
    for i in range(0, rows):
        for j in range(0, cols):
            index = i*cols+j
            labels[index] = pseudo_labels[i, j]
    with open(os.path.join(data_path, 'samples.npy'), 'bw') as outfile:
        np.save(outfile, samples)
    training_samples, training_labels, test_samples, test_labels = up_sampling(samples, labels, training_ratio, manner)
    with open(os.path.join(data_path, 'training_samples.npy'), 'bw') as outfile:
        np.save(outfile, training_samples)
    with open(os.path.join(data_path, 'training_labels.npy'), 'bw') as outfile:
        np.save(outfile, training_labels)
    with open(os.path.join(data_path, 'test_samples.npy'), 'bw') as outfile:
        np.save(outfile, test_samples)
    with open(os.path.join(data_path, 'test_labels.npy'), 'bw') as outfile:
        np.save(outfile, test_labels)
    return training_samples, training_labels, test_samples, test_labels

# generate training set using both CVA and CAD
#根据 CVA 和 CAD 产生两组变化检测结果 第三章的初始化部分
def generate_training_set_both(hsi_t1, hsi_t2, hsi_gt_b, mode, patch_size, training_ratio, data_path, hsi_gt_m=None, manner='random', pca=True, numComponents=64, isStandard=False):
    print('generating training set')
    print('-------------------------------')
    gt_b = hsi_gt_b.copy()
    gt_b = gt_b.reshape([-1])
    [rows, cols, bands] = hsi_t1.shape
    pseudo_labels = generate_pseudo_labels_both(hsi_t1, hsi_t2, hsi_gt_b, hsi_gt_m, mode, isStandard)
    samples = generate_training_samples(hsi_t1, hsi_t2, patch_size, pca, numComponents)
    samples = np.transpose(samples, (0, 3, 1, 2))
    dim = samples.shape[1]
    with open(os.path.join(data_path, 'samples.npy'), 'bw') as outfile:
        np.save(outfile, samples)
    labels = np.zeros([rows*cols]).astype(np.int64)
    for i in range(0, rows):
        for j in range(0, cols):
            index = i*cols+j
            labels[index] = pseudo_labels[i, j]
    uncertain_num = np.sum(labels == -1)
    certain_num = rows*cols-uncertain_num
    print('number of total samples is {}'.format(rows*cols))
    print('number of credible samples is {}'.format(certain_num))
    certain_samples = np.zeros([certain_num, dim, patch_size, patch_size])
    uncertain_samples = np.zeros([uncertain_num, dim, patch_size, patch_size])
    certain_labels = np.zeros([certain_num]).astype(np.int64)
    count_cert = 0
    count_uncert = 0
    count = 0
    for i in range(rows*cols):
        if labels[i] == -1:
            uncertain_samples[count_uncert, :, :, :] = samples[i, :, :, :]
            count_uncert = count_uncert + 1
        else:
            if labels[i] == gt_b[i]:
                count = count + 1
            certain_samples[count_cert, :, :, :] = samples[i, :, :, :]
            certain_labels[count_cert] = labels[i]
            count_cert = count_cert + 1
    acc = count / certain_num
    print('Accuracy of the training set pseudo labels is: {}'.format(acc))
    training_samples, training_labels, test_samples, test_labels = up_sampling(certain_samples, certain_labels, training_ratio, manner)
    with open(os.path.join(data_path, 'uncertain_samples.npy'), 'bw') as outfile:
        np.save(outfile, uncertain_samples)
    with open(os.path.join(data_path, 'training_samples.npy'), 'bw') as outfile:
        np.save(outfile, training_samples)
    with open(os.path.join(data_path, 'training_labels.npy'), 'bw') as outfile:
        np.save(outfile, training_labels)
    with open(os.path.join(data_path, 'test_samples.npy'), 'bw') as outfile:
        np.save(outfile, test_samples)
    with open(os.path.join(data_path, 'test_labels.npy'), 'bw') as outfile:
        np.save(outfile, test_labels)
    print('-------------------------------')
    return training_samples, training_labels, test_samples, test_labels

def load_training_set(path):
    print('loading training set')
    print('-------------------------------')
    training_samples = np.load(os.path.join(path, 'training_samples.npy'))
    training_labels = np.load(os.path.join(path, 'training_labels.npy'))
    test_samples = np.load(os.path.join(path, 'test_samples.npy'))
    test_labels = np.load(os.path.join(path, 'test_labels.npy'))
    return training_samples, training_labels, test_samples, test_labels


def get_dataloader(training_samples, training_labels, test_samples, test_labels, batch_size, num_workers, shuffle):
    X_train = torch.from_numpy(training_samples)
    X_test = torch.from_numpy(test_samples)
    y_train = torch.from_numpy(training_labels)
    y_test = torch.from_numpy(test_labels)
    #class ChangeDetectionDataset：def __init__(self, samples, labels):
    train_dataset = ChangeDetectionDataset(X_train, y_train)
    test_dataset = ChangeDetectionDataset(X_test, y_test)
    #dataloader的封装
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader


def reports(y_p, y_t):
    y_pred = y_p.copy()
    y_true = y_t.copy()
    y_pred = y_pred.reshape([-1])
    y_true = y_true.reshape([-1])
    kc1 = cohen_kappa_score(y_true, y_pred)
    acc1 = accuracy_score(y_true, y_pred)
    kc2 = cohen_kappa_score(y_true, 1-y_pred)
    acc2 = accuracy_score(y_true, 1-y_pred)
    if acc1 > acc2:
        acc = acc1
        kc = kc1
        idx = y_pred
    else:
        acc = acc2
        kc = kc2
        idx = 1-y_pred
    return acc, kc, idx


def detect_changes_batch(model, data_path, hsi_gt_b, device, batch_size, KD_mode='NKD'): # infer batch_size samples a time
    print('detecting changes')
    samples = np.load(os.path.join(data_path, 'samples.npy'))
    model.eval()
    num = samples.shape[0]
    idx = np.zeros([num])
    count = 0
    with torch.no_grad():
        while count < num:
            batch_num = batch_size if count+batch_size <= num else num-count
            samples_batch = samples[count:count+batch_num, :, :, :]
            samples_batch = torch.from_numpy(samples_batch)
            samples_batch = samples_batch.to(device)
            samples_batch = samples_batch.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred = model(samples_batch)
            else:
                pred, __ = model(samples_batch)
            pred = Softmax()(pred).argmax(1).cpu().numpy()
            indexes = np.arange(count, count+batch_num, 1)
            idx[indexes.astype(np.int64)] = pred
            count = count+batch_num
    acc, kc, idx = reports(idx, hsi_gt_b)
    mask = idx.reshape(hsi_gt_b.shape)
    print('accuracy = {}, kappa = {}'.format(acc, kc))
    print('-------------------------------')
    return acc, kc, mask


def draw_curve(data, label, xlabel, ylabel, line, file_path=None, file_name=None, new_figure=True):
    epochs = range(len(data))
    if new_figure == True:
        plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(epochs, data, line, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    if file_name != None:
        plt.savefig(os.path.join(file_path, file_name))

def draw_curve_student(loss_list, train_DKD_list, low_rank_loss_list, file_path=None, file_name=None):
    epochs = range(len(loss_list))
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # loss_list_cpu = loss_list.cpu().numpy()
    plt.plot(epochs, loss_list, 'b', label='Total loss')
    plt.plot(epochs, train_DKD_list, 'k', label='Hard loss')
    plt.plot(epochs, low_rank_loss_list, 'r', label='Low rank loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if file_name != None:
        plt.savefig(os.path.join(file_path, file_name))


def save(data, file_name):
    with open(file_name, 'bw') as outfile:
        np.save(outfile, data)

# detect changes using the student model trained by SimKD
def detect_changes_batch_SimKD(module_list, data_path, hsi_gt_b, device, batch_size, KD_mode='NKD', model_name='vgg'): # infer batch_size samples a time
    print('detecting changes')
    samples = np.load(os.path.join(data_path, 'samples.npy'))
    for module in module_list:
        module.eval()
    model_stu = module_list[0]
    model_tea = module_list[-1]
    num = samples.shape[0]
    idx = np.zeros([num])
    count = 0
    with torch.no_grad():
        while count < num:
            batch_num = batch_size if count+batch_size <= num else num-count
            samples_batch = samples[count:count+batch_num, :, :, :]
            samples_batch = torch.from_numpy(samples_batch)
            samples_batch = samples_batch.to(device)
            samples_batch = samples_batch.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(samples_batch)
                pred_stu = model_stu(samples_batch)
            else:
                __, fea_tea = model_tea(samples_batch)
                __, fea_stu = model_stu(samples_batch)

            if model_name == 'vgg':
                cls_t = model_tea.classifier
            elif model_name == 'resnet':
                cls_t = model_tea.fc

            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](fea_stu, fea_tea, cls_t)
            logits_s = pred_feat_s
            pred = Softmax()(logits_s).argmax(1).cpu().numpy()
            indexes = np.arange(count, count+batch_num, 1)
            idx[indexes.astype(np.int64)] = pred
            count = count+batch_num
    acc, kc, idx = reports(idx, hsi_gt_b)
    mask = idx.reshape(hsi_gt_b.shape)
    print('accuracy = {}, kappa = {}'.format(acc, kc))
    print('-------------------------------')
    return acc, kc, mask

# draw convergence curve
def draw_convergence(dataset, model):
    plt.figure()
    path = os.path.join(os.path.join('result', os.path.join(dataset, model)), 'loss.mat')
    loss = sio.loadmat(path)['train_loss']
    plt.rcParams['axes.facecolor'] = 'snow'
    plt.rc('font', family='Times New Roman', size=20)
    loss = np.reshape(loss, [-1])
    x = range(1, 101)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x, loss, color='b')
    plt.savefig('convergence_{}_{}.png'.format(dataset, model), bbox_inches='tight')


# detect changes using the student model trained by SRRL
def detect_changes_batch_SRRL(module_list, data_path, hsi_gt_b, device, batch_size, KD_mode='NKD', model_name='vgg'): # infer batch_size samples a time
    print('detecting changes')
    samples = np.load(os.path.join(data_path, 'samples.npy'))
    for module in module_list:
        module.eval()
    model_stu = module_list[0]
    model_tea = module_list[-1]
    num = samples.shape[0]
    idx = np.zeros([num])
    count = 0
    with torch.no_grad():
        while count < num:
            batch_num = batch_size if count+batch_size <= num else num-count
            samples_batch = samples[count:count+batch_num, :, :, :]
            samples_batch = torch.from_numpy(samples_batch)
            samples_batch = samples_batch.to(device)
            samples_batch = samples_batch.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(samples_batch)
                pred_stu = model_stu(samples_batch)
            else:
                __, fea_tea = model_tea(samples_batch)
                __, fea_stu = model_stu(samples_batch)

            if model_name == 'vgg':
                cls_t = model_tea.classifier
            elif model_name == 'resnet':
                cls_t = model_tea.fc

            trans_feat_s, pred_feat_s = module_list[1](fea_stu, cls_t)
            logits_s = pred_feat_s
            pred = Softmax()(logits_s).argmax(1).cpu().numpy()
            indexes = np.arange(count, count+batch_num, 1)
            idx[indexes.astype(np.int64)] = pred
            count = count+batch_num
    acc, kc, idx = reports(idx, hsi_gt_b)
    mask = idx.reshape(hsi_gt_b.shape)
    print('accuracy = {}, kappa = {}'.format(acc, kc))
    print('-------------------------------')
    return acc, kc, mask