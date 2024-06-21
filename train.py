import torch
from torch.nn import Softmax
from tqdm import tqdm

# training process of the teacher model
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    total_acc = 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)

        # Compute the prediction error
        pred = model(X)
        loss = loss_fn(Softmax()(pred), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)
        total_acc = total_acc + (Softmax()(pred).argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    total_acc /= size
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, total_acc

# test process of the teacher model
def test(dataloader, model, loss_fn, device, KD_mode='NKD'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred = model(X)
            else:
                pred, __ = model(X)
            test_loss += loss_fn(Softmax()(pred), y).item()
            correct += (Softmax()(pred).argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def train_student_igkd(dataloader, model_tea, model_stu, loss_fn, optimizer, device, temp, KD_mode='NKD', loss_lr=None, lr_reg=None):
    size = len(dataloader.dataset)
    model_tea.eval()
    model_tea.cuda()
    model_stu.train()
    model_stu.cuda()
    total_loss, total_hard_loss, total_soft_loss, total_low_rank_loss, total_acc = 0, 0, 0, 0, 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)

        # Compute the prediction error
        if KD_mode == 'NKD':
            pred_tea = model_tea(X)
            pred_stu = model_stu(X)
        else:
            pred_tea, fea_tea = model_tea(X)
            pred_stu, fea_stu = model_stu(X)
            #print(fea_tea.shape) #([128, 512, 7, 7])
            #print(fea_stu.shape) #([128, 256, 7, 7])
            #import time
            #time.sleep(1000)
            loss_low_rank = 0
            #loss_low_rank = loss_lr(fea_tea, fea_stu)

        # loss computed by soft labels
        soft_logits_tea = Softmax()(pred_tea * 1.0/temp)
        soft_logits_stu = Softmax()(pred_stu * 1.0/temp)
        loss_soft = torch.nn.BCEWithLogitsLoss().cuda()(soft_logits_stu, soft_logits_tea)

        # loss computed by hard labels
        loss_hard = loss_fn(Softmax()(pred_stu), y)

        # total loss
        reg = temp*temp
        reg = 1.0/reg
        loss = reg*loss_hard + loss_soft
        if KD_mode == 'KD':
            loss = loss + lr_reg*loss_low_rank

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)
        total_hard_loss = total_hard_loss + reg*loss_hard.item() / len(dataloader)
        total_soft_loss = total_soft_loss + loss_soft.item() / len(dataloader)
        if KD_mode == 'KD':
            total_low_rank_loss = 0
            #total_low_rank_loss = total_low_rank_loss + lr_reg*loss_low_rank.item() / len(dataloader)
        total_acc = total_acc + (Softmax()(pred_stu).argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            loss_hard, loss_soft = loss_hard.item(), loss_soft.item()
            if KD_mode == 'KD':
                loss_low_rank = 0
                # loss_low_rank = loss_low_rank.item()
                print(f"total loss: {loss:>7f} hard loss: {reg*loss_hard:>7f} soft loss:  {loss_soft:>7f} low rank loss: {lr_reg*loss_low_rank:>7f} [{current:>5d}/{size:>5d}]")
            else:
                print(f"total loss: {loss:>7f} hard loss: {reg*loss_hard:>7f} soft loss:  {loss_soft:>7f} [{current:>5d}/{size:>5d}]")
    total_acc /= size
    if KD_mode == 'KD':
        print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n"
              f"Avg hard loss: {total_hard_loss:>8f}, Avg soft loss: {total_soft_loss:>8f}, Avg low rank loss: {total_low_rank_loss:>8f}")
    else:
        print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n"
              f"Avg hard loss: {total_hard_loss:>8f}, Avg soft loss: {total_soft_loss:>8f}")
    return total_loss, total_hard_loss, total_soft_loss, total_low_rank_loss, total_acc


def test_student_igkd(dataloader, model_tea, model_stu, loss_fn, device, temp, KD_mode='NKD', loss_lr=None, lr_reg=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_tea.eval()
    model_stu.eval()
    test_hard_loss, test_soft_loss, test_low_rank_loss, test_loss, correct = 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(X)
                pred_stu = model_stu(X)
            else:
                pred_tea, fea_tea = model_tea(X)
                pred_stu, fea_stu = model_stu(X)
                loss_low_rank = loss_lr(fea_tea, fea_stu)

            # loss computed by soft labels
            soft_logits_tea = Softmax()(pred_tea * 1.0/temp)
            soft_logits_stu = Softmax()(pred_stu * 1.0/temp)
            loss_soft = torch.nn.BCEWithLogitsLoss().cuda()(soft_logits_stu, soft_logits_tea)

            # loss computed by hard labels
            loss_hard = loss_fn(Softmax()(pred_stu), y)

            # total loss
            reg = temp*temp
            reg = 1.0/reg
            test_loss += (reg*loss_hard.item() + loss_soft.item())
            if KD_mode == 'KD':
                test_loss += lr_reg*loss_low_rank.item()

            test_hard_loss += reg*loss_hard.item()
            test_soft_loss += loss_soft.item()
            test_low_rank_loss += lr_reg*loss_low_rank.item()
            correct += (Softmax()(pred_stu).argmax(1) == y).type(torch.float).sum().item()

    test_hard_loss /= num_batches
    test_soft_loss /= num_batches
    test_low_rank_loss /= num_batches
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg total loss: {test_loss:>8f} \n"
          f"Avg hard loss: {test_hard_loss:>8f}, Avg soft loss: {test_soft_loss:>8f}, Avg low rank loss: {test_low_rank_loss:>8f}")
    return test_loss, test_hard_loss, test_soft_loss, test_low_rank_loss, correct


# DKD
def train_student(dataloader, model_tea, model_stu, loss_fn, optimizer, device, temp, KD_mode='NKD', lr_reg=None):
    size = len(dataloader.dataset)
    model_tea.eval()
    model_tea.cuda()
    model_stu.train()
    model_stu.cuda()
    # total_loss, total_hard_loss, total_soft_loss, total_low_rank_loss, total_acc = 0, 0, 0, 0, 0
    total_loss, total_DKD_loss, total_low_rank_loss, total_acc=0, 0, 0, 0
    for batch, (X,y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)

        # Compute the prediction error

        pred_tea, fea_tea = model_tea(X)
        pred_stu, fea_stu = model_stu(X)
        #loss_low_rank = loss_lr(fea_tea, fea_stu)
        
        #pred_stu和pred_tea是logits！
        loss_DKD = loss_fn(pred_stu, pred_tea, y, 1, 8, temp)
        # loss computed by soft labels
        # soft_logits_tea = Softmax()(pred_tea * 1.0/temp)
        # soft_logits_stu = Softmax()(pred_stu * 1.0/temp)
        # loss_soft = torch.nn.BCEWithLogitsLoss().cuda()(soft_logits_stu, soft_logits_tea)

        # # loss computed by hard labels
        # loss_hard = loss_fn(Softmax()(pred_stu), y)

        # total loss
        # reg = temp*temp
        # reg = 1.0/reg
        # loss = reg*loss_hard + loss_soft

        #total_loss
        loss = loss_DKD

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)

        total_DKD_loss = total_DKD_loss + loss_DKD.item() / len(dataloader)
        # total_hard_loss = total_hard_loss + reg*loss_hard.item() / len(dataloader)
        # total_soft_loss = total_soft_loss + loss_soft.item() / len(dataloader)

        total_acc = total_acc + (Softmax()(pred_stu).argmax(1) == y).type(torch.float).sum().item()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     loss_DKD = loss_DKD.item()
        #     # loss_hard, loss_soft = loss_hard.item(), loss_soft.item()
        #     loss_low_rank = loss_low_rank.item()

        #     print(f"total loss: {loss:>7f} loss_DKD: {loss_DKD:>7f} low rank loss: {lr_reg*loss_low_rank:>7f} [{current:>5d}/{size:>5d}]")
    total_acc /= size
    #:>8f表示字段宽度为8，右对齐，转换为浮点数
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n"
            f"Avg DKD loss: {total_DKD_loss:>8f}, Avg low rank loss: {total_low_rank_loss:>8f}")

    return total_loss, total_DKD_loss, total_low_rank_loss, total_acc


# test process of the student model
def test_student(dataloader, model_tea, model_stu, loss_fn, device, temp, KD_mode='NKD', lr_reg=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_tea.eval()
    model_stu.eval()
    test_low_rank_loss, test_DKD_loss, test_loss, correct = 0, 0, 0, 0

    with torch.no_grad():
        for batch, (X,y) in enumerate(tqdm(dataloader)):
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)

            pred_tea, fea_tea = model_tea(X)
            pred_stu, fea_stu = model_stu(X)

            # loss computed by soft labels
            # soft_logits_tea = Softmax()(pred_tea * 1.0/temp)
            # soft_logits_stu = Softmax()(pred_stu * 1.0/temp)
            # loss_soft = torch.nn.BCEWithLogitsLoss().cuda()(soft_logits_stu, soft_logits_tea)

            # # loss computed by hard labels
            # loss_hard = loss_fn(Softmax()(pred_stu), y)

            loss_DKD = loss_fn(pred_stu, pred_tea, y, 1, 8, temp)


            test_loss += loss_DKD
            # test_hard_loss += reg*loss_hard.item()
            # test_soft_loss += loss_soft.item()
            test_DKD_loss += loss_DKD.item()
            correct += (Softmax()(pred_stu).argmax(1) == y).type(torch.float).sum().item()

    test_DKD_loss /= num_batches
    test_low_rank_loss /= num_batches
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg total loss: {test_loss:>8f} \n"
          f"Avg DKD loss: {test_DKD_loss:>8f},  Avg low rank loss: {test_low_rank_loss:>8f}")
    return test_loss, test_DKD_loss, test_low_rank_loss, correct


# SimKD -- "Knowledge Distillation with the Reused Teacher Classifier" -- CVPR 2022
def train_student_SimKD(dataloader, module_list, criterion_list, optimizer, device, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_kd = criterion_list[0]
    model_stu = module_list[0]
    model_tea = module_list[-1]
    total_loss, total_acc = 0, 0
    for batch, (X,y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)

        # Compute the prediction error
        if KD_mode == 'NKD':
            pred_tea = model_tea(X)
            pred_stu = model_stu(X)
        else:
            pred_tea, fea_tea = model_tea(X)
            pred_stu, fea_stu = model_stu(X)

        if model_name == 'vgg':
            cls_t = model_tea.classifier
        elif model_name == 'resnet':
            cls_t = model_tea.fc
        trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](fea_stu, fea_tea, cls_t)
        logits_s = pred_feat_s
        loss_kd = criterion_kd(trans_feat_s, trans_feat_t)

        loss = loss_kd

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)
        total_acc = total_acc + (Softmax()(logits_s).argmax(1) == y).type(torch.float).sum().item()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"total loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    total_acc /= size
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n")
    return total_loss, total_acc


# SimKD -- "Knowledge Distillation with the Reused Teacher Classifier" -- CVPR 2022
def test_student_SimKD(dataloader, module_list, criterion, device, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    model_stu = module_list[0]
    model_tea = module_list[-1]
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(X)
                pred_stu = model_stu(X)
            else:
                pred_tea, fea_tea = model_tea(X)
                pred_stu, fea_stu = model_stu(X)

            if model_name == 'vgg':
                cls_t = model_tea.classifier
            elif model_name == 'resnet':
                cls_t = model_tea.fc

            __, __, output = module_list[1](fea_stu, fea_tea, cls_t)

            loss = criterion(output, y)

            # total loss
            test_loss += loss.item()
            correct += (Softmax()(output).argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg total loss: {test_loss:>8f} \n")
    return test_loss, correct


# SemCKD -- "Cross-Layer Distillation with Semantic Calibration" -- AAAI 2021
def train_student_SemCKD(dataloader, module_list, criterion_list, optimizer, device, batch_size, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_kd = criterion_list[0]
    criterion_cls = criterion_list[1]
    criterion_div = criterion_list[2]
    model_stu = module_list[0]
    model_tea = module_list[-1]
    total_loss, total_acc = 0, 0
    for batch, (X,y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)
        if X.shape[0] < batch_size:
            continue
        # Compute the prediction error
        if KD_mode == 'NKD':
            pred_tea = model_tea(X)
            pred_stu = model_stu(X)
        else:
            pred_tea, fea_tea = model_tea(X)
            pred_stu, fea_stu = model_stu(X)

        cls = 1
        div = 0.001
        k_d = 0.001
        s_value, f_target, weight = module_list[1](fea_stu, fea_tea)
        loss_kd = criterion_kd(s_value, f_target, weight)
        loss_cls = criterion_cls(Softmax()(pred_stu), y)
        loss_div = criterion_div(pred_stu, pred_tea)
        loss = cls*loss_cls + div*loss_div + k_d*loss_kd

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)
        total_acc = total_acc + (Softmax()(pred_stu).argmax(1) == y).type(torch.float).sum().item()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"total loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    total_acc /= size
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n")
    return total_loss, total_acc

# SemCKD -- "Cross-Layer Distillation with Semantic Calibration" -- AAAI 2021
def test_student_SemCKD(dataloader, module_list, criterion, device, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    model_stu = module_list[0]
    model_tea = module_list[-1]
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(X)
                pred_stu = model_stu(X)
            else:
                pred_tea, fea_tea = model_tea(X)
                pred_stu, fea_stu = model_stu(X)

            loss = criterion(pred_stu, y)
            output = pred_stu

            # total loss
            test_loss += loss.item()
            correct += (Softmax()(output).argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg total loss: {test_loss:>8f} \n")
    return test_loss, correct

# SRRL -- "KNOWLEDGE DISTILLATION VIA SOFTMAX REGRESSION REPRESENTATION LEARNING" -- ICLR 2021
def train_student_SRRL(dataloader, module_list, criterion_list, optimizer, device, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]
    model_stu = module_list[0]
    model_tea = module_list[-1]
    total_loss, total_acc = 0, 0
    for batch, (X,y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)

        # Compute the prediction error
        if KD_mode == 'NKD':
            pred_tea = model_tea(X)
            pred_stu = model_stu(X)
        else:
            pred_tea, fea_tea = model_tea(X)
            pred_stu, fea_stu = model_stu(X)

        if model_name == 'vgg':
            cls_t = model_tea.classifier
        elif model_name == 'resnet':
            cls_t = model_tea.fc

        trans_feat_s, pred_feat_s = module_list[1](fea_stu, cls_t)
        loss_kd = criterion_kd(trans_feat_s, fea_tea) + criterion_kd(pred_feat_s, pred_tea)
        logits_s = pred_feat_s
        loss_cls = criterion_cls(Softmax()(logits_s), y)

        loss = loss_kd + loss_cls

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() / len(dataloader)
        total_acc = total_acc + (Softmax()(logits_s).argmax(1) == y).type(torch.float).sum().item()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"total loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    total_acc /= size
    print(f"Train Error: \n Accuracy: {(100*total_acc):>0.1f}%, Avg total loss: {total_loss:>8f} \n")
    return total_loss, total_acc


# SRRL -- "KNOWLEDGE DISTILLATION VIA SOFTMAX REGRESSION REPRESENTATION LEARNING" -- ICLR 2021
def test_student_SRRL(dataloader, module_list, criterion, device, KD_mode='NKD', model_name='vgg'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    model_stu = module_list[0]
    model_tea = module_list[-1]
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            if KD_mode == 'NKD':
                pred_tea = model_tea(X)
                pred_stu = model_stu(X)
            else:
                pred_tea, fea_tea = model_tea(X)
                pred_stu, fea_stu = model_stu(X)

            if model_name == 'vgg':
                cls_t = model_tea.classifier
            elif model_name == 'resnet':
                cls_t = model_tea.fc

            trans_feat_s, pred_feat_s = module_list[1](fea_stu, cls_t)
            loss = criterion(Softmax()(pred_feat_s), y)

            # total loss
            test_loss += loss.item()
            correct += (Softmax()(pred_feat_s).argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg total loss: {test_loss:>8f} \n")
    return test_loss, correct
