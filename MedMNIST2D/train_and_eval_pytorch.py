import os
import argparse
import time
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict
from models import ResNet18, ResNet50

from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST
from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO


def main(flag, input_root, output_root, end_epoch, gpu_ids, batch_size, download):

    flag_to_class2d = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organamnist": OrganAMNIST,
        "organcmnist": OrganCMNIST,
        "organsmnist": OrganSMNIST,
        "bloodmnist": BloodMNIST,
        "tissuemnist": TissueMNIST}

    lr = 0.001
    n_epochs = end_epoch
    gamma=0.1
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]

    info = INFO[flag]
    n_classes = len(info["label"])
    task = info["task"]
    DataClass = flag_to_class2d[flag]

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
    dir_path = os.path.join(output_root, flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])
     
    train_dataset = DataClass(root=input_root, 
                                split='train', 
                                transform=data_transform, 
                                download=download)
    val_dataset = DataClass(root=input_root, 
                                split='val', 
                                transform=data_transform, 
                                download=download)
    test_dataset = DataClass(root=input_root, 
                                split='test', 
                                transform=data_transform, 
                                download=download)
    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    print('==> Building and training model...')
    
    model = ResNet18(in_channels=3, num_classes=n_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(dir_path, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    global iteration
    iteration = 0

    log = '%s\n' % (flag)
    
    for epoch in trange(n_epochs):        
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_loader, task, criterion, device)
        val_metrics = test(model, val_loader, task, criterion, device)
        test_metrics = test(model, test_loader, task, criterion, device)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

            state = {
                'net': model.state_dict(),
            }

            path = os.path.join(dir_path, 'epoch_%d_model.pth' % (epoch))
            torch.save(state, path)

            train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
            val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
            test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

            log = log + '%s\n' % (epoch) + train_log + val_log + test_log + '\n'

            print('train AUC: %.5f ACC: %.5f' % (train_metrics[1], train_metrics[2]))
            print('val AUC: %.5f ACC: %.5f' % (val_metrics[1], val_metrics[2]))
            print('test AUC: %.5f ACC: %.5f' % (test_metrics[1], test_metrics[2]))
            
    with open(os.path.join(os.path.join(output_root, flag), '%s_log.txt' % (flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, task, criterion, device):

    model.eval()
    
    total_loss = []
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")

    args = parser.parse_args()
    flag = args.flag
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    
    main(flag, input_root, output_root, end_epoch, gpu_ids, batch_size, download)
