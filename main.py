import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse

from models.vgg import *
from utils import progress_bar

import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from dataset import testset, trainset_1, trainset_2, trainset_4, trainset_8


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--nodes', default=8, type=int, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 100

logs = 'runs/runs_%d_bs_%d' % (args.nodes, args.bs)


def average_model(model):
    """ Model averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


# Training
def train(net, trainloader, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs, targets)
        inputs, targets = inputs[0].to(device), targets[0].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(net, testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs[0].to(device), targets[0].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_%d.pth' % args.nodes)
        best_acc = acc

    return acc, test_loss/(batch_idx+1)


def run(rank, size):

    # torch.manual_seed(1234)

    rank = dist.get_rank()

    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logs)

    # Data
    print('==> Preparing data..')

    if size == 1:
        trainset = trainset_1
    elif size == 2:
        trainset = trainset_2
    elif size == 4:
        trainset = trainset_4
    elif size == 8:
        trainset = trainset_8
    else:
        print("Number of clients not supported yet. Modify dataset.py to conduct dataset splitting for more clients.")
        return

    trainloader = torch.utils.data.DataLoader(
        trainset[rank], batch_size=args.bs, shuffle=True, num_workers=2)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = VGG('VGG11')
    net = net.cuda()


    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    for epoch in range(start_epoch, start_epoch+epochs):
        train(net, trainloader, optimizer, epoch)
        acc, loss = test(net, testloader, epoch)

        dist.barrier()
        average_model(net)

        if dist.get_rank() == 0:
            print("after averaging: ")
            acc, loss = test(net, testloader, epoch)
            sys.stdout.flush()
            writer.add_scalar("test_accuracy", acc, epoch)
            writer.add_scalar("test_loss", loss, epoch)

        scheduler.step()


def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



if __name__ == "__main__":

    mp.set_start_method('spawn')

    size = args.nodes

    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

