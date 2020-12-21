import os
import torch
import torchvision
import torch.nn as nn
from models import *
from utils import progress_bar
import argparse


parser = argparse.ArgumentParser(description='PyTorch Evaluation')
parser.add_argument('--model', default="ckpt_1", type=str, help= 'model name to evaluate')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def test(net, testloader, epoch):
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

	return acc


from dataset import testset

testloader = torch.utils.data.DataLoader(
		testset, batch_size=100, shuffle=False, num_workers=2)


net = VGG('VGG11')

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

checkpoint = torch.load('./checkpoint/%s.pth' % (args.model))

net.load_state_dict(checkpoint['net'])

best_acc = checkpoint['acc']

start_epoch = checkpoint['epoch']

net = net.cuda()


test(net, testloader, 100)