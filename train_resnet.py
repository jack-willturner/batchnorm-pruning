import argparse
import json
import os

import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo
from tensorboardX import SummaryWriter

from models import *
from utils  import *

import time
writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Initial training script')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--save_file', default='resnet', type=str, help='save file for checkpoints')
parser.add_argument('--base_file', default='bbb', type=str, help='base file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')


# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=512, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-epochs', '--no_epochs', default=24, type=int, metavar='epochs', help='no. epochs')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet18()
if torch.cuda.device_count > 1:
  print('Using multiple GPUs')
  model = nn.DataParallel(model)

model.to(device)

train_set_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_set_raw  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

t = Timer()
train_set = list(zip(transpose(normalise(pad(train_set_raw.train_data, 4))), train_set_raw.train_labels))
test_set  = list(zip(transpose(normalise(test_set_raw.test_data)), test_set_raw.test_labels))

error_history = []
epoch_step = json.loads(args.epoch_step)

def train():
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)


def validate():
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(valloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        loss.backward()
        model.zero_grad()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(valloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_top1', top1.avg, epoch)
    writer.add_scalar('val_top5', top5.avg, epoch)

    # Record Top 1 for CIFAR
    error_history.append(top1.avg)


def adjust_opt(optimizer, epoch):
    knots, vals = [0, 5, 24], [0, 0.4, 0]
    lr = np.interp(epoch, knots, vals)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    filename = 'checkpoints/%s.t7' % args.save_file
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1,momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, args.no_epochs+1):
        adjust_opt(optimizer, epoch)

        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])

        # train for one epoch
        train()
        # # evaluate on validation set
        validate()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'error_history': error_history,
        }, filename=filename)

