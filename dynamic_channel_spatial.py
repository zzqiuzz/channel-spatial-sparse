# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils.time import convert_secs2time, time_string, time_file_str
from utils.profile import *
import utils.globalvar as gvar
#from models import print_log
import models
import random
import numpy as np
from tensorboardX import SummaryWriter
from torchsummary import summary
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default="/data/ilsvrc12_torch",
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./temp', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dynamicresnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: dynamicresnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 12)')
parser.add_argument('--gpu', type=str, metavar='gpuid', help='gpu.')
parser.add_argument('--epochs', default=70, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume_normal', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_from', default='', type=str, metavar='PATH', help='path to pretrained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--extract',action='store_true',help='extract features.')
parser.add_argument('--pretrained',action='store_true',help='use pretrained model.')
parser.add_argument('--show',action='store_true',help='show model architecture.')
parser.add_argument('--flops',action='store_true',help='calc flops given a pretrained model.')
parser.add_argument('--debug',action='store_true',help='debug.')
parser.add_argument('--channel_removed_ratio',default=0.2,type=float,help='removed ratio.')
parser.add_argument('--spatial_removed_ratio',default=0.1,type=float,help='removed ratio.')
parser.add_argument('--Is_spatial',action='store_true',help='use spatial module or not,default is channel with conv.')
parser.add_argument('--lasso',action='store_true',help='add l1 regularization to channel module.')
parser.add_argument('--l1_coe',default=1e-8,type=float,help='coe of l1 regularization.')
parser.add_argument('--sep_wd',action='store_true',help='seprate weight decay.')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()
gvar._init()
gvar.set_value('removed_ratio_c',args.channel_removed_ratio)
gvar.set_value('removed_ratio_s',args.spatial_removed_ratio)
gvar.set_value('is_spatial',args.Is_spatial)
gvar.set_value('lasso',args.lasso)
def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
    	os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch,args.prefix)), 'w')
    if args.pretrained:
        gvar.set_value('log',log)
    # version information

    print_log("Using  GPUs : {}".format(str(args.gpu)), log)
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=args.pretrained)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    if args.debug:
        #for k,v in model.named_modules():
        all_parameters = model.parameters()
        weight_parameters = []
        for name,value in model.named_parameters():
            if 'dynamic_channel' in name:
                weight_parameters.append(value)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
            #print(k)
        return
    if args.show:
        input_data = torch.randn([1,3,224,224])
        #summary(model.cuda(),(3,224,224))
        #model = model.cpu()
        with SummaryWriter(log_dir='./log',comment='resnet18') as w:
            w.add_graph(model,(input_data))
        return 
    if args.flops:
        input_data = torch.randn([1,3,224,224])
        flops, params = profile(model,inputs=(input_data, ))
        print(flops)
        print("flops,:{},params:{}".format(clever_format(flops), params))
        return  
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      model.features = torch.nn.DataParallel(model.features)
      model.cuda()
    else:
      model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.sep_wd:
        all_parameters = model.parameters()
        dynamic_parameters = []
        for name,value in model.named_parameters():
            if 'dynamic_channel' in name:
                dynamic_parameters.append(value)
        dynamic_parameters_id = list(map(id, dynamic_parameters))
        backbone_parameters = list(filter(lambda p: id(p) not in dynamic_parameters_id, all_parameters))
        optimizer = torch.optim.SGD([{'params': backbone_parameters},
                                    {'params': dynamic_parameters,'weight_decay': 1e-8}],
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:   
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, 
                                                     args.epochs//2, args.epochs//4*3], gamma=0.1)
    # optionally resume from a checkpoint
    if args.resume_normal:
        if os.path.isfile(args.resume_normal):
            print_log("=> loading checkpoint '{}'".format(args.resume_normal), log)
            checkpoint = torch.load(args.resume_normal)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume_normal, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume_normal), log)
    elif args.resume_from: # increse channel_removed_ratio as FBS
        if os.path.isfile(args.resume_from):
            if not args.lasso:
            	print_log("=> loading pretrained model '{}'".format(args.resume_from), log)
            	print_log("=> increase channel removed ratio to '{}'".format(args.channel_removed_ratio), log)
            	checkpoint = torch.load(args.resume_from)
            	args.start_epoch = 0
            	model.load_state_dict(checkpoint['state_dict'])
            	print_log("=> loaded pretrained model '{}' (epoch {})".format(args.resume_from, args.start_epoch), log)
            elif args.lasso:
                print_log("=> loading pretrained model '{}'".format(args.resume_from), log)
                print_log("=> increase channel removed ratio to '{}'".format(args.channel_removed_ratio), log)
                checkpoint = torch.load(args.resume_from)
                args.start_epoch = 0
                oldmodel = checkpoint['state_dict']
                #for k,v in oldmodel.items():
                #    print(k)
                for key,value in model.state_dict().items():
                    if "channel_l1" in key:
                        continue
                    if "spatial_l1" in key:
                        continue
                    value.copy_(oldmodel[key])
                print_log("=> loaded pretrained model '{}' (epoch {})".format(args.resume_from, args.start_epoch), log)
                #return
    cudnn.benchmark = True
    for epoch in range(args.start_epoch):
        scheduler.step()
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.extract:
        extract(val_loader,model)
        return 
        
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    
    if args.evaluate:
        validate(val_loader, model, criterion,log)
        return

    filename = os.path.join(args.save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch,log)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(args.arch, epoch, args.epochs, time_string(), need_time), log)
        scheduler.step()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log)
        # evaluate on validation set
        val_acc_2 = validate(val_loader, model, criterion, log)
        
        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #l1_loss = torch.new_zeros(0,requires_grad=True)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target.cuda(async=True)
        #input_var = torch.autograd.Variable(input)
        #target_var = torch.autograd.Variable(target)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        ## compute output
        output = model(input)
        loss = criterion(output, target)
        ####add L1 regularization
        if args.lasso:
        	for _, m in model.named_modules():
        	    if hasattr(m,"channel_l1"):
        	        #l1_loss += m.channel_predictor.cpu()
        	        loss += args.l1_coe * m.channel_l1#.squeeze(0)
        	    if hasattr(m,"spatial_l1"):
        	        #l1_loss += m.channel_predictor.cpu()
        	        loss += args.l1_coe * m.spatial_l1#.squeeze(0)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5), log)

def get_activation(indata,outdata,inshape,outshape):
    def hook(model,input,output):
        outdata.append(output.detach().cpu().numpy())
        indata.append(input[0].detach().cpu().numpy())
        inshape.append(input[0].shape)
        outshape.append(output.shape)
    return hook
def extract(val_loader,model):
    model.eval()
    in_data = []
    activation = []
    in_shape = []
    out_shape = []
    weight = []
    #layer_name = "layer2_0_conv1"
    layer_name = "lasso-e6_layer1.1_dynamicblock"
    outfileName = layer_name + ".npz" 
    for i,data in enumerate(val_loader):
        input = data[0].cuda()
        model.module.layer1[1].dynamicblock1.dynamic_channel.register_forward_hook(get_activation(in_data,activation,in_shape,out_shape))
        output = model(input)
        np.savez(outfileName,name=layer_name,in_data=in_data,feature=activation,in_shape=in_shape,out_shape=out_shape)
        print("Input & Output extracted.")
        #for _,m in enumerate(model.named_modules()):
        #    if m[0] == "module.layer1.1.bn2": #"module.layer2.0.conv1":
        #        for param in m[1].parameters():
        #            weight = param.data.detach().cpu().numpy()
        #            np.savez(outfileName,name=layer_name,in_data=in_data,feature=activation,in_shape=in_shape,out_shape=out_shape,weight=weight)
        #print("weights saved.")
        break
        
def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            #target = target.cuda(async=True)
            #input_var = torch.autograd.Variable(input, volatile=True)
            #target_var = torch.autograd.Variable(target, volatile=True)
            input = input.cuda(non_blocking=True) 
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)
  
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
  
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
  
            if i % args.print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     i, len(val_loader), batch_time=batch_time, loss=losses,
                     top1=top1, top5=top5), log)
  
        print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()
  
  
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch,log):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print_log("learning rate @ {} is '{}')".format(epoch,lr), log)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
