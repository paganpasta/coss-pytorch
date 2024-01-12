#!/usr/bin/env python
import os
import time
import json
import torch.optim
import seed.builder
import torch.nn.parallel
import seed.models as models
import torch.distributed as dist
from tools.opts import parse_opt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from tools.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from wrapper import Wrapper
from tools.utils import simclr_aug, mocov1_aug, mocov2_aug, swav_aug, adjust_learning_rate, \
     soft_cross_entropy,  AverageMeter, ValueMeter, ProgressMeter, resume_training, \
     load_simclr_teacher_encoder, load_moco_teacher_encoder, load_swav_teacher_encoder, save_checkpoint
from losses import coss, dino, dinoss, dist
from torch.utils.data import Subset
import torch.nn.functional as F


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

AUGMENTATIONS = {'swav': swav_aug, 'moco_v1': mocov1_aug, 'moco_v2':mocov2_aug, 'simclr': simclr_aug}
LOSSES = {'coss': coss, 'dino':dino, 'dinoss':dinoss, 'dist': dist}

def main(args):

    # set-up the output directory
    os.makedirs(args.output, exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

        # create logger
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(),
                              color=False, name=args.loss)

        if dist.get_rank() == 0:
            path = os.path.join(args.output, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            logger.info("Full config saved to {}".format(path))

        # save the distributed node machine
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    else:
        # create logger
        logger = setup_logger(output=args.output, color=False, name=args.loss)
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        logger.info('Single GPU mode for debugging.')

    # create model
    logger.info("=> creating student encoder '{}'".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'".format(args.teacher_arch))

    # some architectures are not supported yet. It needs to be expanded manually.
    t_model = models.__dict__[args.teacher_arch]()
    if 'clip' not in args.teacher_arch:
        from collections import OrderedDict
        new_tdict = OrderedDict()
        tdict = torch.load(args.distill)['state_dict']
        for k,v in tdict.items():
            if 'module.' in k:
                new_tdict[k[len('module.'):]] = v
            else:
                new_tdict[k] = v
        msg = t_model.load_state_dict(new_tdict, strict=False)
        print(msg)
    else:
        print('NO WEIGHTS TO LOAD FOR TEACHER. Pretrained loaded directly for CLIP.')
    
    model = Wrapper(models.__dict__[args.student_arch](), t_model, 'clip' in args.teacher_arch)
    logger.info(t_model)
    logger.info(model)
    
    for name, param in t_model.named_parameters():
        param.requires_grad = False

    if args.distributed:
        logger.info('Entering distributed mode.')
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)
        t_model = torch.nn.parallel.DistributedDataParallel(t_model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)

        logger.info('Model now distributed.')

        args.lr_mult = args.batch_size / 256
        args.warmup_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr_mult * args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # tensorboard
        if dist.get_rank() == 0:
            summary_writer = SummaryWriter(log_dir=args.output)
        else:
            summary_writer = None

    else:
        args.lr_mult = 1
        args.warmup_epochs = 5

        model = model.cuda()
        t_model = t_model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        summary_writer = SummaryWriter(log_dir=args.output)

    file_str = 'Loss_{}-Beta_{}-Epoch_{}_Student_{}_distill-Epoch_{}_subset_{}_l0_{}_l1_{}_subset_{}.pth.tar'\
                .format(args.loss, args.beta, args.epochs, args.student_arch, args.teacher_arch, args.subset, args.l_0, args.l_1, args.subset)
    args.resume = os.path.join(args.output, file_str)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # clear unnecessary weights
    torch.cuda.empty_cache()
    augmentation = AUGMENTATIONS[args.aug]
    train_dataset = ImageFolder(root=args.data, transform=augmentation)
    if args.subset:
        indices, counts  = [], {i:[] for i in range(1000)}
        for idx, target in enumerate(train_dataset.targets):
            counts[target].append(idx)
        for _, count in counts.items():
            subset_size = int(len(count)*args.subset)
            indices = indices + count[:subset_size]
        train_dataset = Subset(train_dataset, indices)
        print('Subset initialized with total size', len(train_dataset))

    logger.info('Dataset defined!')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # ensure batch size is dividable by # of GPUs
        assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), \
            'Batch size is not divisible by num of gpus.'

        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    else:
        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)

    criterion = LOSSES[args.loss]

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed: train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, t_model, criterion, optimizer, epoch, args, logger)
        if summary_writer is not None:
            # Tensor-board logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if not args.distributed or dist.get_rank() == 0:


            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student_arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.output, file_str))

            logger.info('==============> checkpoint saved to {}'.format(os.path.join(args.output, file_str)))


def train(train_loader, model, teacher, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')
    mem = ValueMeter('GPU Memory Used', ':5.0f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses, mem],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    #lr.update(get_learning_rate(optimizer))
    #mem.update(torch.cuda.max_memory_allocated(device=0) / 1024.0 / 1024.0)

    # switch to train mode
    model.train()

    if args.distributed:
        teacher.module.eval()

    else:
        teacher.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    times, memory = [], []
    end = time.time()
    for i, (images, _) in enumerate(train_loader):

        if not args.distributed:
            images = images.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                teacher_feats = teacher(images)
            student_feats = model(images)
            loss = args.beta*criterion(teacher_feats, student_feats, args)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #times.append(time.time()-end)
        #memory.append(torch.cuda.max_memory_allocated())
        #torch.cuda.reset_peak_memory_stats(device=None)

        if i % args.print_freq == 0:
            progress.display(i, logger)

        end = time.time()
    #print(torch.tensor(times).float().mean(), torch.tensor(memory).float().mean())
    #torch.save({'times': torch.tensor(times).float(), 'memory': torch.tensor(memory).float()}, f'{dino}_{args.student_arch}.{args.batch_size}.pth')
    #print(torch.mean(torch.tensor(times).float()), torch.mean(torch.tensor(memory).float()))
    #okiedokie()
    return losses.avg


if __name__ == '__main__':
    main(parse_opt())
