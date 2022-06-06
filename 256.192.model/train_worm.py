import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import torchvision.utils as vutils

from tensorboardX import SummaryWriter


# from config import cfg
from config_worm import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.loss_worm import get_losses
from networks import network_worm
# from dataloader.mscocoMulti_double_only import MscocoMulti_double_only
from dataloader.mscocoMulti_double_only_worm import MscocoMulti_double_only_worm
# from dataloader.mscocoMulti_double_only_worm_bi import MscocoMulti_double_only_worm
# from dataloader.mscocoMulti import MscocoMulti




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    args.checkpoint = './' + args.checkpoint
    tensorboard_path = args.checkpoint + '/runs/'
    # import pdb;pdb.set_trace()
    writer = SummaryWriter(tensorboard_path)
    # create checkpoint dir
    counter = 0
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network_worm.__dict__[cfg.model](
        cfg.output_shape, cfg.num_class, cfg.inter, pretrained=True)

    model = torch.nn.DataParallel(model).to(device)
    # model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion_bce = torch.nn.BCELoss().to(device)
    criterion_abs = torch.nn.L1Loss().to(device)
    # criterion_abs = offset_loss().to(device)
    # criterion1 = torch.nn.MSELoss().to(device) # for Global loss
    # criterion2 = torch.nn.MSELoss(reduce=False).to(device) # for refine loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 eps=1e-08,
                                 weight_decay=cfg.weight_decay)

    if args.resume:
        print(args.resume)
        checkpoint_file_resume = os.path.join(args.checkpoint, args.resume+'.pth.tar')
        if isfile(checkpoint_file_resume):
            print("=> loading checkpoint '{}'".format(checkpoint_file_resume))
            checkpoint = torch.load(checkpoint_file_resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file_resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file_resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel()
                                            for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        MscocoMulti_double_only_worm(cfg),
        batch_size=cfg.batch_size*args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(
            optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss, counter = train(train_loader, model, [
                                    criterion_abs, criterion_bce], writer, counter, optimizer, device)
        print('train_loss: ', train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])
        if epoch % 10 == 0:
            save_model({
                'epoch': epoch + 1,
                'info': cfg.info,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

    writer.export_scalars_to_json("./test.json")
    writer.close()

    logger.close()


def train(train_loader, model, criterions, writer, counter, optimizer, device):
    criterion_abs, criterion_bce = criterions
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # Freezing batchnorm2d
    # print("Freezing mean/var of BatchNorm2d")
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False
    # import pdb; pdb.set_trace()
    for i, (inputs, targets, end_point_label, control_point_label, intersection_areas, directional_mask, connection_pair, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.to(device))
        # targets = targets.type(torch.FloatTensor)
        targets = torch.autograd.Variable(targets.to(device))
        # input_var = inputs.to(device)

        mask_target = targets[:, 0:2, :, :].to(device)
        control_point_target = targets[:, 2, :, :].to(device).unsqueeze(1)
        end_point_target = targets[:, 3, :, :].to(device).unsqueeze(1)
        long_offset_target = targets[:, 4:6, :, :].to(device)
        next_offset_target = targets[:, 6:8, :, :].to(device)
        prev_offset_target = targets[:, 8:10, :, :].to(device)
        control_short_offset_target = targets[:, 10:12, :, :].to(device)
        end_short_offset_target = targets[:, 12:14, :, :].to(device)
        end_point_label = end_point_label.to(device)
        control_point_label = control_point_label.to(device)
        directional_mask = directional_mask.to(device)

        intersection_areas = intersection_areas.to(device)


        # binary_targets = [i.to(device) for i in binary_targets]



        ground_truth = [mask_target, control_point_target, end_point_target,
                        long_offset_target, next_offset_target, prev_offset_target,
                        control_short_offset_target, end_short_offset_target,
                        end_point_label, control_point_label, intersection_areas, directional_mask, connection_pair
                        ]

        # import pdb;pdb.set_trace()
        with torch.enable_grad():
            optimizer.zero_grad()

            outputs = model(input_var)
            loss, loss_mask, loss_control_pt, loss_long_offset, loss_next_offset, loss_short_control_pt, \
             pull_loss_end, push_loss_end, pull_loss_control, push_loss_control, binary_targets_loss, refine_loss, directional_mask_loss= get_losses(ground_truth, outputs)

        # # comput global loss and refine loss
        # for global_output, label in zip(mask_pred, mask_target):

        #     global_output_flat = global_output.view(-1)
        #     label_flat = label.view(-1)
        #     # global_loss = criterion_bce(global_output_flat, torch.autograd.Variable(label_flat.cuda(async=True)))
        #     global_loss = criterion_bce(
        #         global_output_flat, label_flat.to(device))
        #     loss += global_loss
        #     #########################################
        #     # num_points = global_output.size()[1]
        #     # global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
        #     # global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(async=True))) / 2.0
        #     # loss += global_loss
        #     global_loss_record += global_loss.data.item()
        # import pdb; pdb.set_trace()
        # comput offset loss
        # for offset_output, label, kp_maps in zip(global_offset, targets_offset, points_targets):

        #     # import pdb; pdb.set_trace()
        #     # kp_map = points_targets[0][:,1,...]  #  (#,1,width,height)
        #     kp_map = kp_maps[:, 1, ...]  # (#,1,width,height)
        #     kp_map = torch.stack(
        #         (kp_map, kp_map, kp_map, kp_map, kp_map, kp_map, kp_map, kp_map, kp_map, kp_map), 1)

        #     # offset_loss= offset_loss_disc(global_offset, targets_offset.to(device),kp_map.to(device))
        #     offset_loss = offset_loss_disc(
        #         offset_output, label.to(device), kp_map.to(device))
        #     loss += offset_loss
        #     offset_loss_record += offset_loss.data.item()

            #########################################
            # num_points = global_output.size()[1]
            # global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            # global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(async=True))) / 2.0
            # loss += global_loss
        ###################################################

        # refine_loss = criterion_bce(refine_output, refine_target_var)

        # loss += refine_loss

        # import pdb; pdb.set_trace()
        #########################################################
        # refine_loss = criterion2(refine_output, refine_target_var)
        # refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        # # refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        # refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        # refine_loss = ohkm(refine_loss, 8)
        # loss += refine_loss
        # #####################################################
        # refine_loss_record = refine_loss.data.item()
        # record loss

            losses.update(loss.data.item(), inputs.size(0))
            # import pdb;pdb.set_trace()
            loss = loss.to(device)
        # compute gradient and do Optimization step

            loss.backward()
            optimizer.step()
        # import pdb; pdb.set_trace()

        ##########
        # tensor_vis_offset_h = global_offset[3][0,0,...]
        # tensor_vis_offset_v = global_offset[3][0,1,...]
        # tensor_vis_offset_short_h = global_offset[3][0,2,...]
        # tensor_vis_offset_short_v = global_offset[3][0,3,...]
        # tensor_vis_start_point = refine_output[3][0,0,...]
        # tensor_vis_control_point = refine_output[3][0,1,...]
        # # import pdb; pdb.set_trace()
        # vis_offset_short_h = vutils.make_grid(tensor_vis_offset_short_h, normalize=True, scale_each=True)
        # vis_offset_short_v = vutils.make_grid(tensor_vis_offset_short_v, normalize=True, scale_each=True)
        # vis_offset_v = vutils.make_grid(tensor_vis_offset_v, normalize=True, scale_each=True)
        # vis_offset_h = vutils.make_grid(tensor_vis_offset_h, normalize=True, scale_each=True)
        # vis_start_point = vutils.make_grid(tensor_vis_start_point, normalize=True, scale_each=True)
        # vis_control_point = vutils.make_grid(tensor_vis_control_point, normalize=True, scale_each=True)

        # writer.add_image('Image vis_offset_h', vis_offset_h, counter)
        # writer.add_image('Image vis_offset_v', vis_offset_v, counter)
        # writer.add_image('Image vis_offset_short_h', vis_offset_short_h, counter)
        # writer.add_image('Image vis_offset_short_v', vis_offset_short_v, counter)

        # writer.add_image('Image vis_start_point', vis_start_point, counter)
        # writer.add_image('Image vis_control_point', vis_control_point, counter)
        writer.add_scalar('loss', loss.data.item(), counter)
        writer.add_scalar('loss_mask', loss_mask.data.item(), counter)
        writer.add_scalar('loss_control_pt', loss_control_pt.data.item(), counter)
        writer.add_scalar('loss_long_offset', loss_long_offset.data.item(), counter)
        writer.add_scalar('loss_next_offset', loss_next_offset.data.item(), counter)
        writer.add_scalar('loss_short_control_pt', loss_short_control_pt.data.item(), counter)
        writer.add_scalar('pull_end', pull_loss_end, counter)
        writer.add_scalar('push_end', push_loss_end, counter)
        writer.add_scalar('pull_control', pull_loss_control, counter)
        writer.add_scalar('push_control', push_loss_control, counter)
        writer.add_scalar('directional_mask_loss', directional_mask_loss, counter)
        writer.add_scalar('binary_targets_loss', binary_targets_loss, counter)
        writer.add_scalar('refine_loss', refine_loss.data.item(), counter)
        writer.add_scalar('losses.avg', losses.avg, counter)

        counter = counter + 1
        # import pdb; pdb.set_trace()
        if(i % 5 == 0 and i != 0):
            print('iteration {} | loss: {}, avg loss: {}, loss_mask: {} , push_control: {} , pull_control: {}, directional_mask_loss:{} '
                  .format(i, loss.data.item(), losses.avg, loss_mask.data.item(), push_loss_control, pull_loss_control, directional_mask_loss))

    return losses.avg, counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())
