#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data import TrainData, train_data_load
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from net.arangementnet import teeth_arangement_model
from util import IOStream, Tooth_Assembler
from net.loss import GeometricReconstructionLoss, symmetric_loss, spatial_Relation_Loss
import config.config as cfg

def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dictf = {k.replace('module.', ""): v for k, v in pretrained_dict.items() if k.replace('module.', "") in model_dict}
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("over")


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "./dataset/train/"
    train_loader = DataLoader(TrainData(file_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    file_path = "./dataset/test/"
    # test_loader = DataLoader(TrainData(file_path), num_workers=0,
    #                          batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = teeth_arangement_model()
    tooth_assembler = Tooth_Assembler()
    reconl1_loss = GeometricReconstructionLoss()
    model_path = "./outputs/model_2000_rotate_transv2.pth"
    model_initial(model, model_path)

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([{'params': model.local_fea.parameters(), 'lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        # opt = optim.SGD([
        #     {'params': model.teeth_fea.parameters(), 'lr': args.lr},
        #     {'params': model.global_fea.parameters(), 'lr': args.lr},
        #     {'params': model.output.parameters(), 'lr': args.lr}])

    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    model.cuda()
    model.train()
    scaler = GradScaler()
    best_test_acc = 0
    inter_nums = len(train_loader)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_loss = 0.0
        count = 0.0
        recon_loss = 0
        c_loss = 0
        dof_loss = 0
        trans_loss = 0
        angle_loss = 0
        sym_loss = 0
        spl_loss = 0
        # for data, edges, label in train_loader:
        nums = 0
        tic = time.time()
        train_data, train_label, teeth_center, dof = [], [], [], []
        nnums = 0
        for cafh in train_loader:

            train_data, train_label, teeth_center, gdofs, gtrans, tweights, rweights, mask_index =  train_data_load(cafh)
            train_data = train_data.cuda().float()
            train_label = train_label.cuda().float()
            teeth_center = teeth_center.cuda().float()
            gdofs = gdofs.cuda().float()
            gtrans = gtrans.cuda().float()
            tweights = tweights.cuda().float()
            rweights = rweights.cuda().float()

            mask_index = mask_index.cuda().long()

            weights = rweights -1 + tweights

            gdofs = gdofs#

            nums = nums + 1
            batch_size = train_data.size()[0]
            opt.zero_grad()
            # data = torch.squeeze(data)
            with autocast():
                pdofs, ptrans = model(train_data, teeth_center)
                assembled = tooth_assembler(train_data, teeth_center, pdofs, ptrans, device)

                nnums = nnums + 1
                recon_loss_, c_loss_ = reconl1_loss(assembled, train_label, weights, device)
                dof_loss_ = torch.sum(torch.sum(F.smooth_l1_loss(pdofs[mask_index], gdofs[mask_index], reduction= "none"), dim=-1) * rweights[mask_index]) / pdofs[mask_index].shape[0]
                trans_loss_ = torch.sum(torch.sum(F.smooth_l1_loss(ptrans, gtrans, reduction= "none"), dim=-1) * tweights) / ptrans.shape[0]
                # gtrans_numpy = gtrans.detach().cpu().numpy()
                angle_loss_ = torch.sum(1-torch.sum(pdofs[mask_index]*gdofs[mask_index], dim=-1)) / pdofs[mask_index].shape[0]

                sym_loss_ = dof_loss_#symmetric_loss(assembled)
                spl_loss_ = dof_loss_#spatial_Relation_Loss(assembled, train_label, weights, device)

                loss = recon_loss_ + c_loss_ * 1 + dof_loss_ * 10 + angle_loss_ + trans_loss_ * 1 # + 1*spl_loss_ #+ 1*sym_loss_ #

            scaler.scale(loss).backward()
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(opt)
            # Updates the scale for next iteration
            scaler.update()

            count += batch_size
            train_loss += loss.item()
            recon_loss += recon_loss_.item()
            c_loss += c_loss_.item()
            dof_loss += dof_loss_.item()
            trans_loss += trans_loss_.item()
            angle_loss += angle_loss_.item()
            sym_loss += sym_loss_.item()
            spl_loss += spl_loss_.item()

            if nums % cfg.VIEW_NUMS == 0:
                toc = time.time()
                train_loss = train_loss/ (cfg.VIEW_NUMS)
                recon_loss = recon_loss/(cfg.VIEW_NUMS)
                c_loss = c_loss/(cfg.VIEW_NUMS)
                dof_loss = dof_loss/(cfg.VIEW_NUMS)
                trans_loss = trans_loss/(cfg.VIEW_NUMS)
                angle_loss = angle_loss/(cfg.VIEW_NUMS)
                sym_loss = sym_loss/(cfg.VIEW_NUMS)
                spl_loss = spl_loss/(cfg.VIEW_NUMS)

                print("lr = ", opt.param_groups[0]['lr'])
                outstr = 'epoch %d /%d,epoch %d /%d, loss: %.6f, recon_loss: %.6f, c_loss: %.6f, dof_loss: %.6f, trans_loss: %.6f, sym_loss: %.6f, spl_loss: %.6f, angle_loss: %.6f, const time: %.6f' % (
                 epoch,args.epochs, nums, inter_nums, train_loss, recon_loss, c_loss, dof_loss, trans_loss, sym_loss, spl_loss, angle_loss, toc - tic)

                io.cprint(outstr)

                train_loss = 0.0
                count = 0.0
                recon_loss = 0
                c_loss = 0
                dof_loss = 0
                trans_loss =0
                angle_loss = 0
                sym_loss = 0
                spl_loss = 0
                tic = time.time()



        if (epoch) % cfg.SAVE_MODEL == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/teethseg_model_' + str(epoch)+ '.pth')


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls_1024', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=2001, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=1.5*1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)

