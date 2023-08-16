

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class GeometricReconstructionLoss(nn.Module):
    def __init__(self):
        super(GeometricReconstructionLoss, self).__init__()

    def forward(self, X_v, target_X_v, weights, device: torch.device):

        loss = torch.zeros([X_v.shape[0], X_v.shape[1]]).to(device)
        for bn in range(X_v.shape[0]):

            for idx in range(X_v.shape[1]):

                pred = X_v[bn, idx, :, :]
                tag = target_X_v[bn, idx, :, :]
                pred_ = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
                tag_ = tag.unsqueeze(0).repeat(pred.shape[0] ,1,  1)

                diff = torch.sum(torch.pow(torch.sub(pred_, tag_), 2), dim=-1)
                minv = torch.argmin(diff, dim=1)
                # minvy = torch.argmin(diff, dim=0)

                # predd = pred[minvy]

                tagp = tag[minv]
                tmp1 = F.smooth_l1_loss(pred, tagp, reduction="mean")
                # tmp2 = F.smooth_l1_loss(tag, predd, reduction="mean")

                loss[bn, idx]= tmp1 #+ tmp2

        loss = torch.sum(loss * weights)
        prec = torch.mean(X_v, dim=2)
        tarc = torch.mean(target_X_v, dim=2)
        lossc = F.smooth_l1_loss(prec, tarc, reduction="sum") / (prec.shape[0]*3)
        losstc = F.smooth_l1_loss(torch.mean(prec, dim=1), torch.mean(tarc, dim=1), reduction="mean")
        loss = loss/  target_X_v.shape[0] # Variable(loss, requires_grad=True)
        return loss, lossc#losstc + lossc

def symmetric_loss(X_v):

    nums = X_v.shape[1]//2
    rg = X_v[:, 0:nums, :, :]
    lg = X_v[:, nums:, :, :]
    lg = torch.flip(lg, dims=[1])

    rgc = torch.abs(torch.mean(rg, dim=2))
    lgc =  torch.abs(torch.mean(lg, dim=2))

    lossc = F.smooth_l1_loss(rgc[:, :, 0:2], lgc[:, :, 0:2], reduction="sum") / (rgc.shape[0] * 2)

    return lossc


def nearnest_index(pred_, tag_):

    pred = pred_ -torch.mean(pred_, dim=0)
    tag = tag_ -torch.mean(pred_, dim=0)

    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)

    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)

    minv = torch.min(diff, dim=1)[0]

    # nearnestp = tag_[min_index]

    return min_index, minv

def nearnest_value(pred_, tag_):

    pred = pred_
    tag = tag_

    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)

    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)


    nearnestp = tag_[min_index]
    minv = pred_ - nearnestp

    return min_index, minv



def spatial_Relation_Loss(pred, target, weights, device):

    loss = torch.zeros([pred.shape[0], pred.shape[1]]).to(device)
    for bn in range(pred.shape[0]):

        for idx in range(pred.shape[1] -1):

            pred1 = pred[bn, idx, :, :]
            pred2 = pred[bn, idx+1, :, :]


            tag1 = target[bn, idx, :, :]
            tag2 = target[bn, idx+1, :, :]

            # with torch.no_grad():
            min_index1, _ = nearnest_index(pred1, tag1)
            min_index2, _ = nearnest_index(pred2, tag2)

            tag1_ = tag1[min_index1]
            tag2_ = tag2[min_index2]

            min_indexpp, minvp1 = nearnest_value(pred1, pred2)
            min_indextp, minvpt1= nearnest_value(tag1_, tag2)

            min_indexpp, minvp2 = nearnest_value(pred2, pred1)
            min_indextp, minvpt2= nearnest_value(tag2_, tag1)

            # with torch.no_grad():
            # mask1 = torch.abs(minvpt1) < 15
            # mask2 = torch.abs(minvpt2) < 15

            minvp_mask1 = minvp1#[mask1]
            minvpt_mask1 = minvpt1#[mask1]

            minvp_mask2 = minvp2#[mask2]
            minvpt_mask2 = minvpt2#[mask2]

            lossc1 = F.smooth_l1_loss(minvp_mask1, minvpt_mask1, reduction="mean")
            lossc2 = F.smooth_l1_loss(minvp_mask2, minvpt_mask2, reduction="mean")



            loss[bn, idx] = (lossc1 + lossc2)*0.5  # + tmp2

    loss = torch.sum(loss) / weights.shape[0]

    return  loss



