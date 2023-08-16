#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import torch
import numpy as np
from net.arangementnet import teeth_arangement_model
from util import IOStream, Tooth_Assembler
from data.utils import get_files,walkFile
from data.load_test_data import get_test_data, mapping_output
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



def walkFileType(path_root, file_list, type_):

    for root, dirs, files in os.walk(path_root):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历所有的文件夹
        for d in dirs:
            path_file = os.path.join(root, d)
            if type_ in path_file:
                file_list.append(path_file)
def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try to load models
    model = teeth_arangement_model()
    tooth_assembler = Tooth_Assembler()
    model_path = "./outputs/teethseg_model_2000_rotate_transv2.pth"
    model_initial(model, model_path)
    model.cuda()
    model.eval()


    file_path = "./dataset/split_stl"
    dir_list = []
    walkFileType(file_path, dir_list, "end")


    save_root =  "./outputs/"
    for fi in range(0, len(dir_list)):
        file_list = []
        file_path = dir_list[fi]
        get_files(file_path, file_list, ".stl")
        dir_name = os.path.split(file_path)[-1]

        train_data, train_label, teeth_center, gr_matrix, gtrans, Gacenp, Rcp = get_test_data(file_list)

        train_data = train_data.cuda().float()
        train_label = train_label.cuda().float()
        teeth_center = teeth_center.cuda().float()

        # data = torch.squeeze(data)
        with torch.no_grad():
            pdofs, ptrans = model(train_data, teeth_center)
            Rcpt = torch.tensor(Rcp).cuda().float().view(1, 3)
            assembled = tooth_assembler(train_data, teeth_center, pdofs, ptrans, device)

        mapping_output(file_list, Gacenp, Rcp, pdofs, ptrans, gr_matrix, gtrans, save_root+dir_name)

        print("over")


if __name__ == "__main__":

    test()


