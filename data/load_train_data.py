import os
import copy
import numpy as np
import torch
import torch.nn as nn
import vtkmodules.all as vtk
from data.utils import get_files, walkFile, rotate_maxtrix
from config import config as cfg

from pytorch3d.transforms import *

from typing import Dict, List, Optional, Tuple, Callable



class TrainDataAugmentation(nn.Module):
    def __init__(self):
        super(TrainDataAugmentation, self).__init__()

    @torch.no_grad()
    def forward(self, X: torch.Tensor):
        teeth_num = X.shape[0]

        trans = Transform3d().compose(Translate(-X["C"]),
                                      Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (cfg.teeth_nums, 3)), "XYZ")),
                                      Translate(torch.clamp(torch.randn(cfg.teeth_nums, 3), -3.14, 3.14)),
                                      Translate(X["C"]))
        X = trans.transform_points(X["X_v"])
        X = X.clone().reshape(shape=X["X"].shape)
        X_matrices = trans.inverse().get_matrix()


        final_trans_mat = X_matrices  # trans.get_matrix()
        X["6dof"] = se3_log_map(final_trans_mat)

        return X



def data_load_no_centering(file_path):

    file_data = np.load(file_path, allow_pickle=True).item()

    teeth_nums = []
    teeth_points = []
    for key in file_data:
        teeth_nums.append(int(key))
        teeth_points.append(file_data[key])

    teeth_nums = np.array(teeth_nums)
    order_index = np.argsort(teeth_nums)
    teeth_nums = teeth_nums[order_index]

    teeth_points = np.array(teeth_points)
    teeth_points = teeth_points[order_index]

    # teeth_points = teeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    # teeth_points = teeth_points - np.mean(teeth_points, axis=0)
    # teeth_points = teeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

    return teeth_points, teeth_nums

def teeth_whole_rotate(teeth_points, rt):

    teeth_points = teeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    teeth_points = (rt.dot(teeth_points.T)).T
    teeth_points = teeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)


    return teeth_points




class TrainData():
    def __init__(self, file_root):
        self.data_dir = file_root

        self.train_list = None
        self.prepare(self.data_dir)

    def prepare(self, file_path):

        file_list = []
        get_files(file_path, file_list, "end.npy")

        self.train_list = file_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):

        file_path = self.train_list[item]

        return file_path
def train_data_load(file_path_):

    tRteeth_points, tGteeth_points, tteeth_center, tgdofs, ttrans_mats, tweights, rweights, rcpoints, mask_ = [],[],[],[],[],[],[],[],[]

    for ffi in range(len(file_path_)):
        file_path = file_path_[ffi]
        #load data and centering
        Gteeth_points, teeth_nums = data_load_no_centering(file_path)

        #flags = 1 : If you have data before and after orthodontic treatment.
        #flags = 0 : If you have data only after orthodontic treatment.
        #I have data before and after orthodontic treatment, and I enable both 0 and 1 simultaneously.
        flags = 0# np.random.randint(0, 2, 1)[0]
        if 0 == flags:
            Rteeth_points = copy.deepcopy(Gteeth_points)
            mask_.append(ffi)
        else:
            Rteeth_points, teeth_nums_ = data_load_no_centering(file_path.replace("end.npy", "start.npy"))


        #######################################################
        #teeth whole rotate [-30, 30] #x\y\z
        v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
        angle_ = v1 * np.random.randint(0, 5, 1)[0]  # [-30°--30°]
        index = np.random.randint(0, 3, 1)[0]
        rotaxis = cfg.ROTAXIS[index]
        rt = rotate_maxtrix(rotaxis, angle_)
        rt = rt[0:3, 0:3]
        # Gteeth_points = teeth_whole_rotate(Gteeth_points, rt)
        # Rteeth_points = teeth_whole_rotate(Rteeth_points, rt)
        ############################################################

        #Randomly generate how many teeth rotate.
        nums = len(teeth_nums)
        rotate_nums = np.random.randint(0, nums, 1)[0]
        rotate_index = [i for i in range(nums)]
        np.random.shuffle(rotate_index)
        rotate_index = rotate_index[0: rotate_nums]


        Rweights = np.ones((Rteeth_points.shape[0]))
        Tweights = np.ones((Rteeth_points.shape[0]))
        ###################tooth rotate over#############################
        rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)

        if 0 == flags:
            for tid in rotate_index:
                v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
                cen = np.mean(Rteeth_points[tid], axis=0)
                points = Rteeth_points[tid] - cen
                rotaxis = np.random.random(3) *2 -1 + 0.01
                index = np.random.randint(0, 3, 1)[0]
                # rotaxis = cfg.ROTAXIS[index] #rotaxis / np.linalg.norm(rotaxis)
                rotaxis =  rotaxis / np.linalg.norm(rotaxis)

                angle_ = v1* cfg.Angles[np.random.randint(0, cfg.AgSize, 1)[0]]  #[-30°--30°]
                rt = rotate_maxtrix(rotaxis, angle_)
                rt = rt[0:3, 0:3]
                points_ = (rt.dot(points.T)).T
                Rteeth_points[tid] = points_ + cen

                rms[tid] = rt
                Rweights[tid] = Rweights[tid] + abs(angle_)*3 /100.0
            ###################tooth rotate over#############################

            ###################tooth translation#############################
            rotate_nums = np.random.randint(0, nums, 1)[0]
            rotate_index = [i for i in range(nums)]
            np.random.shuffle(rotate_index)
            rotate_index = rotate_index[0: rotate_nums]
            trans_v = np.array([[-2, -2, 2]])
            for i in range(Rteeth_points.shape[0]):
                index = np.random.randint(0, 3, 1)[0]
                # rotaxis = np.random.random(3) * 2 - 1 + 0.01
                rotaxis = cfg.ROTAXIS[index]  # rotaxis / np.linalg.norm(rotaxis)

                # v1 = np.random.normal(0, 1, size=(1))[0]
                # fg = np.clip(np.array([v1]), -1, 1)
                # trans_v = fg * rotaxis * scalev

                v1 = np.random.normal(0, 1, size=(1))[0]
                v2 = np.random.normal(0, 1, size=(1))[0]
                v3 = np.random.normal(0, 1, size=(1))[0]
                fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)

                if i in rotate_index:
                    Rteeth_points[i] = Rteeth_points[i] + fg *trans_v

            ###################tooth translation  over#############################


            ###################quaternion rotate matrix#############################
            Gcenp = np.mean(Gteeth_points, axis=1)
            Rcenp = np.mean(Rteeth_points, axis=1)
            trans = Transform3d().compose(Translate(torch.tensor(-Rcenp)),
                                          Rotate(torch.tensor(rms[:, 0:3, 0:3])),
                                          Translate(torch.tensor(Gcenp)))
            final_trans_mat = trans.get_matrix()
            gdofs = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
            ###################quaternion rotate matrix over#########################
            # Nteeth_points = trans.transform_points(torch.tensor(Rteeth_points.astype(np.float32))).numpy()
            # Rteeth_points = Nteeth_points

            ###################translation matrix over################################
            #The data obtained from normal orthodontic treatment of teeth shows displacement
            # changes in almost every tooth, so the manufactured dataset also needs to meet this requirement.
            #If your data is not like this, then changes need to be made.
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            rcpoint = np.mean(Rteeth_points, axis=0)
            Rteeth_points = Rteeth_points - rcpoint
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            Gteeth_points = Gteeth_points - np.mean(Gteeth_points, axis=0)
            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

            ###################translation matrix##################################
            trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float32)
            for di in range(Rteeth_points.shape[0]):
                censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
                trans_mats[di] = censd
                Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0
        else:
            gdofs = torch.ones((cfg.teeth_nums, 4))
            trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float32)
            for di in range(Rteeth_points.shape[0]):
                censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
                trans_mats[di] = censd
                Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0


        ###################translation matrix over################################
        Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
        rcpoint = np.mean(Rteeth_points, axis=0)
        Rteeth_points = Rteeth_points - rcpoint
        Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

        Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
        Gteeth_points = Gteeth_points - rcpoint
        Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)

        ###################center point after transformation######################
        teeth_center = []
        for i in range(Rteeth_points.shape[0]):
            cenp = np.mean(Rteeth_points[i], axis=0)
            teeth_center.append(cenp)
        ###################center point after transformation [over]################

        # file_ = open("./outputs/rotate"  + ".txt", "w")
        # for tid in range(nums):
        #     points = Rteeth_points[tid]
        #     for i in range(points.shape[0]):
        #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
        #
        # file_.close()
        #
        # file_ = open("./outputs/gpoint.txt", "w")
        # for tid in range(nums):
        #     points = Gteeth_points[tid]
        #     for i in range(points.shape[0]):
        #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
        #
        # file_.close()

        # print("over")
        tGteeth_points.append(torch.tensor(np.array(Gteeth_points)))
        tRteeth_points.append(torch.tensor(np.array(Rteeth_points)))
        tteeth_center.append(torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1))
        rweights.append(torch.tensor(Rweights))
        tweights.append(torch.tensor(Tweights))
        tgdofs.append(gdofs)
        ttrans_mats.append(torch.tensor(trans_mats))
        rcpoints.append(rcpoint)

    tGteeth_points = torch.stack(tGteeth_points, dim=0)
    tRteeth_points = torch.stack(tRteeth_points, dim=0)
    tteeth_center = torch.stack(tteeth_center, dim=0)
    tweights = torch.stack(tweights, dim=0)
    rweights = torch.stack(rweights, dim=0)
    tgdofs = torch.stack(tgdofs, dim=0)
    ttrans_mats = torch.stack(ttrans_mats, dim=0)
    rcpoints = torch.tensor(np.array(rcpoints))
    mask_ = torch.tensor(np.array(mask_))


    return tRteeth_points, tGteeth_points, tteeth_center, tgdofs, ttrans_mats, tweights, rweights, mask_


















