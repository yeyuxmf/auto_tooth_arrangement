

import os
import copy
import torch
import numpy as np
import vtkmodules.all as vtk
from data.utils import get_files,walkFile
from data.utils import rotate_maxtrix
import config.config as cfg

def read_stl(file_path):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader

def get_test_data(file_list):

    teeth_points = np.zeros((len(file_list), cfg.sam_points, 3), np.float64)

    Tpoints = [None] *len(file_list)
    Tris = [None] *len(file_list)
    for di in range(len(file_list)):
        stl_reader = read_stl(file_list[di])

        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        polydata = stl_reader.GetOutput()
        vertic_nums = polydata.GetNumberOfPoints()
        face_nums = polydata.GetNumberOfCells()
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        triangles = polydata.GetPolys()

        # 得到所有顶点
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        se_index = np.random.randint(0, mesh_points.shape[0], 400)
        se_points = mesh_points[se_index]

        index = int(cfg.INDEX[teeth_nums]) -1
        teeth_points[index] = se_points
        Tpoints[index] = mesh_points
        Tris[index] = polydata

    # 随机产生多少颗牙齿旋转
    nums = teeth_points.shape[0]
    rotate_nums = np.random.randint(3, nums, 1)[0]
    rotate_index = [i for i in range(nums)]
    np.random.shuffle(rotate_index)
    rotate_index = rotate_index[0: rotate_nums]

    ###################ground tooth decentralization################
    Gteeth_points = copy.deepcopy(teeth_points).reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    Gacenp = np.mean(Gteeth_points, axis=0, keepdims=True)
    Gteeth_points = Gteeth_points - Gacenp
    Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
    ###################ground tooth decentralization over############

    Rteeth_points = copy.deepcopy(Gteeth_points)
    rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
    # ROTAXIS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ###################tooth rotate################################
    for tid in rotate_index:
        v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
        cen = np.mean(Rteeth_points[tid], axis=0)
        points = Rteeth_points[tid] - cen
        rotaxis = np.random.random(3) *2 -1 + 0.01
        # index = np.random.randint(0, 3, 1)[0]
        # rotaxis = cfg.ROTAXIS[index]  #
        rotaxis =  rotaxis / np.linalg.norm(rotaxis)
        angle_ = v1 * np.random.randint(0, 300, 1) / 10.0  # [-30°--30°]
        print(tid, " ", rotaxis, "  ", angle_)
        rt = rotate_maxtrix(rotaxis, angle_)
        rt = rt[0:3, 0:3]
        points_ = (rt.dot(points.T)).T
        Rteeth_points[tid] = points_ + cen
        rms[tid] = rt
    ###################tooth rotate over#############################

    ###################tooth translation#############################
    rotate_nums = np.random.randint(3, nums, 1)[0]
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
            Rteeth_points[i] = Rteeth_points[i] + fg * trans_v

    ###################tooth translation over#############################

    Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    Rcp = np.mean(Rteeth_points, axis=0, keepdims=True)
    Rteeth_points = Rteeth_points - Rcp
    Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)


    ###################quaternion rotate matrix#############################
    trans = Transform3d().compose(Rotate(torch.tensor(rms[:, 0:3, 0:3])))
    final_trans_mat = trans.get_matrix()
    # dof = se3_log_map(final_trans_mat)
    dof = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
    ###################quaternion rotate matrix over#########################

    ###################translation matrix over################################
    trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float64)
    for di in range(Rteeth_points.shape[0]):
        censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
        trans_mats[di] = censd
    ###################center point after transformation######################




    teeth_center = []
    for i in range(Rteeth_points.shape[0]):
        cenp = np.mean(Rteeth_points[i], axis=0)
        teeth_center.append(cenp)

    # file_ = open("./outputs/rotate"  + ".txt", "w")
    # for tid in range(nums):
    #     points = Rteeth_points[tid]
    #     for i in range(points.shape[0]):
    #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
    #
    # file_.close()
    # #
    # file_ = open("./outputs/Gpoint.txt", "w")
    # for tid in range(nums):
    #     points = Gteeth_points[tid]
    #     for i in range(points.shape[0]):
    #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
    #
    # file_.close()

    # print("over")
    Gteeth_points = torch.unsqueeze(torch.tensor(np.array(Gteeth_points)), dim=0)
    Rteeth_points = torch.unsqueeze(torch.tensor(np.array(Rteeth_points)), dim=0)
    teeth_center = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1), dim=0)

    return  Rteeth_points, Gteeth_points, teeth_center, rms, trans_mats, Gacenp, Rcp


from pytorch3d.transforms import *

def get_rotate_polydata(triangles, rpoints):
    points = vtk.vtkPoints()
    for p in rpoints:
        points.InsertNextPoint(p[0], p[1], p[2])

    new_plyd = vtk.vtkPolyData()
    new_plyd.SetPoints(points)
    new_plyd.SetPolys(triangles)

    return new_plyd
def write_stl(polydata, save_path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()

def mapping_output(file_list, Gacenp, Rcp, pdofs, ptrans, gr_matrix, gtrans, save_path):

    pred_matrices = torch.cat([quaternion_to_matrix(pdofs[idx]).unsqueeze(0) for idx in range(pdofs.shape[0])], dim=0)
    pred_matrices = torch.squeeze(pred_matrices).detach().cpu().numpy()

    ptrans = torch.squeeze(ptrans).detach().cpu().numpy()


    gappendFilter = vtk.vtkAppendPolyData()
    rappendFilter = vtk.vtkAppendPolyData()
    rvappendFilter = vtk.vtkAppendPolyData()
    for di in range(len(file_list)):
        stl_reader = read_stl(file_list[di])

        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        polydata = stl_reader.GetOutput()
        vertic_nums = polydata.GetNumberOfPoints()
        face_nums = polydata.GetNumberOfCells()

        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        triangles = polydata.GetPolys()

        mesh_points = mesh_points - Gacenp
        cenp = np.mean(mesh_points, axis=0)
        mesh_points = mesh_points - cenp
        index = int(cfg.INDEX[teeth_nums]) - 1

        ################################
        #rotate #translation
        rpoints = (gr_matrix[index].dot(mesh_points.T)).T
        rpoints = rpoints + cenp
        rpoints = rpoints - gtrans[index]


        rvpoints = rpoints
        rcp = np.mean(rvpoints, axis=0)
        rvpoints = rvpoints - rcp
        rvpoints = rvpoints.dot(pred_matrices[index])
        rvpoints = rvpoints + ptrans[index]
        rvpoints = rvpoints + rcp #+ Rcp

        ################################


        mesh_points = mesh_points + cenp
        gpolydata = get_rotate_polydata(triangles, mesh_points)
        rpolydata = get_rotate_polydata(triangles, rpoints)
        rvpolydata = get_rotate_polydata(triangles, rvpoints)

        gappendFilter.AddInputData(gpolydata)
        rappendFilter.AddInputData(rpolydata)
        rvappendFilter.AddInputData(rvpolydata)

    gappendFilter.Update()
    rappendFilter.Update()
    rvappendFilter.Update()

    write_stl(gappendFilter.GetOutput(), save_path+"_g"  + ".stl")
    write_stl(rappendFilter.GetOutput(), save_path+"_r"  + ".stl")
    write_stl(rvappendFilter.GetOutput(), save_path+"_rv" + ".stl")


    return 0


