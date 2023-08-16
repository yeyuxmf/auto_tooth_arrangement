import os
import math
import numpy as np
def walkFile(path_root, file_list):

    for root, dirs, files in os.walk(path_root):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历所有的文件夹
        for d in dirs:
            path_file = os.path.join(root, d)
            file_list.append(path_file)

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

def rotation_matrix(rotate_axis, rotate_angle):
    M_PI = math.pi
    axis = rotate_axis
    angle = rotate_angle

    m = np.zeros((4,4) ,np.float64)
    a = angle * (M_PI / 180.0)
    c = math.cos(a)
    s = math.sin(a)
    one_m_c = 1 - c
    ax = axis / np.sqrt(np.sum(np.power(axis, 2)))

    m[0, 0] = ax[0] * ax[0] * one_m_c + c
    m[0, 1] = ax[0] * ax[1] * one_m_c - ax[2] * s
    m[0, 2] = ax[0] * ax[2] * one_m_c + ax[1] * s

    m[1, 0] = ax[1] * ax[0] * one_m_c + ax[2] * s
    m[1, 1] = ax[1] * ax[1] * one_m_c + c
    m[1, 2] = ax[1] * ax[2] * one_m_c - ax[0] * s

    m[2, 0] = ax[2] * ax[0] * one_m_c - ax[1] * s
    m[2, 1] = ax[2] * ax[1] * one_m_c + ax[0] * s
    m[2, 2] = ax[2] * ax[2] * one_m_c + c

    m[3, 3] = 1.0

    return m

def rotate_maxtrix(rotaxis, angle_):
    M_PI = math.pi
    rt = np.eye(4)  #单位矩阵
    if (np.sqrt(rotaxis.dot(rotaxis)) > 0.001):
        rotaxis = rotaxis / np.sqrt(np.sum(np.power(rotaxis, 2)))
        rotangle = angle_
        rt = rotation_matrix(rotaxis, rotangle)

    return rt