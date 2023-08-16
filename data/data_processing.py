import os
import json
# import openctm
import numpy as np
import vtkmodules.all as vtk
from data.utils import walkFile,get_files


INDEX = {"2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,"8": 7,
         "9": 8, "10": 9, "11": 10, "12": 11, "13": 12,"14": 13, "15": 14,
         "18": 1, "19": 2, "20": 3, "21": 4, "22": 5,"23": 6, "24": 7,
         "25": 8, "26": 9, "27": 10, "28": 11, "29": 12,"30": 13, "31": 14}

INDEXC = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

def data_process_mtc_stl():
    mesh = openctm.import_mesh("G:/teeth_arrangement_data/case_828726/arch_o_u.mtc")
    points = vtk.vtkPoints()
    for p in mesh.vertices:
        points.InsertNextPoint(p[0], p[1], p[2])

    triangles = vtk.vtkCellArray()
    for face in mesh.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)

    plyd = vtk.vtkPolyData()
    plyd.SetPoints(points)
    plyd.SetPolys(triangles)

    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName("G:/teeth_arrangement_data/case_828726/arch_o_u.stl")
    stlWriter.SetInputData(plyd)
    stlWriter.Update()
    stlWriter.SetFileTypeToBinary()
    stlWriter.Write()


def split_teeth():


    file_list = []
    file_path = "H:/teeth_arrangement_data/20230619/stl/"
    get_files(file_path, file_list, ".stl")

    save_root = "H:/teeth_arrangement_data/20230619/split_stl/"
    label_root = "H:/teeth_arrangement_data/20230619/label/"
    for di in range(len(file_list)):
        data_file = file_list[di]
        print(di, " = ", data_file)
        data_name = os.path.split(data_file)[-1]
        label_file = label_root + data_name.replace(".stl", ".json")

        if not os.path.exists(label_file):  #判断当前数据的标签文件是否存在
            return
        with open(label_file, 'r', encoding='utf8') as fp:
            tmask_indexs = json.load(fp)

        tmask_indexs = tmask_indexs["teeth_info"]
        reader = vtk.vtkSTLReader()
        reader.SetFileName(data_file)
        reader.Update()
        ori_plyd = reader.GetOutput()

        d_name = data_name.replace(".stl", "")
        total_tooth_ids = vtk.vtkIdTypeArray()
        total_tooth_ids.SetNumberOfComponents(1)

        polyDatas = {}
        for item in tmask_indexs:   #有多少颗牙齿就循环多少次
            t_index = tmask_indexs[item]["pred_teeth_num"]      #牙齿的医学编号
            vids = tmask_indexs[item]["teeth_vids"]
            vids = list(vids.values())

            teeth_ids = vtk.vtkIdTypeArray()
            teeth_ids.SetNumberOfComponents(1)
            for vid in vids:
                teeth_ids.InsertNextValue(vid)
                total_tooth_ids.InsertNextValue(vid)

            selectionNode = vtk.vtkSelectionNode()
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
            selectionNode.SetSelectionList(teeth_ids)
            selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

            selection = vtk.vtkSelection()
            selection.AddNode(selectionNode)
            extractSelection = vtk.vtkExtractSelection()
            extractSelection.SetInputData(0, ori_plyd)
            extractSelection.SetInputData(1, selection)
            extractSelection.Update()
            extract_op = extractSelection.GetOutput()
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(extract_op)
            surface_filter.Update()
            teeth_plyd = surface_filter.GetOutput()

            if t_index in INDEXC:
                polyDatas[t_index] = teeth_plyd

            if 14 == len(polyDatas):
                save_path = save_root + d_name
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                for key in  polyDatas:
                    teeth_plyd = polyDatas[key]
                    t_index = key


                    stlWriter = vtk.vtkSTLWriter()
                    stlWriter.SetFileName(save_path + "/" +d_name + "_" + str(t_index) + ".stl")
                    stlWriter.SetInputData(teeth_plyd)
                    stlWriter.Update()
                    stlWriter.SetFileTypeToBinary()
                    stlWriter.Write()

def read_stl(file_path):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader
def  train_data():

    file_path = "H:/teeth_arrangement_data/train_data/middle_stl/"
    dir_list = []
    walkFile(file_path, dir_list)


    save_root =  "H:/teeth_arrangement_data/train_data/middle_stl/npy/"
    for fi in range(len(dir_list)):
        file_list = []
        file_path = dir_list[fi]
        get_files(file_path, file_list, ".stl")
        dir_name = os.path.split(file_path)[-1]

        jaw_points = {}
        file_ = open(save_root + dir_name+ ".txt", "w")
        flags = np.zeros((14), np.int32)
        for di in range(len(file_list)):
            stl_reader = read_stl(file_list[di])

            teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
            polydata = stl_reader.GetOutput()
            vertic_nums = polydata.GetNumberOfPoints()
            face_nums = polydata.GetNumberOfCells()

            # 得到所有顶点
            verts = []
            for i in range(polydata.GetNumberOfPoints()):
                verts.append(polydata.GetPoint(i))
            mesh_points = np.array(verts)
            se_index = np.random.randint(0, mesh_points.shape[0], 400)
            se_points = mesh_points[se_index]

            index = int(INDEX[teeth_nums])
            jaw_points[index] = se_points
            flags[index-1] = 1
            for i in range(se_points.shape[0]):

                file_.write(str(se_points[i][0]) +" " +str(se_points[i][1]) +" " +str(se_points[i][2]) +"\n")
        #         faces = []
        #         for id in range(polydata.GetNumberOfCells()):
        #             p0_idx = polydata.GetCell(id).GetPointId(0)
        #             p1_idx = polydata.GetCell(id).GetPointId(1)
        #             p2_idx = polydata.GetCell(id).GetPointId(2)
        #             faces.append([p0_idx, p1_idx, p2_idx])
        #         faces = np.array(faces)
        #
        #         edge0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
        #         edge1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
        #         edge2 = np.stack([faces[:, 2], faces[:, 0]], axis=1)
        #         edges = np.concatenate([edge0, edge1, edge2])
        #
        if 14 == np.sum(flags):
            save_points_path = save_root + dir_name +".npy"
            # save_edges_path = save_root + dir_name + "_" + file_name + "_edges.npy"
            # np.save(save_points_path, mesh_points)
            np.save(save_points_path, jaw_points)
        else:
            print(dir_list[fi])
        file_.close()

def rand_select_train_data():
    import random
    import shutil
    file_path = "H:/teeth_arrangement_data/train_data/train/npy/"
    file_list = []
    get_files(file_path, file_list, "_end.npy")
    random.shuffle(file_list)


    train_stl_root = "H:/teeth_arrangement_data/train_data/train_split_stl/"
    test_stl_root = "H:/teeth_arrangement_data/train_data/test_split_stl/"

    save_root = "H:/teeth_arrangement_data/train_data/test/npy/"
    for i in range(len(file_list)//10):

        file_data = file_list[i]
        end_name = os.path.split(file_data)[-1]
        start_name = end_name.replace("_end.npy", "_start.npy")

        file_name = end_name.replace(".npy", "")

        start_data = file_path + start_name

        dst_start_data = save_root + start_name
        dst_end_data = save_root + end_name

        train_stl_path_end = train_stl_root + file_name
        test_stl_path_end = test_stl_root + file_name
        train_stl_path_start = train_stl_root + file_name.replace("end", "start")
        test_stl_path_start = test_stl_root + file_name.replace("end", "start")

        shutil.move(train_stl_path_start, test_stl_path_start)
        shutil.move(train_stl_path_end, test_stl_path_end)

        print("over")
        shutil.move(start_data, dst_start_data)
        shutil.move(file_data, dst_end_data)



if __name__ =="__main__":
    print("over")
    #split_teeth()
    #data_process_mtc_stl()

    train_data()
    # rand_select_train_data()