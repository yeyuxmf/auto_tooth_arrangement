import os
import json
import numpy as np
import scipy.interpolate as spi
import trimesh
from scipy.optimize import minimize
from numpy.linalg import norm
from scipy.spatial import KDTree
from typing import Dict, List, Tuple
from test_collision import ToothCollisionEngine
engine = ToothCollisionEngine("./mesh_collision/x64/Release/mesh_collision.dll")
tooth_num = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
# -------------------- 1. 基础配置 (调参中心) --------------------
class Config:
    tdata_dir = './lower/'
    feature_file = './lower/lower_landmarks.json'
    out_dir = "./outputs/"
    p_type = "FA"
    max_iter = 100
    random_seed = 42

    # --- 关键设置：排牙框架模式 ---
    # 设置为 14 则按 2-15 号排，设置为 16 则按 1-16 号排
    total_teeth_mode = 14

    # 能量项初始权重
    weights_init = {"f1": 0.40625,"f2": 0.0625,"f3": 0.25,"f4": 0.09375,"f5": 0.001,"f6": 0.001,
                          "f7": 0.1875}#"f7": 0.1875

    # 牙齿个体权重系数
    bweight = {"1": 0, "2": 1.5, "3": 0.5, "4": 0.4, "5": 1.2, "6": 1, "7": 0.5, "8": 0.8,
               "9": 0.8, "10": 0.5, "11": 1, "12": 1.2, "13": 0.4, "14": 0.5, "15": 1.5, "16": 0}


def angle_between(m_proj, T, B):
    # 1. 计算单位向量
    T = T / np.linalg.norm(T)
    m_proj = m_proj / np.linalg.norm(m_proj)
    B = B / np.linalg.norm(B)

    # 计算叉积和点积
    cross_product = np.cross(m_proj, T)
    sin_theta = np.dot(cross_product, B)  # 旋转方向的正负
    cos_theta = np.dot(m_proj, T)  # 夹角的余弦值

    # 计算角度（atan2 处理所有象限）
    theta = np.arctan2(sin_theta, cos_theta)

    return theta
# -------------------- 2. 工具函数库 --------------------
class Utils:
    @staticmethod
    def normalize(v):
        return v / (norm(v) + 1e-12)

    @staticmethod
    def dist_point2curve(p, curve_pts, curv_s):
        diff = curve_pts - p
        d2 = np.einsum('ij,ij->i', diff, diff)
        idx = np.argmin(d2)
        return np.sqrt(d2[idx]), curv_s[idx]

    @staticmethod
    def signed_distance_to_mesh(point, mesh):
        tree = KDTree(mesh.vertices)
        d_min, _ = tree.query(point, k=1)
        return d_min


# -------------------- 3. 牙齿实体类 --------------------
class ToothEntity:
    def __init__(self, tid: str, feat: dict, mesh: trimesh.Trimesh):
        self.tid = tid
        self.tid_num = int(tid)
        self.features = feat
        self.mesh = mesh
        self.initial_fa = np.array(feat[Config.p_type])
        self.obb_initial = mesh.bounding_box_oriented.to_mesh()


        self.intrinsic_m = None
        self._compute_lcs()



    def _compute_lcs(self):
        f = self.features
        m_raw = np.array(f["MCP"]) - np.array(f["DCP"])
        if 7 <= self.tid_num <= 10:
            b_raw_unnorm = np.array(f["FA"]) - np.array(f["CEP"])
        elif self.tid_num in [6, 11]:
            b_raw_unnorm = np.array(f["FA"]) - self.mesh.centroid
        elif self.tid_num in [4, 5, 12, 13]:
            b_raw_unnorm = np.array(f["BCP"]) - np.array(f["LCP"])
        elif self.tid_num in [2, 3, 14, 15]:
            b_raw_unnorm = np.array(f["FA"]) - np.array(f["CFP"])
        else:
            b_raw_unnorm = np.array([0, 1, 0])

        m_vec = Utils.normalize(m_raw)
        if self.tid_num >= 9: m_vec = -m_vec
        b_vec = Utils.normalize(np.cross(m_vec, [0, 0, 1]))
        if np.dot(b_vec, Utils.normalize(b_raw_unnorm)) < 0: b_vec = -b_vec
        self.intrinsic_m = m_vec


# -------------------- 4. 牙弓系统类 --------------------
class ArchSystem:
    def __init__(self, teeth: List[ToothEntity]):
        self.teeth = teeth
        self.arch_fn, self.mid_fn = self._fit_arch_curves()
        self.s_samples = np.linspace(0, 1, 500)
        self.pts = self.arch_fn(self.s_samples)
        self.mid_pts = self.mid_fn(np.linspace(0, 1, 200))
        self.tangents = np.zeros_like(self.pts)
        self.normals = np.zeros_like(self.pts)  # 主法线 (指向曲线弯曲方向，即颊舌向)
        self.binormals = np.zeros_like(self.pts)  # 副法线 (垂直于曲线所在局部平面)
        self._precompute_frames()

    def _fit_arch_curves(self):
        pts = np.array([t.initial_fa for t in self.teeth])
        weights = np.array([Config.bweight.get(t.tid, 1.0) for t in self.teeth])
        t_param = np.linspace(0, 1, len(pts))
        s_factor = len(pts) * 0.3

        def get_spline(data, w):
            tx, cx, kx = spi.splrep(t_param, data[:, 0], w=w, k=3, s=s_factor)
            ty, cy, ky = spi.splrep(t_param, data[:, 1], w=w, k=3, s=s_factor)
            tz, cz, kz = spi.splrep(t_param, data[:, 2], w=w, k=3, s=s_factor)
            return lambda s: np.stack([spi.splev(np.atleast_1d(s), (tx, cx, kx)),
                                       spi.splev(np.atleast_1d(s), (ty, cy, ky)),
                                       spi.splev(np.atleast_1d(s), (tz, cz, kz))], axis=1)

        arch_fn = get_spline(pts, weights)
        mid_pts = pts.copy()
        mid_pts[:, 0] = 0
        mid_fn = get_spline(mid_pts, weights)
        return arch_fn, mid_fn

    def _precompute_frames(self):
        n = len(self.pts)
        valid_b_mask = np.zeros(n, dtype=bool)

        # --- 第一步：初步计算切线 T 并寻找所有有效弯曲处的副法线 B ---
        for i in range(n):
            idx_prev = max(0, i - 2)
            idx_next = min(n - 1, i + 2)

            p_curr = self.pts[i]
            p_prev = self.pts[idx_prev]
            p_next = self.pts[idx_next]

            # 计算切线 T
            T = Utils.normalize(p_next - p_prev)
            self.tangents[i] = T

            # 计算局部弯曲决定的 B
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            B_raw = np.cross(v1, v2)
            B_norm = np.linalg.norm(B_raw)

            # 只有当 V1, V2 不共线时才记录 B
            if B_norm > 1e-8:
                self.binormals[i] = B_raw / B_norm
                valid_b_mask[i] = True

        # --- 第二步：插值填充缺失的 B (针对直线段) ---
        # 1. 处理起始端的直线：找到第一个有效的 B，并向头部填充
        first_valid_idx = np.where(valid_b_mask)[0][0]
        if first_valid_idx > 0:
            self.binormals[:first_valid_idx] = self.binormals[first_valid_idx]
            valid_b_mask[:first_valid_idx] = True

        # 2. 处理中间及末端的空隙：使用前向填充 (Forward Fill)
        # 对于牙弓线，前向填充足以保证平面的连续性
        for i in range(1, n):
            if not valid_b_mask[i]:
                self.binormals[i] = self.binormals[i - 1]
                valid_b_mask[i] = True

        # --- 第三步：最终计算主法线 N 并确保正交序列 ---
        for i in range(n):
            T = self.tangents[i]
            B = Utils.normalize(self.binormals[i])  # 再次归一化确保精度

            # N = B x T
            # 这样即便 T 发生了细微变化，坐标系依然保持严格正交
            N = np.cross(B, T)

            self.binormals[i] = B
            self.normals[i] = N

# -------------------- 5. 核心优化器类 --------------------
class ArrangementOptimizer:
    def __init__(self, teeth: List[ToothEntity], arch: ArchSystem):
        self.teeth = teeth
        self.arch = arch
        self.N = len(teeth)
        self.original_centers = np.vstack([t.initial_fa for t in teeth])

        # --- 【严谨逻辑：根据排牙框架模式映射 s_ideal】 ---
        if Config.total_teeth_mode == 14:
            # 14牙模式：编号 2-15，映射到 [0, 1]
            self.s_ideal = np.array([(t.tid_num - 2) / 13.0 for t in teeth])
        else:
            # 16牙模式：编号 1-16，映射到 [0, 1]
            self.s_ideal = np.array([(t.tid_num - 1) / 15.0 for t in teeth])


        self.s0 = np.zeros_like(self.original_centers)
        for i in range(self.N):
            idx = np.argmin(norm(self.arch.pts - self.original_centers[i], axis=1))
            self.s0[i] = self.arch.pts[idx]

        self.call_count, self.last_print_iter = 0, -1

    def _get_dynamic_weights(self, iter_idx):
        w = Config.weights_init
        if iter_idx < 1: return w["f1"], w["f2"], w["f3"], w["f4"], w["f5"], w["f6"], w["f7"]
        prog = np.clip((iter_idx - 1) / (Config.max_iter - 1), 0.0, 1.0)
        alpha = 1.0 - 0.5 * (prog ** 1.0)
        new_f3 = w["f3"] * alpha
        delta = (w["f3"] - new_f3) / 3.0
        return w["f1"] + delta, w["f2"], new_f3, w["f4"], w["f5"] + delta, w["f6"] + delta, w["f7"]

    # def collision_gap_goals(self, centers, thetas, tids):
    #     F5_E, F6_E, margin_max = 0, 0, 0.1
    #     for i in range(self.N):
    #         delta1 = centers[i] - self.teeth[i].initial_fa
    #         R1 = trimesh.transformations.rotation_matrix(thetas[i], [0, 1, 0], self.teeth[i].initial_fa)
    #         obb_i = self.teeth[i].obb_initial.copy().apply_transform(R1).apply_translation(delta1)
    #         if i + 1 < self.N:
    #             j = i + 1
    #             delta2 = centers[j] - self.teeth[j].initial_fa
    #             R2 = trimesh.transformations.rotation_matrix(thetas[j], [0, 1, 0], self.teeth[j].initial_fa)
    #             obb_j = self.teeth[j].obb_initial.copy().apply_transform(R2).apply_translation(delta2)
    #             dists = [Utils.signed_distance_to_mesh(v, obb_j) for v in obb_i.vertices]
    #             dists += [Utils.signed_distance_to_mesh(v, obb_i) for v in obb_j.vertices]
    #             d_min = np.min(dists)
    #             if abs(self.teeth[i].tid_num - self.teeth[j].tid_num) == 1:
    #                 F6_E += abs(d_min - margin_max)
    #             elif d_min < 0:
    #                 F5_E += abs(d_min)
    #     return F5_E / self.N, F6_E / self.N
    def collision_gap_goals(self, centers, thetas, tids):
        F5_E, F6_E, margin_max = 0, 0, 0.1

        for ki, tid1 in enumerate(tooth_num):
            if tid1 not in tids:
                continue
            i= tids.index(tid1)
            delta1 = centers[i] - self.teeth[i].initial_fa
            idx = np.argmin(norm(self.arch.pts - centers[i], axis=1))
            B = self.arch.binormals[idx]
            mat1 = trimesh.transformations.rotation_matrix(thetas[i], B, self.teeth[i].initial_fa)
            mat1[0:3, 3] += delta1
            for kj in range(ki +1, min(len(tooth_num), ki + 2)):
                tid2 = tooth_num[kj]
                if tid2 not in tids:
                    continue
                j = tids.index(tid2)
                delta2 = centers[j] - self.teeth[j].initial_fa
                idx = np.argmin(norm(self.arch.pts - centers[j], axis=1))
                B = self.arch.binormals[idx]
                mat2 = trimesh.transformations.rotation_matrix(thetas[j], B, self.teeth[j].initial_fa)
                mat2[0:3, 3] += delta2

                collided1, dists = engine.compute_collision(int(tid1), mat1, int(tid2), mat2)

                d_min = np.min(dists)
                if dists >0.000001:
                    F6_E += abs(d_min - margin_max)
                elif d_min < 0:
                    F5_E += abs(d_min)

        return F5_E / self.N, F6_E / self.N
    def get_rot_vec(self, intrinsic_m, thetas, idx):

        T = self.arch.tangents[idx]  # 切线 (局部 Z)
        N = self.arch.normals[idx]  # 法线 (局部 X)
        B = self.arch.binormals[idx]  # 副法线 (局部 Y, 旋转轴)

        # 2. 将牙齿固有方向 m_orig 投影到局部坐标系下
        # 通过点积求出 m_orig 在局部三个轴上的分量
        m_orig = intrinsic_m
        vec = np.dot(m_orig, B) * B
        m_proj = m_orig - vec

        axis = B / np.linalg.norm(B)

        thetas2 = angle_between(m_proj, T, axis)


        # 罗德里格斯公式
        cos_theta = np.cos(thetas2)
        sin_theta = np.sin(thetas2)
        cross_term = np.cross(axis, m_proj)

        curr_m = (
                m_proj * cos_theta +
                cross_term * sin_theta +
                axis * np.dot(axis, m_proj) * (1 - cos_theta)
        )


        return T, curr_m, thetas2

    def objective(self, x_flat):
        centers = x_flat[:3 * self.N].reshape(self.N, 3)
        thetas = x_flat[3 * self.N:]
        iter_idx = self.call_count // (len(x_flat) + 1)
        w1, w2, w3, w4, w5, w6, w7 = self._get_dynamic_weights(iter_idx)

        f1, f4, f7 = 0, 0, 0
        for i, t in enumerate(self.teeth):
            d_c, s_c = Utils.dist_point2curve(centers[i], self.arch.pts, self.arch.s_samples)
            f1 += d_c
            f4 += abs(s_c - self.s_ideal[i])
            idx = np.argmin(norm(self.arch.pts - centers[i], axis=1))

            T, curr_m, thetas2 =  self.get_rot_vec(t.intrinsic_m, thetas[i], idx)
            # --- 核心修改 ---
            diff = thetas[i] - thetas2
            # 保证弧度差值在 [-pi, pi] 之间
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            f7 += (diff ** 2)

        f2, v_p = 0, 0
        tids = [tooth.tid for tooth in self.teeth]
        for l, r in [('8', '9'), ('7', '10')]:
            if l in tids and r in tids:
                il, ir = tids.index(l), tids.index(r)
                dl, _ = Utils.dist_point2curve(centers[il], self.arch.mid_pts, np.linspace(0, 1, 200))
                dr, _ = Utils.dist_point2curve(centers[ir], self.arch.mid_pts, np.linspace(0, 1, 200))
                f2 += abs(dl - dr);
                v_p += 1

        f3 = np.mean(norm(centers - self.s0, axis=1))
        f5, f6 = self.collision_gap_goals(centers, thetas, tids)
        total = (w1 * (f1 / self.N) + w2 * (f2 / (v_p if v_p > 0 else 1)) + w3 * f3 + w4 * (
                    f4 / self.N) + w5 * f5 + w6 * f6 + w7 * (f7 / self.N))

        if iter_idx > self.last_print_iter:
            print(
                f"Iter {iter_idx:03d} | Total:{total:.4f} | f1:{f1 / self.N:.4f} f2:{f2 / (v_p if v_p > 0 else 1):.4f} f3:{f3:.4f} f4:{f4 / self.N:.4f} f5:{f5:.4f} f6:{f6:.4f} f7:{f7 / self.N:.4f} | Weights(w1/w2/w3/w4/w5/w6):{w1:.2f}/{w2:.2f}/{w3:.2f}/{w4:.2f}/{w5:.2f}/{w6:.2f}/{w7:.2f}/")
            self.last_print_iter = iter_idx
        self.call_count += 1
        return total



# -------------------- 6. 执行与导出 --------------------
def main():
    with open(Config.feature_file, 'r', encoding='utf-8') as f:
        feat_data = json.load(f)
    teeth_entities = []
    engine.initialize(Config.tdata_dir)
    for tid in sorted(feat_data.keys(), key=int):
        path = os.path.join(Config.tdata_dir, f"{tid}.stl")
        if os.path.exists(path):
            teeth_entities.append(ToothEntity(tid, feat_data[tid], trimesh.load_mesh(path)))

    arch_sys = ArchSystem(teeth_entities)
    optimizer = ArrangementOptimizer(teeth_entities, arch_sys)



    np.savetxt(os.path.join(Config.out_dir, "FA_point.txt"),   optimizer.original_centers, fmt="%.6f")
    np.savetxt(os.path.join(Config.out_dir, "arch_curve_500pts.txt"), arch_sys.pts, fmt="%.6f")

    # FA_Point = 修正 run 函数内部的小引用错误，直接在 main 里写优化调用逻辑
    x0 = np.concatenate([optimizer.original_centers.ravel(), np.zeros(optimizer.N)])
    bounds = [(p - 30, p + 30) for p in x0[:3 * optimizer.N]] + [(-np.pi / 2, np.pi / 2) for _ in range(optimizer.N)]
    res = minimize(optimizer.objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': Config.max_iter, 'factr': 0, 'ftol': 0})
    final_pos, final_rot = res.x[:3 * optimizer.N].reshape(optimizer.N, 3), res.x[3 * optimizer.N:]

    os.makedirs(Config.out_dir, exist_ok=True)
    axis_len = np.linspace(0, 7, 50)
    f_tm = open(os.path.join(Config.out_dir, "tooth_mvec.txt"), "w")
    for i, tooth in enumerate(teeth_entities):
        m = tooth.mesh.copy()
        m.apply_translation(final_pos[i] - tooth.initial_fa)

        idx = np.argmin(norm(optimizer.arch.pts - final_pos[i], axis=1))
        pts = optimizer.arch.pts[idx]
        B = optimizer.arch.binormals[idx]  # Y轴：副法线（垂直于牙弓平面，旋转轴）

        m.apply_transform(trimesh.transformations.rotation_matrix(final_rot[i], B, final_pos[i]))
        m.export(os.path.join(Config.out_dir, f"{tooth.tid}.stl"))
        R = trimesh.transformations.rotation_matrix(final_rot[i], B)[:3, :3]
        T, curr_m, thetas2 = optimizer.get_rot_vec(tooth.intrinsic_m, final_rot[i], idx)

        for p in (axis_len.reshape(-1, 1) * curr_m.reshape(1, 3) + pts): f_tm.write(f"{p[0]} {p[1]} {p[2]}\n")
    f_tm.close()

    f_am = open(os.path.join(Config.out_dir, "arch_mvec.txt"), "w")
    for pos in final_pos:
        idx = np.argmin(norm(arch_sys.pts - pos, axis=1))
        for p in (axis_len.reshape(-1, 1) * arch_sys.tangents[idx].reshape(1, 3) + arch_sys.pts[idx]): f_am.write(
            f"{p[0]} {p[1]} {p[2]}\n")
    f_am.close()

    print(f"✅ 优化导出完成！目标目录: {Config.out_dir}")


if __name__ == '__main__':


    main()
