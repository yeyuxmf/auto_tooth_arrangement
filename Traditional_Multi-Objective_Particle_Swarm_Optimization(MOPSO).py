# auto_arrange_init_weights.py - 权重在 IMOPSO 初始化中配置
# 作者：AI助手 + 正畸专家
# 核心设计：
#   ✅ 权重作为 IMOPSO.__init__() 参数传入
#   ✅ 存储为 self.f2_weight, self.f3_weight, self.f4_weight
#   ✅ 所有内部方法使用实例属性

import os
import random
import json
import time
import numpy as np
import scipy.interpolate as spi
import trimesh
import matplotlib.pyplot as plt
from numpy.linalg import norm
from trimesh.collision import CollisionManager

bweight = {"1": 0, "2": 1.5, "3": 0.5, "4": 0.4, "5": 1.2, "6": 1, "7": 0.5, "8": 0.8,
           "9": 0.8, "10": 0.5, "11": 1, "12": 1.2, "13": 0.4, "14": 0.5, "15": 1.5, "16": 0}

tooth_lk = {18: ['MLCP', 'MBCP', 'DBCP'], 17: ['MLCP', 'MBCP', 'DBCP'], 16: ['MLCP', 'MBCP', 'DBCP'],
            15: ['LCP', 'BCP', 'MCP', 'DCP'], 14: ['LCP', 'BCP', 'MCP', 'DCP'],
            13: ['MCP', 'DeCP', 'DCP'], 12: ['MCP', 'CEP', 'DCP'], 11: ['MCP', 'CEP', 'DCP'],
            21: ['MCP', 'CEP', 'DCP'], 22: ['MCP', 'CEP', 'DCP'], 23: ['MCP', 'DeCP', 'DCP'],
            24: ['LCP', 'BCP', 'MCP', 'DCP'], 25: ['LCP', 'BCP', 'MCP', 'DCP'],
            26: ['MLCP', 'MBCP', 'DBCP'], 27: ['MLCP', 'MBCP', 'DBCP'], 28: ['MLCP', 'MBCP', 'DBCP']}

# -------------------- 基础配置 --------------------
cfg = dict(
    tdata_dir='./upper/',
    feature_file='./upper/upper_landmarks.json',
    p_type="FA",
    n_particle=50,
    n_iter=50,
    T_stagnation=5,
    sigma1=0.03,
    beta_levy=1.5,
    random_seed=42
)
np.random.seed(cfg['random_seed'])
random.seed(cfg['random_seed'])


# -------------------- 工具函数 --------------------
def normalize(v):
    return v / (norm(v) + 1e-12)


def dist_point2curve(p, curve_pts, curv_s):
    # s = np.linspace(*s_range, n_sample)
    # pts = curve(s)
    diff = curve_pts - p
    d2 = np.einsum('ij,ij->i', diff, diff)
    idx = np.argmin(d2)
    return np.sqrt(d2[idx]), curv_s[idx]


def levy_flight(beta, size):
    sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    return u / (np.abs(v) ** (1 / beta))


# -------------------- 数据加载 --------------------
def load_data():
    with open(cfg['feature_file'], 'r', encoding='utf-8') as f:
        feat = json.load(f)
    meshes = {}
    for tid in feat:
        mesh_path = f"{cfg['tdata_dir']}/{tid}.stl"
        if os.path.exists(mesh_path):
            meshes[tid] = trimesh.load_mesh(mesh_path)
        else:
            print(f"⚠️ 警告：{mesh_path} 不存在，跳过")
    return feat, meshes


features, meshes = load_data()
TIDS = sorted(features.keys(), key=lambda x: int(x))
N = len(TIDS)
print(f"✅ 加载 {N} 颗牙齿: {TIDS}")


def fit_3d_spline_alternative(t_param, pts, k=3, weights=None):
    if weights is None:
        weights = np.ones_like(t_param)

    splines = []
    for dim in range(3):
        # 使用 splrep 和 BSpline 组合
        tck = spi.splrep(t_param, pts[:, dim], k=k, w=weights)
        spline = spi.BSpline(*tck)
        splines.append(spline)

    def curve(t):
        return np.column_stack([spline(t) for spline in splines])

    return curve


# def build_arch_central_axis():
#     # 1. 提取原始点集 pts (N, 3)
#     pts = []
#     weights = np.zeros(len(TIDS))
#     for tid in TIDS:
#         p = features[tid]["FA"]
#         pts.append(p)
#         weights[TIDS.index(tid)] = bweight[tid]
#     pts = np.array(pts)
#
#     # 2. 参数化变量 t_param（更安全的边界处理）
#     t_param = np.linspace(0, 1, len(pts))
#
#     # 3. 拟合 arch_curve（通过所有原始点）
#     arch_curve = fit_3d_spline_alternative(t_param, pts, k=3, weights=weights)
#
#     # 4. 构造 mid_pts（x 坐标设为 0，y 和 z 保持不变）
#     mid_pts = pts.copy()
#     mid_pts[:, 0] = 0
#
#     # 5. 拟合 mid_curve（x=0 的曲线）
#     mid_curve = fit_3d_spline_alternative(t_param, mid_pts, k=3, weights=weights)
#
#     return arch_curve, mid_curve

def build_arch_central_axis():
    # 1. 提取原始点集 pts (N, 3)
    pts = []
    weights = np.zeros(len(TIDS))
    for tid in TIDS:
        p = features[tid][cfg["p_type"]]
        pts.append(p)
        weights[TIDS.index(tid)] = bweight[tid]
    pts = np.array(pts)

    t_param_raw = np.linspace(0, 1, len(pts))

    # --- 关键步骤：加权近似拟合 (splrep) ---
    # S (平滑因子): 用于控制拟合的松紧程度。S=0 为插值（但需要不重复点），
    # S > 0 为近似。此处根据点数设置一个经验值，允许一定的平滑度。
    # 如果希望拟合非常紧密，可以设置 S 为一个很小的值，如 s=1e-6
    S_FACTOR = len(pts) * 0.2

    # 3. 拟合牙弓曲线 (arch_curve)
    # splrep 对 X, Y, Z 分量分别拟合

    # X 分量
    tx, cx, kx = spi.splrep(t_param_raw, pts[:, 0], w=weights, k=3, s=S_FACTOR)
    # Y 分量
    ty, cy, ky = spi.splrep(t_param_raw, pts[:, 1], w=weights, k=3, s=S_FACTOR)
    # Z 分量
    tz, cz, kz = spi.splrep(t_param_raw, pts[:, 2], w=weights, k=3, s=S_FACTOR)

    # 4. 拟合中线 (mid_curve): x=0 投影
    # X 坐标设为 0 进行拟合
    t_mid_x, c_mid_x, k_mid_x = spi.splrep(t_param_raw, np.zeros_like(pts[:, 0]), w=weights, k=3, s=S_FACTOR)

    # 5. 将 (t, c, k) 形式转换为 BSpline 对象 (如果需要保持 make_interp_spline 的输出类型)

    # 注意：splrep 得到的 tck 形式与 make_interp_spline 得到的 BSpline 对象结构不同，
    # 且 splev 接受 tck 元组，但不直接返回 BSpline 对象。
    # 为了方便后续使用，我们直接返回 tck 元组：

    arch_tck = (tx, cx, kx, ty, cy, ky, tz, cz, kz)
    mid_tck = (t_mid_x, c_mid_x, k_mid_x, ty, cy, ky, tz, cz, kz)

    # 或者，如果您确实需要 make_interp_spline 的 BSpline 对象，
    # 您需要定义一个评估函数来包装 splev：

    arch_curve = lambda t: np.stack([
        spi.splev(np.atleast_1d(t), (tx, cx, kx)),  # 确保 t 是至少 1D 数组
        spi.splev(np.atleast_1d(t), (ty, cy, ky)),
        spi.splev(np.atleast_1d(t), (tz, cz, kz))
    ], axis=1)  # 堆叠后形状为 (N_pts, 3)

    mid_curve = lambda t: np.stack([
        spi.splev(np.atleast_1d(t), (t_mid_x, c_mid_x, k_mid_x)),
        spi.splev(np.atleast_1d(t), (ty, cy, ky)),
        spi.splev(np.atleast_1d(t), (tz, cz, kz))
    ], axis=1)  # 堆叠后形状为 (N_pts, 3)

    return arch_curve, mid_curve  # 返回的是可调用函数 (lambda)


# -------------------- 构建牙弓曲线 --------------------
# def build_arch_central_axis():
#     # try:
#     #     pts5 = np.array([
#     #         features['2']['DBCP'],
#     #         features['5']['BCP'],
#     #         features['8']['MCP'],
#     #         features['12']['BCP'],
#     #         features['15']['DBCP']
#     #     ])
#     # except KeyError as e:
#     #     print(f"❌ 特征点缺失：{e}")
#     #     raise
#     pts = []
#     for tid in TIDS:
#         p = features[tid]["FA"]
#         pts.append(p)
#     pts = np.array(pts)
#
#
#     t_param = np.linspace(0, 1, len(pts))
#     arch_curve = spi.make_interp_spline(t_param, pts, k=3)
#
#
#     mid_pts = pts.copy()
#     mid_pts[:, 0] = 0
#     mid_curve = spi.make_interp_spline(t_param, mid_pts, k=3)
#     return arch_curve, mid_curve


arch_curve, mid_curve = build_arch_central_axis()
print("✅ 牙弓曲线构建完成")


# -------------------- IMOPSO 优化器 --------------------
class IMOPSO:
    def __init__(self, arch_curve, mid_curve, f1_weight=0.1,
                 f2_weight=0.1, f3_weight=0.6, f4_weight=0.3, max_move_distance=30,
                 k_max=1.0, k_min=0.1):
        """
        初始化 IMOPSO 优化器
        参数:
            f2_weight: 对称性权重 (默认0.1)
            f3_weight: 锚点归位权重 (默认0.6)
            f4_weight: 牙弓分布权重 (默认0.3)
        """
        # 验证权重
        total_weight = f1_weight + f2_weight + f3_weight + f4_weight
        assert abs(total_weight - 1.0) < 1e-6, f"权重之和必须为1.0，当前和={total_weight}"

        self.f1_weight = f1_weight
        self.f2_weight = f2_weight
        self.f3_weight = f3_weight
        self.f4_weight = f4_weight

        print(
            f"🔧 优化器权重配置：f3={self.f3_weight:.2f}, f4={self.f4_weight:.2f}, f2={self.f2_weight:.2f}, f1={self.f1_weight:.2f}")

        self.arch_curve = arch_curve
        self.mid_curve = mid_curve
        self.n = cfg['n_particle']
        self.dim = N * 3
        self.max_iter = cfg['n_iter']
        self.T = cfg['T_stagnation']

        self.k_max = k_max
        self.k_min = k_min
        self.max_move_distance = max_move_distance
        # 原始质心
        self.original_centers = np.vstack([features[t][cfg["p_type"]] for t in TIDS])

        self.mid_curv_s = np.linspace(start=0, stop=1, num=200)
        self.mid_curv_pts = self.mid_curve(self.mid_curv_s)
        self.arch_curve_s = np.linspace(start=0, stop=1, num=500)
        self.arch_curve_pts = self.arch_curve(self.arch_curve_s)
        # 理想等分锚点
        self.s0, self.s_ideal = self.build_ideal_arch_anchors()

        # 初始化粒子群
        centers_flat = self.original_centers.ravel()
        self.x = centers_flat + np.random.uniform(-2, 2, (self.n, self.dim))
        self.v = np.zeros_like(self.x)
        self.pbest = self.x.copy()
        self.pbest_f = np.array([self.objective(p) for p in self.x])

        self.ConA, self.DivA, self.OpsA = [], [], []
        self.archive_update(first=True)
        self.stag_count = np.zeros(self.n)

        # 历史记录
        self.history_f2 = []
        self.history_f3 = []
        self.history_f4 = []

    def build_ideal_arch_anchors(self):
        """按牙位编号在牙弓曲线上等距分布理想锚点"""
        tooth_nums = np.array([int(tid) for tid in TIDS])
        min_tooth = np.min(tooth_nums)
        max_tooth = np.max(tooth_nums)

        s_ideal = (tooth_nums - min_tooth) / (max_tooth - min_tooth + 1e-8)
        s_ideal = np.clip(s_ideal, 0.0, 1.0)

        s0 = np.zeros_like(self.original_centers)

        # for i, s in enumerate(s_ideal):
        #     s0[i] = self.arch_curve(s)
        for i, t in enumerate(TIDS):
            point = np.array([features[t][cfg["p_type"]]])
            diff = self.arch_curve_pts - point
            d2 = np.einsum('ij,ij->i', diff, diff)
            idx = np.argmin(d2)
            s0[i] = self.arch_curve_pts[idx]

        # with open("./outputs/ancorpoint.txt", "w") as file_:
        #     for i in range(len(s0)):
        #         point = s0[i]
        #         file_.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

        return s0, s_ideal

    def objective(self, x):
        """🎯 三目标函数（使用实例权重）"""
        x = x.reshape(N, 3)
        f1 = 0.0

        # f2: 对称性
        left_right_pairs = [('2', '15'), ('3', '14'), ('4', '13'), ('5', '12'),
                            ('6', '11'), ('7', '10'), ('8', '9')]
        f2 = 0.0
        valid_pairs = 0
        for left, right in left_right_pairs:
            if left in TIDS and right in TIDS:
                il = TIDS.index(left)
                ir = TIDS.index(right)
                dl = dist_point2curve(x[il], self.mid_curv_pts, self.mid_curv_s)[0]
                dr = dist_point2curve(x[ir], self.mid_curv_pts, self.mid_curv_s)[0]
                f2 += abs(dl - dr)
                valid_pairs += 1
        f2 = f2 / valid_pairs if valid_pairs > 0 else 0.0

        # f3: 到理想锚点的距离
        f3 = sum(norm(x[i] - self.s0[i]) for i in range(N)) / N

        # f4: 牙弓等距分布一致性
        f4 = 0.0
        for i in range(N):
            dist_curve, s_current = dist_point2curve(x[i], self.arch_curve_pts, self.arch_curve_s)
            f4 += abs(s_current - self.s_ideal[i])
            f1 = f1 + dist_curve
        f1 = f1 / N
        # 使用实例权重进行加权

        f1_weighted = f1 * 1.0  # 基础值
        f2_weighted = f2 * 1.0  # 基础值
        f3_weighted = f3 * 1.0  # 放大倍数（强化主导地位）
        f4_weighted = f4 * 1.0  # 放大倍数

        return np.array([f1_weighted, f2_weighted, f3_weighted, f4_weighted])

    def dominates(self, a, b):
        """四目标支配判断：f1, f2, f3, f4"""
        # 目标函数数组 a, b 的结构是 [f1, f2, f3, f4]
        a_obj = np.array([a[0], a[1], a[2], a[3]])
        b_obj = np.array([b[0], b[1], b[2], b[3]])
        # 当且仅当 a 不比 b 差，并且至少在一个目标上优于 b 时，a 支配 b
        return np.all(a_obj <= b_obj) and np.any(a_obj < b_obj)

    def archive_update(self, first=False):
        pop_f = np.array([self.objective(p) for p in self.x])
        if first:
            pool = list(zip(self.x, pop_f))
        else:
            pool = list(zip(self.x, pop_f)) + list(zip(self.pbest, self.pbest_f))

        # 非支配排序
        nd_indices = []
        for i, (_, fi) in enumerate(pool):
            dominated = False
            for j, (_, fj) in enumerate(pool):
                if i != j and self.dominates(fj, fi):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)

        nd = [pool[i] for i in nd_indices] if nd_indices else pool[:max(1, len(pool) // 2)]

        def sort_cut(arr, key, size):
            arr_sorted = sorted(arr, key=key)
            return arr_sorted[:size]

        size = max(1, len(nd) // 4)

        # ConA: 最小化 f3 和 f1（锚点归位 + 贴合度）→ 收敛导向
        # 修正：将 f1 和 f3 加权 (因为它们都代表收敛)
        self.ConA = sort_cut(nd, lambda t: self.f3_weight * t[1][2] + self.f1_weight * t[1][0], size)

        # DivA: 最大化 f2（鼓励不对称探索）
        self.DivA = sort_cut(nd, lambda t: -t[1][1], size)

        # OpsA: 使用实例权重加权综合
        self.OpsA = sort_cut(nd, lambda t: self.f1_weight * t[1][0] + self.f2_weight * t[1][1] + self.f3_weight * t[1][
            2] + self.f4_weight * t[1][3], size)

    def run(self):
        """执行优化（全链路使用实例权重）"""
        for g in range(self.max_iter):
            Cgb = np.zeros(self.dim)
            Dgb = np.zeros(self.dim)

            for d in range(self.dim):
                # Cgb: 从 ConA（f3最优）中选
                if len(self.ConA) > 0:
                    pc = random.choice(self.ConA)[0]
                    Cgb[d] = pc[d]
                else:
                    # 修正：备用方案也应该考虑 F1 和 F3 的综合最优
                    best_idx = np.argmin(self.pbest_f[:, 2] * self.f3_weight + self.pbest_f[:, 0] * self.f1_weight)
                    Cgb[d] = self.pbest[best_idx][d]

                # Dgb: 从 DivA（f2最差）中选
                if len(self.DivA) > 0:
                    pd = random.choice(self.DivA)[0]
                    Dgb[d] = pd[d]
                else:
                    worst_idx = np.argmax(self.pbest_f[:, 1])  # f2最大
                    Dgb[d] = self.pbest[worst_idx][d]

            for i in range(self.n):
                fit_old = self.objective(self.x[i])

                # 停滞检测：使用实例权重计算加权改进
                if g > self.T:
                    old_weighted = (self.f1_weight * fit_old[0] +  # 👈 修正：纳入 f1
                                    self.f3_weight * fit_old[2] +
                                    self.f4_weight * fit_old[3] +
                                    self.f2_weight * fit_old[1])
                    pbest_weighted = (self.f1_weight * self.pbest_f[i][0] +  # 👈 修正：纳入 f1
                                      self.f3_weight * self.pbest_f[i][2] +
                                      self.f4_weight * self.pbest_f[i][3] +
                                      self.f2_weight * self.pbest_f[i][1])

                    improvement = abs(old_weighted - pbest_weighted)
                    if improvement < 1e-4:
                        self.stag_count[i] += 1
                    else:
                        self.stag_count[i] = 0
                else:
                    self.stag_count[i] = 0

                # Levy扰动：帮助f3最差的粒子
                if self.stag_count[i] >= self.T and g > 0.5 * self.max_iter:
                    if len(self.DivA) > 0:
                        worst_f3_idx = np.argmax([t[1][2] for t in self.DivA])
                        elite = self.DivA[worst_f3_idx][0]
                    else:
                        worst_idx = np.argmax(self.pbest_f[:, 2])  # f3最大
                        elite = self.pbest[worst_idx]

                    S = levy_flight(cfg['beta_levy'], self.dim) * (elite - self.pbest[i])
                    self.x[i] = self.pbest[i] + S
                    self.v[i] = np.zeros(self.dim)
                else:
                    w = 0.9 - 0.5 * g / self.max_iter
                    c1, c2, c3 = 1.5, 0.8, 0.8
                    r1, r2, r3 = np.random.rand(3)

                    x_reshaped = self.x[i].reshape(N, 3)
                    distance_to_anchor = norm(x_reshaped - self.s0, axis=-1)

                    # 计算基础速度分量
                    cognitive = c1 * r1 * (self.pbest[i] - self.x[i])
                    social_converge = c2 * r2 * (Cgb - self.x[i])
                    social_diverse = c3 * r3 * (Dgb - self.x[i])
                    base_velocity = w * self.v[i] + cognitive + social_converge + social_diverse

                    # ✅ 动态k值策略
                    progress = g / self.max_iter
                    base_k = self.k_max - (self.k_max - self.k_min) * progress

                    # 距离归一化
                    max_dist = np.max(distance_to_anchor) + 1e-8
                    distance_normalized = distance_to_anchor / max_dist

                    # 自适应k值：距离大的牙用更大的k
                    adaptive_k_per_tooth = base_k * (0.5 + 0.5 * distance_normalized)

                    # 计算最大允许速度
                    max_allowed_speed_per_tooth = distance_to_anchor * adaptive_k_per_tooth

                    # 最小速度保障
                    min_speed = 0.01
                    max_allowed_speed_per_tooth = np.maximum(max_allowed_speed_per_tooth, min_speed)

                    # 计算每颗牙当前速度向量的大小
                    velocity_magnitudes = np.zeros(N)
                    for j in range(N):
                        start_idx = j * 3
                        end_idx = start_idx + 3
                        velocity_magnitudes[j] = norm(base_velocity[start_idx:end_idx])

                    # 计算缩放因子
                    scale_factors = np.ones(N)
                    for j in range(N):
                        if velocity_magnitudes[j] > max_allowed_speed_per_tooth[j]:
                            scale_factors[j] = max_allowed_speed_per_tooth[j] / velocity_magnitudes[j]

                    # 应用缩放
                    scale_multiplier = np.repeat(scale_factors, 3)
                    self.v[i] = base_velocity * scale_multiplier
                    self.x[i] += self.v[i]

                    # 相对锚点的边界约束（向量化）
                    x_reshaped = self.x[i].reshape(N, 3)
                    min_bound = self.s0 - self.max_move_distance
                    max_bound = self.s0 + self.max_move_distance
                    x_reshaped = np.clip(x_reshaped, min_bound, max_bound)
                    self.x[i] = x_reshaped.ravel()

                self.x[i] = np.clip(self.x[i], -30, 30)

                # pbest更新：使用支配关系
                fit_new = self.objective(self.x[i])
                if self.dominates(fit_new, self.pbest_f[i]):
                    self.pbest[i] = self.x[i].copy()
                    self.pbest_f[i] = fit_new

            self.archive_update()

            if len(self.OpsA) > 0:
                best_f = self.OpsA[0][1]
                self.history_f2.append(best_f[1])
                self.history_f3.append(best_f[2])
                self.history_f4.append(best_f[3])
                print(
                    f'🔄 迭代 {g:04d} | f1={best_f[0]:.4f} |f2={best_f[1]:.4f} | f3={best_f[2]:.4f} | f4={best_f[3]:.4f}')
            else:
                print(f'🔄 迭代 {g:03d} | 归档为空！')

        best_x = self.OpsA[0][0] if len(self.OpsA) > 0 else self.pbest[0]
        return best_x.reshape(N, 3)


# -------------------- 旋转相关 --------------------
def build_local_frame(tid):
    A = np.array(features[tid]['A'])
    B = np.array(features[tid]['B'])
    C = np.array(features[tid]['C'])
    Z = normalize(B - A)
    Y = normalize(np.cross(Z, C - B))
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z], axis=1)


def build_arch_frame(p, arch_curve):
    # 建立采样的曲点供搜索
    s_samples = np.linspace(0, 1, 1000)
    pts_samples = arch_curve(s_samples)
    _, s = dist_point2curve(p, pts_samples, s_samples)  # 👈 修复：传入采样点
    h = 1e-4
    T = normalize(arch_curve(s + h) - arch_curve(s - h))
    N_vec = normalize(np.array([T[1], -T[0], 0]))
    B = np.cross(T, N_vec)
    return np.stack([N_vec, B, T], axis=1)


def rotation_matrix_from_frames(R_local, R_arch):
    return R_arch @ R_local.T


# -------------------- 碰撞检测 --------------------
def collision_pairs(meshes, positions, TIDS):
    mgr = CollisionManager()
    mesh_objs = {}
    for i, tid in enumerate(TIDS):
        m = meshes[tid].copy()
        m.apply_translation(positions[i] - m.centroid)
        mgr.add_object(tid, m)
        mesh_objs[tid] = m

    collide = []
    for i in range(len(TIDS)):
        for j in range(i + 1, len(TIDS)):
            tid1 = TIDS[i]
            tid2 = TIDS[j]
            if abs(int(tid1) - int(tid2)) <= 3:
                if mgr.in_collision_single(mesh_objs[tid1], mesh_objs[tid2]):
                    collide.append((tid1, tid2, 1))
    return collide


def adjust(positions, TIDS, arch_curve, step_size=cfg['sigma1']):
    # 建立采样的曲点供搜索
    s_ref = np.linspace(0, 1, 500)
    pts_ref = arch_curve(s_ref)

    for iter in range(50):
        coll = collision_pairs(meshes, positions, TIDS)
        if not coll:
            print(f"✅ 碰撞消除完成，共迭代 {iter} 次")
            break
        for tid1, tid2, _ in coll:
            i1 = TIDS.index(tid1)
            i2 = TIDS.index(tid2)
            p1 = positions[i1]
            p2 = positions[i2]
            _, s1 = dist_point2curve(p1, pts_ref, s_ref)  # 👈 修复：传入采样点
            T1 = normalize(arch_curve(s1 + 0.01) - arch_curve(s1 - 0.01))
            _, s2 = dist_point2curve(p2, pts_ref, s_ref)  # 👈 修复：传入采样点
            T2 = normalize(arch_curve(s2 + 0.01) - arch_curve(s2 - 0.01))
            positions[i1] -= T1 * step_size * 0.5
            positions[i2] += T2 * step_size * 0.5
    return positions


# -------------------- 主流程 --------------------
def main():
    print('=== 🚀 创建 IMOPSO 优化器 ===')

    # ✅ 在这里设置权重！
    pso = IMOPSO(
        arch_curve,
        mid_curve,
        f1_weight=0.5,  # 靠近曲线
        f2_weight=0.1,  # 对称性
        f3_weight=0.3,  # 锚点归位（主导）
        f4_weight=0.1  # 牙弓分布（辅助）
    )

    centers = pso.run()

    out_dir = "./outputs/"
    os.makedirs(out_dir, exist_ok=True)

    # 导出STL
    for i, tid in enumerate(TIDS):
        m = meshes[tid].copy()
        delta = centers[i] - features[tid][cfg["p_type"]]
        m.apply_translation(delta)
        m.export(os.path.join(out_dir, f"{tid}.stl"))
    print(f"✅ 已导出 {N} 颗排列后 STL")

    # 保存牙弓曲线
    s500 = np.linspace(0, 1, 500)
    pts500 = arch_curve(s500)
    np.savetxt(os.path.join(out_dir, "arch_curve_500pts.txt"), pts500, fmt="%.6f")

    with open("./outputs/ancorpoint.txt", "w") as file_:
        for i in range(len(pso.s0)):
            point = pso.s0[i]
            file_.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

    with open("./outputs/FApoint.txt", "w") as file_:
        for i in range(len(centers)):
            point = centers[i]
            file_.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

    # 计算旋转
    print('=== 🔄 计算旋转 ===')
    rotations = []
    for i, tid in enumerate(TIDS):
        R_loc = build_local_frame(tid)
        R_arch = build_arch_frame(centers[i], arch_curve)
        R = rotation_matrix_from_frames(R_loc, R_arch)
        rotations.append(R)

    # 碰撞微调
    print('=== 🔧 碰撞微调 ===')
    centers = adjust(centers, TIDS, arch_curve)

    # 保存结果
    np.save(os.path.join(out_dir, 'centers.npy'), centers)
    np.save(os.path.join(out_dir, 'rotations.npy'), np.array(rotations))

    # 分析移动距离
    original_centers = np.vstack([features[t][cfg["p_type"]] for t in TIDS])
    moves = centers - original_centers
    move_norms = np.linalg.norm(moves, axis=1)

    print("\n📊 === 每颗牙齿移动距离 ===")
    total_move = 0
    for i, tid in enumerate(TIDS):
        print(f"🦷 牙位 {tid}: {move_norms[i]:.3f} mm")
        total_move += move_norms[i]
    print(f"📈 总移动量: {total_move:.3f} mm")

    np.savetxt(os.path.join(out_dir, 'tooth_movements.txt'), move_norms, fmt='%.3f')

    # 绘制优化历史
    plt.figure(figsize=(15, 4))

    plt.subplot(131)
    plt.plot(pso.history_f3, 'g-^', markersize=4, linewidth=2)
    plt.title(f'f3: Anchor Deviation\n(权重={pso.f3_weight:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('Distance (mm)')
    plt.grid(True)

    plt.subplot(132)
    plt.plot(pso.history_f4, 'b-o', markersize=4, linewidth=2)
    plt.title(f'f4: Arch Distribution\n(权重={pso.f4_weight:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('Deviation')
    plt.grid(True)

    plt.subplot(133)
    plt.plot(pso.history_f2, 'r-s', markersize=4, linewidth=2)
    plt.title(f'f2: Symmetry\n(权重={pso.f2_weight:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('Asymmetry (mm)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 打印最终分析
    print("\n" + "=" * 70)
    print("🎯 最终排牙质量分析")
    print("=" * 70)
    total_f3 = 0
    total_f4 = 0
    # 获取用于分析的采样点
    s_anal = np.linspace(0, 1, 1000)
    pts_anal = arch_curve(s_anal)
    for i, tid in enumerate(TIDS):
        d_to_anchor = norm(centers[i] - pso.s0[i])
        _, s_current = dist_point2curve(centers[i], pts_anal, s_anal)
        s_dev = abs(s_current - pso.s_ideal[i])

        total_f3 += d_to_anchor
        total_f4 += s_dev

        print(f"🦷 {tid:>3} | 到锚点距离: {d_to_anchor:.4f}mm | s值偏差: {s_dev:.4f}")

    print(f"\n📊 总体指标:")
    print(f"   总锚点偏差: {total_f3:.4f} mm")
    print(f"   总分布偏差: {total_f4:.4f}")
    print(f"   平均每颗牙锚点偏差: {total_f3 / N:.4f} mm")


if __name__ == '__main__':
    main()
