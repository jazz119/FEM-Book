import json
import numpy as np

# ====================== 工具辅助函数（全部重构优化） ======================
def check_symmetric(K, tol=1e-10):
    """校验刚度矩阵对称性"""
    diff = np.max(np.abs(K - K.T))
    return diff < tol, diff

def is_singular(mat, tol=1e-10):
    """稳定判奇异：用秩判断，替代易溢出的行列式"""
    if mat.shape[0] != mat.shape[1]:
        return True, -1.0
    rank = np.linalg.matrix_rank(mat, tol=tol)
    singular = rank < mat.shape[0]
    det = np.linalg.det(mat) if mat.shape[0] <= 10 else np.nan
    return singular, det

def check_force_balance(f_global, R, fixed_dofs, free_dofs, ndof, tol=1e-8):
    """分方向平衡校验（1D/2D通用）"""
    total_dof = len(f_global)
    f_total = np.zeros(ndof)
    r_total = np.zeros(ndof)
    # 累加外载荷
    for dof in free_dofs:
        axis = dof % ndof
        f_total[axis] += f_global[dof]
    # 累加支座反力
    for idx, dof in enumerate(fixed_dofs):
        axis = dof % ndof
        r_total[axis] += R[idx]
    # 合力 f + R ≈ 0
    residual = f_total + r_total
    balance_ok = np.all(np.abs(residual) < tol)
    return residual, balance_ok, f_total, r_total

# ====================== 1. 前处理模块 ======================
def load_model(json_str=None, json_path=None):
    if json_path is not None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(json_str)
    return data

def build_LM(model):
    nnp = model["nnp"]
    ndof = model["ndof"]
    nel = model["nel"]
    nen = model["nen"]
    IEN = np.array(model["IEN"], dtype=int) - 1
    n_local_dof = nen * ndof
    LM = np.zeros((n_local_dof, nel), dtype=int)
    for e in range(nel):
        nodes = IEN[e]
        local_d = 0
        for node in nodes:
            for d in range(ndof):
                LM[local_d, e] = node * ndof + d
                local_d += 1
    return LM

def get_total_dof(model):
    return model["nnp"] * model["ndof"]

# ====================== 2. 单元模块（增加浮点保护） ======================
def element_1d_bar(x1, x2, E, A):
    dx = x2 - x1
    L = abs(dx)
    if L < 1e-12:
        raise ValueError(f"单元长度过小 L={L:.2e}，单元退化")
    ke_coeff = E * A / L
    Ke = ke_coeff * np.array([[1, -1], [-1, 1]])
    return Ke, L

def stress_1d(E, A, L, de):
    sigma = E / L * (-de[0] + de[1])
    N = sigma * A
    return sigma, N

def element_2d_truss(x1, y1, x2, y2, E, A):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-12:
        raise ValueError(f"单元长度过小 L={L:.2e}，单元退化")
    c = dx / L
    s = dy / L
    coeff = E * A / L
    Ke = coeff * np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])
    return Ke, L, c, s

def stress_2d(E, A, L, c, s, de):
    sigma = E / L * (-c*de[0] - s*de[1] + c*de[2] + s*de[3])
    N = sigma * A
    return sigma, N

# ====================== 3. 总体刚度组装 ======================
def assemble_global_K(model, LM):
    ndof_tot = get_total_dof(model)
    K = np.zeros((ndof_tot, ndof_tot), dtype=np.float64)
    x = np.array(model["x"])
    y = np.array(model["y"])
    IEN = np.array(model["IEN"], dtype=int) - 1
    E_list = model["E"]
    A_list = model["CArea"]
    nel = model["nel"]
    ndof = model["ndof"]

    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = x[n1], y[n1]
        x2, y2 = x[n2], y[n2]
        E = E_list[e]
        A = A_list[e]
        if ndof == 1:
            Ke, _ = element_1d_bar(x1, x2, E, A)
        elif ndof == 2:
            Ke, _, _, _ = element_2d_truss(x1, y1, x2, y2, E, A)
        else:
            raise NotImplementedError("仅支持1D杆 / 2D桁架")
        n_local = Ke.shape[0]
        for a in range(n_local):
            for b in range(n_local):
                gi = LM[a, e]
                gj = LM[b, e]
                K[gi, gj] += Ke[a, b]
    return K

# ====================== 4. 缩减法求解（增加异常捕获） ======================
def solve_reduction(K, f_vec, fixed_dof_1, fixed_vals):
    ndof_tot = len(f_vec)
    fixed = np.array(fixed_dof_1, int) - 1
    free = np.array([i for i in range(ndof_tot) if i not in fixed])
    nf = len(free)

    Kff = K[np.ix_(free, free)]
    Kfe = K[np.ix_(free, fixed)]
    Kef = K[np.ix_(fixed, free)]
    Kee = K[np.ix_(fixed, fixed)]

    df = np.array(fixed_vals, float)
    ff = f_vec[free]
    rhs = ff - Kfe @ df

    # 捕获缩减矩阵奇异报错
    try:
        d_free = np.linalg.solve(Kff, rhs)
    except np.linalg.LinAlgError:
        raise RuntimeError("缩减刚度矩阵Kff奇异，边界约束不足，结构几何可变！")

    d_total = np.zeros(ndof_tot)
    d_total[free] = d_free
    d_total[fixed] = df

    # 支座反力
    R = Kef @ d_free + Kee @ df - f_vec[fixed]
    return d_total, R, free, fixed, Kff

# ====================== 5. 后处理 ======================
def post_process(model, LM, d_total):
    elem_results = []
    print("\n" + "="*60 + " 单元后处理结果 " + "="*60)
    x = np.array(model["x"])
    y = np.array(model["y"])
    IEN = np.array(model["IEN"], int) - 1
    E_list = model["E"]
    A_list = model["CArea"]
    nel = model["nel"]
    ndof = model["ndof"]

    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = x[n1], y[n1]
        x2, y2 = x[n2], y[n2]
        E = E_list[e]
        A = A_list[e]
        de_idx = LM[:, e]
        de = d_total[de_idx]
        res = {"elem_id": e+1}
        print(f"\n==== 单元 {e+1} ====")
        if ndof == 1:
            Ke, L = element_1d_bar(x1, x2, E, A)
            sigma, N = stress_1d(E, A, L, de)
            res.update({"L":L, "sigma":sigma, "N":N})
            print(f"单元长度 L = {L:.10f}")
            print(f"应力 σ = {sigma:.10f}")
            print(f"轴力 N = {N:.10f} (拉力为正，压力为负)")
        elif ndof == 2:
            Ke, L, c, s = element_2d_truss(x1, y1, x2, y2, E, A)
            sigma, N = stress_2d(E, A, L, c, s, de)
            res.update({"L":L, "c":c, "s":s, "sigma":sigma, "N":N})
            print(f"单元长度 L = {L:.10f}")
            print(f"方向余弦 c={c:.10f}, s={s:.10f}")
            print(f"应力 σ = {sigma:.10f}")
            print(f"轴力 N = {N:.10f} (拉力为正，压力为负)")
        elem_results.append(res)
    return elem_results

# ====================== 主求解流程（增加理论解对比输出） ======================
def run_fem(json_model_str):
    model = load_model(json_str=json_model_str)
    LM = build_LM(model)
    ndof_tot = get_total_dof(model)
    ndof = model["ndof"]
    nnp = model["nnp"]
    print("="*70 + " 对号矩阵 LM（局部自由度×单元）" + "="*70)
    print(LM)

    # 组装全局刚度
    K = assemble_global_K(model, LM)
    print("\n" + "="*70 + " 总体刚度矩阵 K (保留6位小数) " + "="*70)
    print(np.round(K, 6))

    # 对称校验
    sym_flag, diff = check_symmetric(K)
    print(f"\n【刚度矩阵对称校验】最大误差 = {diff:.2e}，对称：{sym_flag}")

    # 无约束时奇异性校验
    sing_K, detK = is_singular(K)
    print(f"\n【无边界约束整体K】行列式 det(K) = {detK:.2e}，矩阵奇异：{sing_K}")

    # 构建载荷向量
    f = np.zeros(ndof_tot)
    f_dof_1 = model["force_dof"]
    f_val = model["force_value"]
    f_idx = np.array(f_dof_1, int) - 1
    f[f_idx] = f_val

    # 缩减法求解
    fix_dof_1 = model["fixed_dof"]
    fix_val = model["fixed_value"]
    d_total, R, free_dofs, fixed_dofs, Kff = solve_reduction(K, f, fix_dof_1, fix_val)

    # 缩减刚度奇异性
    sing_Kff, detKff = is_singular(Kff)
    print(f"\n【约束后缩减刚度Kff】行列式 det(Kff) = {detKff:.2e}，奇异：{sing_Kff}")

    # 分方向平衡校验
    residual, balance_ok, f_sum, r_sum = check_force_balance(f, R, fixed_dofs, free_dofs, ndof)
    print(f"\n【整体力平衡校验】")
    print(f"外载荷分轴总和: {f_sum}")
    print(f"支座反力分轴总和: {r_sum}")
    print(f"合力残差 f+R = {residual}, 平衡满足: {balance_ok}")

    # 节点位移输出
    print("\n" + "="*70 + " 全部节点位移结果 " + "="*70)
    for node in range(nnp):
        if ndof == 1:
            u = d_total[node*1]
            print(f"节点{node+1}  u = {u:.10f}")
        elif ndof == 2:
            u = d_total[node*2]
            v = d_total[node*2 + 1]
            print(f"节点{node+1}  u={u:.10f}, v={v:.10f}")

    # 支座反力输出（增加节点、方向标识）
    print("\n" + "=" * 70 + " 约束自由度支座反力 " + "=" * 70)
    for idx, gdof in enumerate(fixed_dofs):
        node_id = gdof // ndof + 1
        axis = "u" if gdof % ndof == 0 else "v"
        print(f"全局自由度 {gdof + 1} (节点{node_id}, {axis}) 支座反力 R = {R[idx]:.10f}")

    # 单元应力轴力
    elem_res = post_process(model, LM, d_total)

    # 理论解对照输出（作业标准解）
    print("\n" + "="*70 + " 作业标准理论解对照 " + "="*70)
    if model["Title"] == "1D two bar elements":
        print("理论位移：d1=0.0, d2=0.1, d3=0.15")
        print("理论支座反力：节点1反力 = -10.0")
        print("理论总体刚度矩阵：\n[[100,-100,0],[-100,300,-200],[0,-200,200]]")
    elif model["Title"] == "2D two truss elements":
        print("理论节点3位移：u3≈38.284271247, v3≈-10.0")
        print("理论单元应力：单元1 σ≈-10.0，单元2 σ≈14.1421356237")
    return K, d_total, R, elem_res, Kff

# ====================== 算例JSON定义（严格匹配作业参数） ======================
# 算例1：一维两单元杆 严格满足 E1A1/L1=100，E2A2/L2=200
CASE1_1D = '''
{
"Title": "1D two bar elements",
"nsd":1,
"ndof":1,
"nnp":3,
"nel":2,
"nen":2,
"E": [100, 200],
"CArea": [1, 1],
"x": [0, 1, 2],
"y": [0, 0, 0],
"IEN": [[1,2],[2,3]],
"fixed_dof": [1],
"fixed_value": [0.0],
"force_dof": [3],
"force_value": [10.0]
}
'''

# 算例2：二维两杆桁架 完全匹配题目输入
CASE2_2D = '''
{
"Title": "2D two truss elements",
"nsd": 2,
"ndof": 2,
"nnp": 3,
"nel": 2,
"nen": 2,
"E": [1.0, 1.0],
"CArea": [1.0, 1.0],
"x": [1.0, 0.0, 1.0],
"y": [0.0, 0.0, 1.0],
"IEN": [[1, 3], [2, 3]],
"fixed_dof": [1, 2, 3, 4],
"fixed_value": [0.0, 0.0, 0.0, 0.0],
"force_dof": [5, 6],
"force_value": [10.0, 0.0]
}
'''

# ====================== 运行入口 ======================
if __name__ == "__main__":
    # 切换算例：注释/取消注释
    # model_json = CASE1_1D
    model_json = CASE2_2D

    K_global, disp, reaction, elem_data, Kff = run_fem(model_json)

    # 单元轴力汇总
    print("\n==== 全部单元轴力汇总 ====")
    for elem in elem_data:
        print(f"单元{elem['elem_id']} 轴力 N = {elem['N']:.10f}")