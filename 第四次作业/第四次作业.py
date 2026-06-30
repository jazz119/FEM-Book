import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元长度、方向余弦、全局6×6刚度矩阵
    :param x1: 节点1坐标 [x1,y1,z1] list/np.array
    :param x2: 节点2坐标 [x2,y2,z2] list/np.array
    :param E: 弹性模量 Pa
    :param A: 横截面积 m^2
    :return: L, (cx,cy,cz), Ke (6×6 np.array)
    """
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 退化单元：两点重合报错
    if np.isclose(L, 0.0):
        raise ValueError("错误：两个节点坐标重合，无效杆单元！")

    cx = dx / L
    cy = dy / L
    cz = dz / L
    dir_cos = np.array([cx, cy, cz])

    # 坐标变换矩阵 [ -cx, -cy, -cz, cx, cy, cz ]
    T = np.array([[-cx, -cy, -cz, cx, cy, cz]])
    k_local = E * A / L
    Ke = k_local * T.T @ T  # 全局刚度矩阵 6×6
    return L, dir_cos, Ke


def truss3d_element_stress(L, dir_cos, E, A, de):
    """
    复用已算出的L、方向余弦，由节点位移de计算轴向应变、应力、轴力
    :param L: 单元长度
    :param dir_cos: [cx,cy,cz] 方向余弦数组
    :param de: 位移向量 [u1,v1,w1,u2,v2,w2]
    :return: epsilon, sigma, N
    """
    cx, cy, cz = dir_cos
    de = np.array(de, dtype=np.float64)
    # 应变位移矩阵 B = 1/L * [-cx,-cy,-cz,cx,cy,cz]
    B = 1 / L * np.array([[-cx, -cy, -cz, cx, cy, cz]])
    epsilon = (B @ de).item()
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N


def check_matrix_property(Ke):
    """检验刚度矩阵性质：对称、特征值、奇异性"""
    print("===== 刚度矩阵性质检验 =====")
    # 1. 对称性
    is_sym = np.allclose(Ke, Ke.T)
    print(f"1. 矩阵是否对称：{is_sym}")
    # 2. 特征值（半正定：全部>=0，允许微小负浮点误差）
    eig_vals = np.linalg.eigvalsh(Ke)
    print(f"2. 全部特征值：\n{eig_vals.round(6)}")
    # 放大容差到1e-9，完全覆盖浮点误差
    tol = 1e-9
    all_pos_semi = np.all(eig_vals >= -tol)
    print(f"3. 是否半正定（所有特征值≥0）：{all_pos_semi}")
    # 3. 秩/奇异判断（单根杆秩=1，行列式≈0）
    det = np.linalg.det(Ke)
    print(f"4. 刚度矩阵行列式（接近0为奇异）：{det:.2e}\n")
    return eig_vals


def test_case1():
    """算例1：沿X轴一维杆"""
    print("==================== 算例1：X轴一维杆 ====================")
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9
    A = 1.0e-4
    de = [0, 0, 0, 1.0e-3, 0, 0]

    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(L, dir_cos, E, A, de)

    print(f"杆长 L = {L:.4f} m")
    print(f"方向余弦 cx,cy,cz = {dir_cos.round(4)}")
    print("全局刚度矩阵 Ke (原始数值)：")
    print(np.round(Ke, 2))
    print(f"轴向应变 ε = {eps:.6e}")
    print(f"轴向应力 σ = {sig / 1e6:.2f} MPa")
    print(f"轴力 N = {N:.2e} N\n")
    check_matrix_property(Ke)


def test_case2():
    """算例2：空间斜杆 (0,0,0)→(1,2,2)"""
    print("==================== 算例2：空间任意斜杆 ====================")
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9
    A = 2.0e-4
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(L, dir_cos, E, A, de)

    print(f"杆长 L = {L:.4f} m")
    print(f"方向余弦 cx,cy,cz = {dir_cos.round(4)}")
    print("全局刚度矩阵 Ke (单位：MN，除以1e6便于观察)：")
    print(np.round(Ke / 1e6, 4))
    print(f"轴向应变 ε = {eps:.6e}")
    print(f"轴向应力 σ = {sig / 1e6:.2f} MPa")
    print(f"轴力 N = {N:.2e} N\n")
    eig = check_matrix_property(Ke)

    # 刚体平移验证：所有节点统一位移，应变=0，无内力
    print("===== 刚体平移验证（两端同位移 [0.001,0.001,0.001]） =====")
    de_rigid = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    eps_r, sig_r, N_r = truss3d_element_stress(L, dir_cos, E, A, de_rigid)
    # 极小浮点值截断到12位小数
    eps_r = round(eps_r, 12)
    N_r = round(N_r, 12)
    print(f"刚体平移应变 ε = {eps_r:.2e}")
    print(f"刚体平移轴力 N = {N_r:.2e}\n")


def stiffness_column_physical_test():
    """任务4：刚度矩阵物理意义验证"""
    print("==================== 刚度矩阵物理意义验证 ====================")
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9
    A = 2e-4
    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)

    # 取第4个自由度u2（索引3，0开始），令其位移=1，其余0
    j = 3
    de_unit = np.zeros(6)
    de_unit[j] = 1.0
    Fe = Ke @ de_unit
    print(f"令自由度{j + 1}单位位移，其余为0，等效节点力Fe（单位：MN）：")
    print(np.round(Fe / 1e6, 4))
    print(f"刚度矩阵第{j + 1}列（单位：MN）：")
    print(np.round(Ke[:, j] / 1e6, 4))
    print("结论：Fe 等于刚度矩阵第j列；k_ij代表仅j自由度单位位移时i方向所需外力\n")


if __name__ == "__main__":
    # 运行全部测试
    test_case1()
    test_case2()
    stiffness_column_physical_test()

    # 测试退化单元报错（取消注释查看）
    # truss3d_element_stiffness([0,0,0],[0,0,0],200e9,1e-4)