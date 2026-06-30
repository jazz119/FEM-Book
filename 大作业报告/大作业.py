import numpy as np
import matplotlib.pyplot as plt

# ====================== 规定核心函数1：SUPG最优alpha计算 ======================
def alpha_supg(Pe):
    """
    计算SUPG稳定参数 alpha_opt = coth(Pe) - 1/Pe
    Pe趋近0时防止除零异常
    """
    if abs(Pe) < 1e-8:
        return 0.0
    coth_pe = 1 / np.tanh(Pe)
    return coth_pe - 1.0 / Pe

# ====================== 规定核心函数2：单元刚度矩阵 ======================
def element_matrix(kappa, v, le, alpha):
    """
    生成两节点线性单元对流扩散单元矩阵 Ke (2×2)
    kappa: 原始扩散系数
    v: 对流速度
    le: 单元长度
    alpha: 人工扩散控制参数
    """
    # 等效扩散系数（人工扩散修正）
    kappa_bar = kappa + alpha * v * le / 2.0
    # 扩散项对称矩阵
    K_diff = (kappa_bar / le) * np.array([[1, -1],
                                          [-1, 1]])
    # 对流项非对称矩阵
    K_conv = (v / 2.0) * np.array([[-1, 1],
                                   [-1, 1]])
    Ke = K_diff + K_conv
    return Ke

# ====================== 规定核心函数3：整体求解器 ======================
def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    有限元组装、边界条件施加、方程组求解
    返回：节点坐标x、数值解theta、精确解theta_exact、总刚度矩阵K
    """
    le = L / nel
    nnodes = nel + 1
    # 1. 生成均匀网格节点
    x = np.linspace(0, L, nnodes)
    # 2. 初始化总刚与右端向量
    K = np.zeros((nnodes, nnodes))
    F = np.zeros(nnodes)
    # 3. 遍历所有单元，组装总刚度矩阵
    for elem_idx in range(nel):
        Ke = element_matrix(kappa, v, le, alpha)
        # 当前单元两个全局节点编号
        node1 = elem_idx
        node2 = elem_idx + 1
        K[node1:node2+1, node1:node2+1] += Ke
    # 4. 施加Dirichlet边界条件 theta(0)=0, theta(L)=1
    # 左边界 x=0
    K[0, :] = 0.0
    K[0, 0] = 1.0
    F[0] = 0.0
    # 右边界 x=L
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    F[-1] = 1.0
    # 5. 求解线性方程组 K·theta = F
    theta = np.linalg.solve(K, F)
    # 6. 稳定计算精确解，使用expm1防止大数指数溢出
    global_Pe = v * L / kappa
    theta_exact = np.expm1(v * x / kappa) / np.expm1(global_Pe)
    return x, theta, theta_exact, K

# ====================== 主程序入口，完成全部4项任务 + 附加题 ======================
if __name__ == "__main__":
    # 全局固定参数（作业要求）
    L = 1.0
    nel_base = 20
    v = 1.0
    Pe_cases = [0.1, 3.0]  # 两组对比工况
    error_record = []

    print("========== 一维对流扩散有限元求解程序 ==========\n")
    for Pe_target in Pe_cases:
        print(f"===== 单元Pe = {Pe_target} 计算开始 =====")
        le = L / nel_base
        # 由Pe反推扩散系数 kappa = v*le/(2Pe)
        kappa = v * le / (2 * Pe_target)
        print(f"单元长度 le = {le:.4f}, 扩散系数 κ = {kappa:.6e}")

        # 1. 标准Galerkin α=0
        x_gal, theta_gal, theta_ex, K_galerkin = solve_advection_diffusion(nel_base, L, v, kappa, alpha=0.0)
        max_err_gal = np.max(np.abs(theta_gal - theta_ex))

        # 2. 迎风格式 α=1
        x_upwind, theta_upwind, _, _ = solve_advection_diffusion(nel_base, L, v, kappa, alpha=1.0)
        max_err_upwind = np.max(np.abs(theta_upwind - theta_ex))

        # 3. SUPG/Petrov-Galerkin α=α_opt
        alpha_opt = alpha_supg(Pe_target)
        x_supg, theta_supg, _, _ = solve_advection_diffusion(nel_base, L, v, kappa, alpha=alpha_opt)
        max_err_supg = np.max(np.abs(theta_supg - theta_ex))

        # 记录误差
        error_record.append([Pe_target, max_err_gal, max_err_upwind, max_err_supg])
        print(f"标准Galerkin最大节点误差: {max_err_gal:.6e}")
        print(f"迎风格式最大节点误差:     {max_err_upwind:.6e}")
        print(f"SUPG稳定格式最大误差:      {max_err_supg:.6e}\n")

        # 任务4：Pe=3时输出总刚并分析矩阵对称性、正定性
        if abs(Pe_target - 3.0) < 1e-6:
            print("==== Pe=3.0 标准Galerkin总刚度矩阵 ====")
            print(K_galerkin)
            # 判断对称
            sym_flag = np.allclose(K_galerkin, K_galerkin.T)
            print(f"矩阵是否对称：{sym_flag}")
            # 判断正定（特征值全部大于0）
            eig_vals = np.linalg.eigvalsh(K_galerkin)
            pos_def_flag = np.all(eig_vals > -1e-10)
            print(f"矩阵是否正定：{pos_def_flag}")
            print(f"矩阵最小特征值：{np.min(eig_vals):.4e}\n")

        # 任务3：绘图，一张图包含四条曲线
        plt.figure(figsize=(10, 6))
        plt.plot(x_gal, theta_ex, "k-", lw=2, label="Exact Analytical Solution")
        plt.plot(x_gal, theta_gal, "r--", lw=1.5, label=f"Standard Galerkin α=0")
        plt.plot(x_upwind, theta_upwind, "g-.", lw=1.5, label=f"Upwind Scheme α=1")
        plt.plot(x_supg, theta_supg, "b:", lw=2, label=f"SUPG α_opt={alpha_opt:.4f}")
        plt.xlabel("Spatial coordinate x")
        plt.ylabel(r"Field variable $\theta(x)$")
        plt.title(f"Advection-Diffusion, Element Pe = {Pe_target}, nel={nel_base}")
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig(f"Pe_{Pe_target}.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 输出误差汇总表格
    print("==================== 全局误差汇总表 ====================")
    print(f"{'Pe':<6}{'Galerkin MaxErr':<18}{'Upwind MaxErr':<18}{'SUPG MaxErr':<18}")
    for line in error_record:
        pe, eg, eu, es = line
        print(f"{pe:<6.1f}{eg:<18.6e}{eu:<18.6e}{es:<18.6e}")

    # ====================== 附加题：网格加密收敛测试 ======================
    print("\n========== 附加题：网格加密收敛测试 Pe=3.0 ==========")
    Pe_add = 3.0
    nel_list = [10, 20, 40, 80]
    err_gal_conv = []
    err_supg_conv = []
    h_list = []
    for n in nel_list:
        h = L / n
        h_list.append(h)
        kappa_t = v * h / (2 * Pe_add)
        # Galerkin
        _, th_g, th_ex_t, _ = solve_advection_diffusion(n, L, v, kappa_t, 0.0)
        eg = np.max(np.abs(th_g - th_ex_t))
        # SUPG
        a_opt_t = alpha_supg(Pe_add)
        _, th_s, _, _ = solve_advection_diffusion(n, L, v, kappa_t, a_opt_t)
        es = np.max(np.abs(th_s - th_ex_t))
        err_gal_conv.append(eg)
        err_supg_conv.append(es)
        print(f"网格单元数 nel={n:3d}, 单元长度 h={h:.4f}, Galerkin误差={eg:.6e}, SUPG误差={es:.6e}")

    # 绘制双对数收敛曲线
    plt.figure(figsize=(9, 5))
    plt.loglog(h_list, err_gal_conv, "ro-", label="Standard Galerkin")
    plt.loglog(h_list, err_supg_conv, "bs-", label="SUPG Stabilized Method")
    plt.xlabel("Element size $l_e$ (log scale)")
    plt.ylabel("Maximum absolute error (log scale)")
    plt.title("Error Convergence Curve with Mesh Refinement (Pe=3.0)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("convergence_curve.png", dpi=300, bbox_inches="tight")
    plt.show()