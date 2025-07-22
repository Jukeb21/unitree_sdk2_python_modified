import numpy as np
from scipy.optimize import root

A0 = 0.07106
A1 = 0.275437
A2 = 0.232857
A3 = 0.054
G = 0.2618
Z = 0.089680227
Y = 0.336857
X = 0.23469034612943444



def equations(vars):
    x0, x1, x2 = vars
    eq1 = (-A1 * np.cos(x0) + A2 * np.sin(x0 - x2)) * np.cos(G + x1) + A0 * np.sin(G) - Z
    eq2 = (A1 * np.sin(x0) + A2 * np.cos(x0 - x2) + A3) - Y
    eq3 = (A1 * np.cos(x0) - A2 * np.sin(x0 - x2)) * np.sin(G + x1) + A0 * np.cos(G) - X

    return [eq1, eq2, eq3]

# 初始猜测值
initial_guess = [0.5, 0.5, 0.5]
solution = root(equations, initial_guess)

if solution.success:
    x0_num, x1_num, x2_num = solution.x
    print(x0_num)
    x0_num=np.arctan(np.tan(x0_num)*np.cos(x1_num))
    print(x0_num)
    j3_num = x0_num - x2_num
    if -3.14 <= x0_num <= 1.57 and -0.38 <= x1_num <= 3.4 and -0.471 <= x2_num <= 0.349 and -1.012 <= j3_num <= 1.012:
        print(f"解为: x0 = {x0_num}, x1 = {x1_num}, x2 = {x2_num}, x3 = {j3_num}")
    else:
        print("超过关节限制")
else:
    print("方程组无解")