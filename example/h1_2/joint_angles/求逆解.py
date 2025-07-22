import numpy as np
import sympy as sp
import sys
sys.stdout.reconfigure(encoding='utf-8')   

A1=1
A2=1
A3=1
Z=(1-np.sqrt(3))/2
Y=(3+np.sqrt(3))/2

if (abs(A1 ** 2 + A2 ** 2 - Z ** 2 - (Y - A3) ** 2) < 2 * abs(A1 * A2)):
    print("开始求解")
    x1, x2 = sp.symbols('x1 x2')
    eq1 = -A1 * sp.cos(x1) + A2 * sp.sin(x2) - Z
    eq2 = A1 * sp.sin(x1) + A2 * sp.cos(x2) - (Y - A3)
    solution = sp.solve([eq1, eq2], (x1, x2))

    if solution:
        condition=1
        for i in range(len(solution)):
            j1,j2=solution[i]
            j1_num = float(j1.evalf())
            j2_num = float(j2.evalf())
            if -3.14 <= j1_num <= 1.57: # and -0.471 <= j2_num <= 0.349:
                print(f"解为: x1 = {j1_num}, x2 = {j2_num}")
                condition=0
            else:
                pass
        if  condition:
            print("超过关节限制")
        else:
            pass
else:
    print("超过范围，方程组无解")