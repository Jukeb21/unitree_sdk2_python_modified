import numpy as np

# 常数定义
A0 = 0.07106
A1 = 0.275437
A2 = 0.232857
A3 = 0.054
G = 0.2618


# 测试部分：正向运算
print("="*50)
print("测试部分：正向运算")
print("="*50)

# 输入四个角度参数作为测试值
x0_test = 0.5  # 测试角度1
x1_test = 0.3  # 测试角度2  
x2_test = 0.2  # 测试角度3
x3_test = 0.1  # 测试角度4

print(f"输入测试角度: x0 = {x0_test}, x1 = {x1_test}, x2 = {x2_test}, x3 = {x3_test}")

# 根据equations函数中的公式进行正向运算，计算XYZ值
X_test = (A1 * np.cos(x0_test) - A2 * np.sin(x0_test - x2_test)) * np.sin(G + x1_test) + A0 * np.cos(G)
Y_test = (A1 * np.sin(x0_test) + A2 * np.cos(x0_test - x2_test) + A3)
Z_test = (-A1 * np.cos(x0_test) + A2 * np.sin(x0_test - x2_test)) * np.cos(G + x1_test) + A0 * np.sin(G)

print(f"正向运算结果: X = {X_test}, Y = {Y_test}, Z = {Z_test}")