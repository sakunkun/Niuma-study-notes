import torch
from torch.utils.cpp_extension import load

lib = load(name="matrixMul", sources=["matrixMul.cu"])

a = torch.randn(2048, 1024, dtype=torch.float32, device="cuda")
b = torch.randn(1024, 512, dtype=torch.float32, device="cuda")
c = torch.empty(2048, 512, dtype=torch.float32, device="cuda")

def judge(a, b, c):
    # 判断c是否等于a*b
    if torch.allclose(c, torch.matmul(a, b), atol=1e-3):
        print("c 等于 a*b")
    else:
        print("c 不等于 a*b")

# lib.matrixMul_shared(a, b, c)
# # print("c = ", c)
# judge(a, b, c)

# lib.matrixMul_simple(a, b, c)
# judge(a, b, c)
# # print("c = ", c)

lib.matrixMul_shared_2(a, b, c)
judge(a, b, c)
# print("c = ", c)

lib.matrixMul_shared_3(a, b, c)
judge(a, b, c)
# print("c = ", c)