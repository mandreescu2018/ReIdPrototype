# initialize torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import time

# print(torch.__version__)
# print(torchvision.__version__)

print("current_device: ", torch.cuda.current_device())

matrix_size = 32*512

x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)


print("************** CPU Speed Test **************")
start_time = time.time()
res = torch.matmul(x, y)
print("Time: ", time.time() - start_time)

for _ in range(3):
    print("************** GPU Speed Test **************")
    x_gpu = x.to('cuda')
    y_gpu = y.to('cuda')
    torch.cuda.synchronize()

    start_time = time.time()
    res_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()

    print("Time: ", time.time() - start_time)
    print("res_gpu: ", res_gpu.device)

