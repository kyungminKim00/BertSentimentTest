import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cupyx.profiler import benchmark

# # cupy test
# def my_func(a):
#     return cp.sqrt(cp.sum(a**2, axis=-1))
# a = cp.random.random((10240, 10240))
# print(benchmark(my_func, (a,), n_repeat=20))


# parameters test
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 5x5의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면, 하나의 숫자만을 특정(specify)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # 배치 차원을 제외한 모든 차원을 하나로 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1의 .weight
