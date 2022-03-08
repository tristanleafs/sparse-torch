import torch
from custom_classes import Splatter
import time
import numpy as np
import matplotlib.pyplot as plt


model_normal = torch.nn.Sequential(
    torch.nn.Conv2d(1,1,3)
)

model_splatter = Splatter(3,3)
normal_time = []
splatter_time = []
sparsity = np.linspace(0,1,10)
for i in sparsity:

    input = (torch.rand(size=(100,1,24,24)) < i).float()

    t0 = time.perf_counter()
    output1 = model_normal(input)
    normal_time.append(time.perf_counter()-t0)

    t0 = time.perf_counter()
    output2 = model_normal(input)
    splatter_time.append(time.perf_counter()-t0)

normal_time = np.array(normal_time)
splatter_time = np.array(splatter_time)

print("Normal Time:",normal_time)
print("Splatter Time:", splatter_time)
plt.subplot()
plt.plot(normal_time, sparsity)