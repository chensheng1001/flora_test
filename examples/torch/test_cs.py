from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 初始化 accelerator
accelerator = Accelerator(gradient_accumulation_steps=4)

# 简单模型
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

# 模拟一些数据
X = torch.randn(20, 10)
y = torch.randn(20, 1)
dataloader = DataLoader(TensorDataset(X, y), batch_size=2)

# 注册 post-accumulate-grad hook
def post_hook(param):
    if accelerator.sync_gradients:  # 到达 accumulation stepif
        print("test")

# hook 只需要注册一次
for p in model.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(post_hook)

# 准备 accelerator 的包装
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 训练一个 epoch
model.train()
for step, (inputs, targets) in enumerate(dataloader):
    with accelerator.accumulate(model):  # 自动管理梯度累积
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)

        if accelerator.sync_gradients:  # 到达 accumulation step
            optimizer.step()
            optimizer.zero_grad()
            print(f"[Step {step}] optimizer step executed\n")
