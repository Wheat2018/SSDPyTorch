import torch
y = torch.tensor([0.3, 0.7])
p = torch.tensor([0.7, 0.3]).requires_grad_()

l = torch.log(p) * y
l_sum = l.sum()
y[0] = 5
l_sum.backward(torch.ones_like(l_sum))
print(l_sum)
print(p.grad)


