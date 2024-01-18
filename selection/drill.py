

import torch

a = torch.rand((3,4))
b = torch.isfinite(a).all()
print(b)
c = a.nonzero()
print(c)




