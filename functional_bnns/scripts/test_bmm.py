"""
I have 3 lists of matrices in pytorch:
```
n,N = 10,100
d = 10
A = [ torch.randn(n,n) for _ in range(d) ]
B = [ torch.randn(N,n) for _ in range(d) ]
C = [ torch.randn(N,N) for _ in range(d) ]
```
how can I compute the list of matrix products `[ C[j] - B[j]@A[j]@B[j].T for j in range(d) ]` more efficiently than by list comprehension?
"""



### ~~~
## ~~~ Set up my problem
### ~~~

import torch
n,N = 10,100
d = 10
A = [ torch.randn(n,n) for _ in range(d) ]
B = [ torch.randn(N,n) for _ in range(d) ]
C = [ torch.randn(N,N) for _ in range(d) ]



### ~~~
## ~~~ Test chat GPT's proposed solution
### ~~~

from bnns.GPR import compute_C_minus_BABt
result = compute_C_minus_BABt(
        A = torch.stack(A), # Shape: (d, n, n)
        B = torch.stack(B), # Shape: (d, N, n)
        C = torch.stack(C)  # Shape: (d, N, N)
    )

error = result - torch.stack([ C[j] - B[j]@A[j]@B[j].T for j in range(d) ])
assert (error**2).mean() < 1e-9