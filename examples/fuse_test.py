import cupy as cp
import numpy as np
import nvtx

def do_some_stuff(x, y):
    sumx = cp.sum(x, axis=1)
    return sumx**2 + y / cp.sqrt(y)

@cp.fuse()
def fused_some_stuff(x, y):
    sumx = cp.sum(x, axis=1)
    return sumx**2 + y / cp.sqrt(y)




X = cp.ones((800, 150), dtype=np.float)
Y = cp.full(800, 4.0)


with nvtx.annotate(message="not_fused"):
    _ = do_some_stuff(X,Y).get()

with nvtx.annotate(message="fused"):
    _ = fused_some_stuff(X,Y).get()


with nvtx.annotate(message="not_fused"):
    _ = do_some_stuff(X,Y).get()

with nvtx.annotate(message="fused"):
    _ = fused_some_stuff(X,Y).get()


