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

@cp.fuse()
def split_fuse_1(x):
    return cp.sum(x, axis=1)

@cp.fuse()
def split_fuse_2(sumx, y):
    return sumx**2 + y / cp.sqrt(y)

@cp.fuse()
def fused_with_reduction_at_end(x, y):
    sumx = cp.sum(x, axis=1)
    return sum(x**2 + y / cp.sqrt(y)




X = cp.ones((800, 150), dtype=np.float)
Y = cp.full(800, 4.0)


with nvtx.annotate(message="not_fused_asnumpy"):
    _ = cp.asnumpy(do_some_stuff(X, Y))

with nvtx.annotate(message="fused_asnumpy"):
    _ = cp.asnumpy(fused_some_stuff(X, Y))


with nvtx.annotate(message="not_fused_asnumpy"):
    _ = cp.asnumpy(do_some_stuff(X, Y))

with nvtx.annotate(message="fused_asnumpy"):
    _ = cp.asnumpy(fused_some_stuff(X, Y))


with nvtx.annotate(message="not_fused_get"):
    _ = do_some_stuff(X, Y).get()

with nvtx.annotate(message="fused_get"):
    _ = fused_some_stuff(X, Y).get()

with nvtx.annotate(message="split_fuse_get"):
    sumx = split_fuse_1(X)
    _ = split_fuse_2(sumx, Y).get()


with nvtx.annotate(message="not_fused_get"):
    _ = do_some_stuff(X, Y).get()

with nvtx.annotate(message="fused_get"):
    _ = fused_some_stuff(X, Y).get()

with nvtx.annotate(message="split_fuse_get"):
    sumx = split_fuse_1(X)
    _ = split_fuse_2(sumx, Y).get()


