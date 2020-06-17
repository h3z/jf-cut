from numba import cuda


@cuda.jit
def f():
    print(1)
    cuda.syncthreads()
    print(2)


f[1, 1000]()
