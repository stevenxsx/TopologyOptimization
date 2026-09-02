"""Vector addition is akin to the Hello World of GPU programming. This script shows how to utilize CuTile for vector addition as shown by NVIDIA in https://www.youtube.com/watch?v=cNDbqFaoQ9k
"""

from math import ceil

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel #This indicates that the next defined function is a CUDA kernel
#Arrays a, b, and c must already be located on the GPU
#A tile size is defined in order to launch a grid of tile-blocks, as opposed to in conventional
#CUDA where we launch a grid of thread-blocks   
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    #Get the 1-dimensional process ID
    pid = ct.bid(0)

    #Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size))

    #Perform elementwise addition
    result = a_tile + b_tile

    #Store result
    ct.store(c, index=(pid, ), tile=result)

def test():
