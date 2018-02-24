# a simple implementation of convoltion operator
# A define header is required for project
DEF_CONV = 1

# SPARSITY THRESHOLD
SPARSITY_THRESHOLD = 1e-1

import numpy as np

# matrix
# Big different is:
#   indices is a list of nonzero element indices in matrix
#   values is a still a full record of matrix elements of
#   both zero and nonzero. A 2-dimension array
#   Theory: Using extra sparse information to save computation time
class Matrix(object):
    def __init__(self, values=None):
        if values is None:
            self.dims = 0
            self.values = None
            self.indices = set()
            return
        assert(type(values) == np.ndarray)
        self.values = values
        # it has to be numpy array
        assert(len(values.shape) == 2)
        self.dims = self.values.shape
        # create sparsity info
        self.update_indices()

    def update_indices(self):
        self.indices = set()
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if abs(self.values[i, j]) >= SPARSITY_THRESHOLD:
                    self.indices.add((i, j))
                else:
                    self.values[i, j] = 0.0

    def __str__(self):
        _str_ = ''
        for (i, j) in self.indices:
            _str_ += "(%d,%d): %s\n" % (i, j, self.values[i,j])
        return _str_

    def __add__(self, other):
        assert(type(other) == Matrix)
        assert(self.dims == other.dims)
        # create a new matrix to store result
        m = Matrix()
        m.values = np.zeros(self.dims)
        m.dims = self.dims
        m.indices = self.indices.copy()
        for i,j in self.indices:
            m.values[i,j] += self.values[i,j]
        for i,j in other.indices:
            m.values[i,j] += other.values[i,j]
            if abs(m.values[i,j]) >= SPARSITY_THRESHOLD:
                m.indices.add((i,j))
            else:
                m.values[i,j] = 0.0
                m.indices.remove((i,j))
        return m








# extend zeros around map
# x: input map
# r_x: radius to extend on axis x
# r_y: radius to extend on axis y
# e: element to fill
def extend(x, r_x, r_y, e):
    return x

# x : must be 2D tensor [H, W]
# w : must be 2D tensor [H, W]
# Notice: Convolution operator is of only stride: (1,1).
#         And the padding method is only 'SAME'.
#         And the size of kernel must be 1,3,5(odd number)

def conv2d(x, w):
    dim_x = x.shape
    dim_w = w.shape
    assert(len(dim_x)==2)
    assert(len(dim_w)==2)
    assert(dim_w[0]%2 and dim_w[1]%2)
    # extend the input map with zeros
    r_x = (dim_w[1]-1)/2
    r_y = (dim_w[0]-1)/2
