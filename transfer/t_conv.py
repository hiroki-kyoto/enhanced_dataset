# test for convolution
from conv import *
import time

if 'DEF_CONV' not in globals():
    from transfer.conv import *

def test_matrix():
    x = np.array([[0.09, 0.0, 0.5], [0.2, 0.3, 0.08]])
    m1 = Matrix(x)
    print(m1)
    x = np.array([[-0.09, 0.3, 0.07], [0.03, -0.3, 0.1]])
    m2 = Matrix(x)
    print(m2)
    m_add = m1 + m2
    print(m_add)

    # check speed

    a = np.random.uniform(low=-1.0, high=1.0, size=[1000, 1000])
    b = np.random.uniform(low=-1.0, high=1.0, size=[1000, 1000])
    cond = abs(a) > 0.999
    a[cond] = 0
    cond = abs(b) > 0.999
    b[cond] = 0
    m1 = Matrix(a)
    m2 = Matrix(b)
    # numpy function:
    start_t = time.clock()
    ######## COUNT IN ########
    m_add = m1 + m2
    end_t = time.clock()
    print("%s seconds." % (end_t - start_t))
    ######## COUNT OUT #######

    #print(m_add)

def main():
    test_matrix()

main()