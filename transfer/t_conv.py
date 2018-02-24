# test for convolution
from conv import *

if 'DEF_CONV' not in globals():
    from transfer.conv import *

def test_matrix():
    x = np.array([[0.09, 0.0, 0.5], [0.2, 0.3, 0.08]])
    m1 = Matrix(x)
    print(m1)
    x = np.array([[0.1, 0.3, 0.07], [0.03, 0.6, 0.1]])
    m2 = Matrix(x)
    print(m2)
    m_add = m1 + m2
    print(m_add)

def main():
    test_matrix()

main()