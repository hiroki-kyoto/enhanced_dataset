# compatibility work for python 2 and 3
def IS_PYTHON_3():
    s = 'abc'
    t = str(type(s))
    return t[1:6]=='class'

def IS_PYTHON_2():
    s = 'abc'
    t = str(type(s))
    return t[1:5] == 'type'