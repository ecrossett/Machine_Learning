import numpy as np

def unroll(image):
    v = image.reshape((image.shape[0],image.shape[1],image.shape[2]),1)
    return v


def normalizeRows(x):
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x / x_norm
    return x

def reshape():
    # create array of zeros
    img = np.zeros((32,32,3))

    print("img:",np.shape(img))

    # reshape into column vector
    x = img.reshape((32*32*3,1))

    print("x:",np.shape(x))

    a = np.random.randn(4,3)
    b = np.random.randn(3,2)
    c = a * b    

    return x, a, b, c

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_gradient(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

    