def softmax(x):
    '''Calculates the softmax for each row of the input x'''
    x_exp = np.exp(x)
    # sum each row of x_exp
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    # compute
    s = x_exp / x_sum
    
    return s