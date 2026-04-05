import numpy as np

def multi_class(W1, W2, X, D, alpha):
    """ one epoch of training """
    for i in range(len(X)):
        x = X[i, :, :].reshape(1, -1).T ## x is of shape (25, 1)
        d = D[i, :].reshape(5, 1) # d: (5, 1) one-hot vector

        v1 = W1 @ x # W1: (nodes, 25)
        y1 = sigmoid(v1) # y1: (nodes, 1)

        v = W2 @ y1 # v: (5, 1)
        y = softmax(v) # y: (5, 1)
        e = d - y # e: (5, 1)
        delta = e # delta: (5, 1)

        e1 = W2.T @ delta
        delta1 = sigmoid_derivative(y1) * e1

        W1 += alpha * delta1 @ x.T
        W2 += alpha * delta @ y1.T

def results(W1, W2, X, D=None, title="Predictions"):
    """Print the prediction results for the input data"""
    print(f"{title}: ")
    
    for i in range(len(X)):
        # 提取并重塑输入样本
        x = X[i, :, :].reshape(1, -1).T  # 转换为 (25, 1) 形状
        
        # 前向传播计算
        v1 = W1 @ x
        y1 = sigmoid(v1)
        v = W2 @ y1
        y = softmax(v)
        
        # 预测类别（取最大值索引+1，因为标签从1开始）
        prediction = np.argmax(y) + 1
        
        # 打印结果
        input_str = "\n".join([" ".join(map(str, row.astype(int))) for row in X[i, :, :]])
        output_str = f"{prediction}"
        desired_str = f" (desired: {D[i]})" if D is not None else ""
        
        print(f"Sample {i+1}:")
        print(f"Input:\n{input_str}")
        print(f"Prediction: {output_str}{desired_str}")
        print(f"Raw softmax output:")
        print([f"{val:.4f}" for val in y.flatten()])
        print("-" * 30)
    print()

def sigmoid(x):
    """activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(r_sigmoid):
    """
    derivative of sigmoid function, but, pass me the result of sigmoid!
    """
    return r_sigmoid * (1 - r_sigmoid)

def softmax(X):
    """softmax function with numerical stability"""
    max_val = np.max(X)
    exp_X = np.exp(X - max_val)
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)

def softmax_derivative(r_softmax):
    """derivative of softmax function"""
    return r_softmax * (1 - r_softmax)