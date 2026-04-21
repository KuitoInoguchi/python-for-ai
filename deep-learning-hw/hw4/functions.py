import numpy as np

def deep_relu(X, Y, W1, W2, W3, W4, alpha):
    """ one epoch of training """
    
    for i in range(len(X)):
        x = X[i, :, :].reshape(25, 1)  # x is of shape (25, 1)
        y_true = Y[i, :].reshape(5, 1)  # y_true: (5, 1) one-hot vector

        # forward propagation
        v1 = W1 @ x
        y1 = relu(v1)
        v2 = W2 @ y1
        y2 = relu(v2)
        v3 = W3 @ y2
        y3 = relu(v3)
        v = W4 @ y3
        y_pred = softmax(v)

        # backward propagation
        error = y_true - y_pred
        delta4 = error  # output layer error

        # hidden layer 3 error
        e3 = W4.T @ delta4
        delta3 = (v3 > 0) * e3  # ReLU derivative

        # hidden layer 2 error
        e2 = W3.T @ delta3
        delta2 = (v2 > 0) * e2  # ReLU derivative

        # hidden layer 1 error
        e1 = W2.T @ delta2
        delta1 = (v1 > 0) * e1  # ReLU derivative

        # weight update
        W1 += alpha * delta1 @ x.T
        W2 += alpha * delta2 @ y1.T
        W3 += alpha * delta3 @ y2.T
        W4 += alpha * delta4 @ y3.T


def results(W1, W2, W3, W4, X, D=None, title="Predictions"): 
     """Print the prediction results for the input data""" 
     print(f"{title}: ") 
     
     for i in range(len(X)): 
         # 提取并重塑输入样本 
         x = X[i, :, :].reshape(1, -1).T  # 转换为 (25, 1) 形状 
         
         # 前向传播计算（三隐层） 
         v1 = W1 @ x 
         y1 = np.maximum(0, v1)  # relu激活函数 
         v2 = W2 @ y1 
         y2 = np.maximum(0, v2)  # relu激活函数 
         v3 = W3 @ y2 
         y3 = np.maximum(0, v3)  # relu激活函数 
         v = W4 @ y3 
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

def softmax(X):
    """softmax function with numerical stability"""
    max_val = np.max(X)
    exp_X = np.exp(X - max_val)
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)

def relu(x):
    """relu function"""
    return np.maximum(0, x)