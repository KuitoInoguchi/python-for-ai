import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(s):
    return s * (1 - s)

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


# Dropout掩码生成函数（1:1对应Matlab的Dropout函数）
# This function is AI-generated.
def dropout_mask(y, drop_ratio):
    m, n = y.shape
    mask = np.zeros_like(y, dtype=np.float64)
    total_elem = m * n
    keep_num = round(total_elem * (1 - drop_ratio))  # 保留的节点数
    # 随机无重复选索引，对应Matlab的randperm(m*n, num)
    selected_idx = np.random.choice(total_elem, keep_num, replace=False)
    # 反向缩放（Inverted Dropout），保证训练/测试数值尺度一致
    mask.flat[selected_idx] = total_elem / keep_num
    return mask


def deep_dropout(X, Y, W1, W2, W3, W4, alpha):

    for i in range(len(X)):
        x = X[i, :, :].reshape(25, 1)
        d = Y[i, :].reshape(-1, 1)
        drop_ratio = 0.2

        v1 = W1 @ x
        y1 = relu(v1)
        mask1 = dropout_mask(y1, drop_ratio)
        y1_drop = y1 * mask1

        v2 = W2 @ y1_drop
        y2 = relu(v2)
        mask2 = dropout_mask(y2, drop_ratio)
        y2_drop = y2 * mask2

        v3 = W3 @ y2_drop
        y3 = relu(v3)
        mask3 = dropout_mask(y3, drop_ratio)
        y3_drop = y3 * mask3

        v4 = W4 @ y3_drop
        y4 = softmax(v4)

        e = d - y4
        delta4 = e
        delta3 = (W4.T @ delta4) * (y3 > 0) * mask3
        delta2 = (W3.T @ delta3) * (y2 > 0) * mask2
        delta1 = (W2.T @ delta2) * (y1 > 0) * mask1

        W1 += alpha * delta1 @ x.T
        W2 += alpha * delta2 @ y1_drop.T
        W3 += alpha * delta3 @ y2_drop.T
        W4 += alpha * delta4 @ y3_drop.T

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

         # # 前向传播计算（四隐层）
         # v1 = W1 @ x
         # y1 = sigmoid(v1)
         # v2 = W2 @ y1
         # y2 = sigmoid(v2)
         # v3 = W3 @ y2
         # y3 = sigmoid(v3)
         # v4 = W4 @ y3
         # y = softmax(v4)

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