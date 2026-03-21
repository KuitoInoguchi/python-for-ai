# This module for data visualization is AI-generated

import numpy as np
import utils
import matplotlib.pyplot as plt

def compute_error(W, X, D):
    """计算误差，即四个实际输出与期望输出之差的平方和"""
    error = 0
    for i in range(len(X)):
        x = X[i:i + 1, :]
        v = np.dot(W, x.T)
        y = utils.sigmoid(v)
        error += (D[i] - y[0][0]) ** 2
    return error


def main():
    """主入口函数，支持三种训练方法：sgd、batch、small_batch"""
    # 训练数据
    X, D = utils.get_preset_training_data()
    
    # 随机权重初始化
    rg = np.random.default_rng(0)
    initial_W = 2 * rg.random((1, 3)) - 1
    
    # 训练方法列表
    methods = ["sgd", "batch", "small_batch"]
    epochs = 1000
    alpha = 0.3
    
    # 存储每种方法的误差历史
    error_histories = {}
    
    for method in methods:
        print(f"Training with {method.upper()}...")

        W = initial_W.copy()
        print(f"Initial weights for {method}: {W}")
        
        # 存储每轮的误差
        errors = []
        
        # 训练
        for epoch in range(epochs):
            if method == "sgd":
                utils.sgd(W, X, D, alpha)
            elif method == "batch":
                utils.batch_sgd(W, X, D, alpha)
            elif method == "small_batch":
                utils.mini_batch_sgd(W, X, D, alpha)
            
            # 计算当前轮的误差
            error = compute_error(W, X, D)
            errors.append(error)
        
        error_histories[method] = errors
        print(f"{method.upper()} training completed.")
    
    # 绘制轮-误差曲线
    plt.figure(figsize=(10, 6))
    for method, errors in error_histories.items():
        plt.plot(range(epochs), errors, label=method.upper())
    
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs Epochs for Different Training Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_vs_epochs.png')
    plt.show()
    
    print("Visualization completed. Plot saved as error_vs_epochs.png")


if __name__ == "__main__":
    main()
