## 作业4：训练浅层NN解决XOR问题
针对XOR问题，下面对比单层神经网络和浅层神经网络的训练效果。
两者训练轮数均为1000轮，学习率均为0.9

### 单层神经网络
- 调用函数：utils.sgd()
- 训练结果：
```
After training with SGD method: 
Weights: [[-0.23750569 -0.11875285  0.11875285]]
input: [0 0 1] -> output: 0.5297 (desired: 0)
input: [0 1 1] -> output: 0.5000 (desired: 1)
input: [1 0 1] -> output: 0.4703 (desired: 1)
input: [1 1 1] -> output: 0.4409 (desired: 0)
```
### 浅层神经网络
- 隐藏层节点数：4
- 调用函数：utils.back_prop_xor()
- 训练结果：
```
After training with SGD method: 
Weight Matrix 1: 
[[ 4.96302643 -4.49428249 -3.10536023]
 [-3.12940538  2.58865519  2.39852958]
 [-4.4010095   5.03225818 -2.8140466 ]
 [ 2.39032983  1.31428158 -3.51514742]]
Weight Matrix 2: 
[[ 5.74058849 -3.66711955  7.5023542  -2.31959545]]
input: [0 0 1] -> output: 0.0596 (desired: 0)
input: [0 1 1] -> output: 0.9476 (desired: 1)
input: [1 0 1] -> output: 0.9612 (desired: 1)
input: [1 1 1] -> output: 0.0356 (desired: 0)
```
对比两次训练结果可以看出，通过引入一个4节点的隐藏层，浅层神经网络
有效解决了XOR的分类问题。

## 作业5：隐藏层节点个数对浅层神经网络训练效果的影响
本环节训练轮数均为1000轮，学习率均为0.9。下面对比分析隐藏层节点
数依次为2，3，5时的训练结果。
### 节点数为2
```
After training with SGD method: 
Weight Matrix 1: 
[[ 2.99067664 -5.86885538 -2.2570351 ]
 [-3.63368746 -5.91251225  0.63406875]]
Weight Matrix 2: 
[[ 4.2271443  -4.70507115]]
input: [0 0 1] -> output: 0.0645 (desired: 0)
input: [0 1 1] -> output: 0.4943 (desired: 1)
input: [1 0 1] -> output: 0.9329 (desired: 1)
input: [1 1 1] -> output: 0.5060 (desired: 0)
```

### 节点数为3
```
After training with SGD method: 
Weight Matrix 1: 
[[ 5.56155207 -5.52271852 -2.99592986]
 [-3.51404122  3.1895444  -1.73847662]
 [ 3.7780567  -3.69539694  2.04992286]]
Weight Matrix 2: 
[[ 8.16137016  4.66706929 -4.60787628]]
input: [0 0 1] -> output: 0.0476 (desired: 0)
input: [0 1 1] -> output: 0.9542 (desired: 1)
input: [1 0 1] -> output: 0.9530 (desired: 1)
input: [1 1 1] -> output: 0.0395 (desired: 0)
```

### 节点数为5
```
After training with SGD method: 
Weight Matrix 1: 
[[-4.37450805 -4.89017388  1.56005058]
 [-3.65052858  2.22959505 -0.86010647]
 [-0.41822858 -0.08949208 -0.58677493]
 [ 3.3078148   2.53981775 -4.48979648]
 [ 3.70166689 -4.62461917 -2.02508705]]
Weight Matrix 2: 
[[-6.33995601  4.30490203  0.7821327  -4.71945635  5.10381508]]
input: [0 0 1] -> output: 0.0416 (desired: 0)
input: [0 1 1] -> output: 0.9476 (desired: 1)
input: [1 0 1] -> output: 0.9565 (desired: 1)
input: [1 1 1] -> output: 0.0518 (desired: 0)
```
对比训练结果发现，当节点数为2时，浅层神经网络不能解决XOR
问题，训练无效；当节点数为3或5时，训练效果良好且无明显差距。

## 作业6：用动量算法训练浅层NN求解XOR问题
两次训练轮数为475轮，学习率均为0.9。
### beta = 0
```
After training with SGD method: 
Weight Matrix 1: 
[[ 4.95672455 -4.94102331 -2.77024319]
 [-2.8524813   2.40979347 -1.19112107]
 [ 3.09069712 -3.03568218  1.62338566]]
Weight Matrix 2: 
[[ 6.42888374  3.49800812 -3.79667151]]
input: [0 0 1] -> output: 0.1216 (desired: 0)
input: [0 1 1] -> output: 0.8764 (desired: 1)
input: [1 0 1] -> output: 0.8887 (desired: 1)
input: [1 1 1] -> output: 0.0959 (desired: 0)
```
### beta = 0.9
``` 
After training with SGD method: 
Weight Matrix 1: 
[[ 2.23770433  1.76620401  0.89345238]
 [-3.19144124 -3.02635623  4.35514384]
 [-6.15280296 -5.87143641  2.00945393]]
Weight Matrix 2: 
[[-2.24226273  5.92154422 -8.06186693]]
input: [0 0 1] -> output: 0.0545 (desired: 0)
input: [0 1 1] -> output: 0.9184 (desired: 1)
input: [1 0 1] -> output: 0.9036 (desired: 1)
input: [1 1 1] -> output: 0.1931 (desired: 0)
```
对比发现，引入动量后，第二次训练的误差收敛大致稍快于第一次训练，
但差距不明显。猜想动量算法可以加快模型训练速度，但在本问题上效果有限。