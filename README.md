# 机器学习期末大作业：时间序列预测
本项目专注于使用LSTM和Transformer模型进行时间序列预测，包含数据处理和模型训练的结构化工作流程。


## 1. 项目文件结构
machine-learning/
├── data/
│   ├── raw/                  # 存放原始数据
│   └── processed/            # 预处理后的数据
│       ├── processed_train.csv  # 预处理后的训练集
│       └── processed_test.csv   # 预处理后的测试集
└── src/
    ├── data_process/
    │   └── preprocess.py     # 数据预处理代码
    └── models/
        ├── lstm/
        │   ├── lstm_90.py    # LSTM模型（90天预测）
        │   ├── lstm_365.py   # LSTM模型（365天预测）
        │   └── lstm_enhance.py  # LSTM改进模型
        └── transformer/
            ├── transformer.py    # Transformer基础模型
            └── transformer_enhance.py  # Transformer改进模型
            

## 2. 快速开始

### 步骤1：克隆仓库
首先，将项目保存到本地机器并导航到项目目录：
```bash
git clone https://github.com/vanila2001/machine-learning.git
cd machine-learning
