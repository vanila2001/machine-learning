import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


# ===================== 参数配置 =====================
@dataclass
class Config:
    window_size: int = 90
    forecast_horizon: int = 90
    batch_size: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 150
    num_experiments: int = 5
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv: str = 'processed_train.csv'
    test_csv: str = 'processed_test.csv'
    drop_cols: list = None


# ===================== 数据预处理 =====================
def preprocess(df, feature_scaler=None, target_scaler=None):
    df['is_weekend'] = df['is_weekend'].astype(int)
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    df['season_num'] = df['season'].map(season_map)
    df['season_sin'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season_num'] / 4)
    df = df.drop(['season', 'season_num'], axis=1)

    no_scale_cols = ['Date', 'Global_active_power', 'season_sin', 'season_cos', 'is_weekend']
    features_to_scale = [col for col in df.columns if col not in no_scale_cols]

    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
        df[features_to_scale] = feature_scaler.fit_transform(df[features_to_scale])
    else:
        df[features_to_scale] = feature_scaler.transform(df[features_to_scale])

    if target_scaler is None:
        target_scaler = MinMaxScaler()
        df['Global_active_power'] = target_scaler.fit_transform(df[['Global_active_power']])
    else:
        df['Global_active_power'] = target_scaler.transform(df[['Global_active_power']])

    return df, feature_scaler, target_scaler


# ===================== 滑动窗口 =====================
def create_sliding_windows(df, window_size, forecast_horizon):
    X, y = [], []
    features = df.drop(['Date', 'Global_active_power'], axis=1).values
    target = df['Global_active_power'].values
    for i in range(len(features) - window_size - forecast_horizon + 1):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size: i + window_size + forecast_horizon])
    return np.array(X), np.array(y)

# ===================== Transformer模型 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不作为参数优化

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
class TransformerModel(nn.Module):
    def __init__(self, num_features, num_outputs, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_linear = nn.Linear(num_features, d_model)  # 映射到d_model维度
        self.pos_encoder = PositionalEncoding(d_model)        # 加入位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_outputs)  # 输出层

    def forward(self, x):
        x = self.input_linear(x)           # → [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)            # 加位置编码
        x = self.transformer_encoder(x)    # Transformer编码器
        x = x[:, -1, :]                    # 取最后一个时间步的表示
        out = self.fc(x)                   # → [batch_size, num_outputs]
        return out

# ===================== 训练函数 =====================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def predict(model, loader, device, target_scaler=None, scale_back=True):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            if scale_back and target_scaler is not None:
                outputs = target_scaler.inverse_transform(outputs)
                y_batch = target_scaler.inverse_transform(y_batch)

            predictions.append(outputs)
            actuals.append(y_batch)
    return np.vstack(predictions), np.vstack(actuals)


# ===================== 主流程 =====================
def main(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv, parse_dates=['Date'])
    test_df = pd.read_csv(cfg.test_csv, parse_dates=['Date'])

    train_df, feature_scaler, target_scaler = preprocess(train_df)
    test_df, _, _ = preprocess(test_df, feature_scaler, target_scaler)

    if cfg.drop_cols:
        train_df.drop(columns=cfg.drop_cols, inplace=True)
        test_df.drop(columns=cfg.drop_cols, inplace=True)

    X_train, y_train = create_sliding_windows(train_df, cfg.window_size, cfg.forecast_horizon)
    X_test, y_test = create_sliding_windows(test_df, cfg.window_size, cfg.forecast_horizon)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(cfg.device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(cfg.device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(cfg.device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(cfg.device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    best_model = None
    best_mse = float('inf')
    mses, maes, mse_stds, mae_stds = [], [], [], []
    for exp in range(cfg.num_experiments):
        print(f'\nExperiment {exp + 1}')
        model = TransformerModel(
            num_features=X_train.shape[2], num_outputs=cfg.forecast_horizon,
            d_model=cfg.d_model, nhead=cfg.nhead,
            num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout_rate
        ).to(cfg.device)

        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        train_losses = []
        for epoch in range(cfg.epochs):
            loss = train(model, train_loader, optimizer, criterion, cfg.device)
            train_losses.append(loss)
            scheduler.step(loss)
            print(f"Epoch {epoch + 1}: Train Loss = {loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        y_pred, y_true = predict(model, test_loader, cfg.device, target_scaler)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mses.append(mse)
        maes.append(mae)
        print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")

        y_pred_std, y_true_std = predict(model, test_loader, cfg.device, target_scaler, scale_back=False)
        test_mse_std = mean_squared_error(y_true_std, y_pred_std)
        test_mae_std = mean_absolute_error(y_true_std, y_pred_std)
        mse_stds.append(test_mse_std)
        mae_stds.append(test_mae_std)
        print(f"测试集（标准化）MSE: {test_mse_std:.4f}, MAE: {test_mae_std:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_model = model

        # 绘图
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.title(f'Training Loss Curve (Exp {exp + 1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Transformer模型 {cfg.forecast_horizon}天预测结果：")
    print(f"平均 MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"平均 MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"平均（标准化）MSE: {np.mean(mse_stds):.4f} ± {np.std(mse_stds):.4f}")
    print(f"平均（标准化）MAE: {np.mean(mae_stds):.4f} ± {np.std(mae_stds):.4f}")

    # 可视化预测
    y_pred, y_true = predict(best_model, test_loader, cfg.device, target_scaler)
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:200, 0], label='True')
    plt.plot(y_pred[:200, 0], label='Predicted')
    plt.title('First-Step Forecast for First 200 Samples')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    # plt.plot(y_true[0], label='True Sequence')
    # plt.plot(y_pred[0], label='Predicted Sequence')
    plt.plot(range(config.forecast_horizon), y_true[0], 'bo-', label='True')
    plt.plot(range(config.forecast_horizon), y_pred[0], 'rx--', label='Predicted')
    plt.title('Full Horizon Prediction (Sample 0)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    config = Config(
        window_size=90,
        forecast_horizon=90,
        # drop_cols=['NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5', 'Global_reactive_power', 'Voltage', 'NBJRR1', 'is_weekend', 'Sub_metering_2','Sub_metering_1','Sub_metering_3']
    )
    main(config)

    config = Config(
        window_size=90,
        forecast_horizon=365,
        epochs=200,
        d_model=256,
        dim_feedforward=512,
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5', 'Global_reactive_power', 'Voltage', 'NBJRR1', 'is_weekend', 'Sub_metering_2','Sub_metering_1','Sub_metering_3']
    )
    main(config)
