import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Optional
import torch.nn.functional as F


class Config:
    def __init__(self,
                 window_size=90,
                 forecast_horizon=90,
                 batch_size=32,
                 dropout_rate=0.2,
                 lr=0.001,
                 epochs=100,
                 num_exp=5,
                 drop_cols=None):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.num_exp = num_exp
        self.drop_cols = drop_cols if drop_cols else []


# ============================ 数据预处理 ============================
def preprocess(df, feature_scaler=None, target_scaler=None):
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    df['season_num'] = df['season'].map(season_map)
    df['season_sin'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season_num'] / 4)

    df['dayofyear'] = df['Date'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(float)
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

    df['is_weekend'] = df['is_weekend'].astype(float)

    df.drop(['season', 'season_num', 'dayofyear', 'weekofyear'], axis=1, inplace=True)

    no_scale_cols = ['Date', 'Global_active_power']
    features_to_scale = [col for col in df.columns if col not in no_scale_cols]
    df[features_to_scale] = df[features_to_scale].astype(np.float32)

    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
        df[features_to_scale] = feature_scaler.fit_transform(df[features_to_scale])
    else:
        df[features_to_scale] = feature_scaler.transform(df[features_to_scale])

    if target_scaler is None:
        target_scaler = MinMaxScaler()
        df['Global_active_power'] = target_scaler.fit_transform(df[['Global_active_power']]).astype(np.float32)
    else:
        df['Global_active_power'] = target_scaler.transform(df[['Global_active_power']]).astype(np.float32)

    return df, feature_scaler, target_scaler


def create_sliding_windows(df, window_size, forecast_horizon, step=1):
    features = df.drop(['Date', 'Global_active_power'], axis=1)
    features = features.astype(np.float32)
    target = df['Global_active_power'].values.astype(np.float32)

    X, y = [], []
    for i in range(0, len(features) - window_size - forecast_horizon + 1, step):
        X.append(features.iloc[i:i + window_size].values)
        y.append(target[i + window_size:i + window_size + forecast_horizon])
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    return X_array, y_array


# ============================ LSTM模型定义 ============================
# 时序注意力
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2))
        attention = self.softmax(scores / (x.size(-1) ** 0.5))
        out = torch.bmm(attention, V)
        return out

# 多尺度LSTM特征提取器
class MultiScaleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.lstm_main = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.downsample = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm_aux = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True
        )
        self.fusion = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
    def forward(self, x):
        out_main, _ = self.lstm_main(x)
        x_ds = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        out_aux, _ = self.lstm_aux(x_ds)
        out_aux = F.interpolate(
            out_aux.transpose(1, 2),
            size=x.size(1),
            mode='linear',
            align_corners=True
        ).transpose(1, 2)
        combined = torch.cat([out_main, out_aux], dim=-1)
        return self.fusion(combined)

#完整模型
class ImprovedLSTMModel(nn.Module):
    def __init__(self, num_features, pred_steps, dropout_rate=0.3):
        super().__init__()
        self.num_features = num_features
        self.pred_steps = pred_steps
        self.feature_extractor = MultiScaleLSTM(input_size=num_features,hidden_size=256,num_layers=2,dropout_rate=dropout_rate)

        self.temporal_attn = TemporalAttention(256)

        if pred_steps <= 90:
            self.decoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, pred_steps)
            )
        else:
            self.step_decoder = nn.LSTMCell(num_features + 256, 256)
            self.step_proj = nn.Linear(256, 1)
            self.trend_decoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, pred_steps - 90)
            )
    def forward(self, x):
        features = self.feature_extractor(x)
        attn_features = self.temporal_attn(features)
        context = attn_features[:, -1]
        if self.pred_steps <= 90:
            return self.decoder(context)
        else:
            predictions = []
            h_t = context
            c_t = torch.zeros_like(h_t)
            last_input = x[:, -1]
            for _ in range(90):
                combined = torch.cat([last_input, h_t], dim=-1)
                h_t, c_t = self.step_decoder(combined, (h_t, c_t))
                pred = self.step_proj(h_t)
                predictions.append(pred)
                last_input = torch.cat([last_input[:, :self.num_features - 1], pred], dim=-1)
            trend = self.trend_decoder(context)
            return torch.cat([
                torch.stack(predictions, dim=1).squeeze(-1),
                trend
            ], dim=1)


# ============================ 训练与预测 ============================
def train_and_evaluate(config: Config):
    train_df = pd.read_csv('/content/drive/MyDrive/ML_homework/data/processed/processed_train.csv',parse_dates=['Date'])
    test_df = pd.read_csv('/content/drive/MyDrive/ML_homework/data/processed/processed_test.csv', parse_dates=['Date'])
    train_df, feature_scaler, target_scaler = preprocess(train_df)
    test_df, _, _ = preprocess(test_df, feature_scaler, target_scaler)
    if config.drop_cols:
        train_df.drop(columns=config.drop_cols, inplace=True, errors='ignore')
        test_df.drop(columns=config.drop_cols, inplace=True, errors='ignore')
    X_train, y_train = create_sliding_windows(train_df, config.window_size, config.forecast_horizon)
    X_test, y_test = create_sliding_windows(test_df, config.window_size, config.forecast_horizon)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train),batch_size=config.batch_size,shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),batch_size=config.batch_size,shuffle=False)

    mses, maes, mse_stds, mae_stds = [], [], [], []
    for exp in range(config.num_exp):
        print(f'\n第{exp + 1}次实验：')
        model = ImprovedLSTMModel(
            num_features=X_train.shape[2],
            pred_steps=config.forecast_horizon,
            dropout_rate=config.dropout_rate
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        best_loss = float('inf')
        train_losses = []
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)

            model.eval()
            with torch.no_grad():
                val_pred, val_true = [], []
                for X_val, y_val in test_loader:
                    pred = model(X_val)
                    val_pred.append(pred.cpu())
                    val_true.append(y_val.cpu())
                val_pred = torch.cat(val_pred).numpy()
                val_true = torch.cat(val_true).numpy()
                val_mse = mean_squared_error(val_true, val_pred)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            train_losses.append(avg_loss)

            if val_mse < best_loss:
                best_loss = val_mse
                torch.save(model.state_dict(), 'best_model.pth')
        plt.figure()
        plt.plot(train_losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()

        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        with torch.no_grad():
            test_pred, test_true = [], []
            for X_test_batch, y_test_batch in test_loader:
                pred = model(X_test_batch)
                test_pred.append(pred.cpu())
                test_true.append(y_test_batch.cpu())

            test_pred = torch.cat(test_pred).numpy()
            test_true = torch.cat(test_true).numpy()
            mse_std = mean_squared_error(test_true, test_pred)
            mae_std = mean_absolute_error(test_true, test_pred)
            test_pred = target_scaler.inverse_transform(test_pred)
            test_true = target_scaler.inverse_transform(test_true)
            mse = mean_squared_error(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)
            mses.append(mse)
            maes.append(mae)
            mse_stds.append(mse_std)
            mae_stds.append(mae_std)
            print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"测试集（标准化）MSE: {mse_std:.4f}, MAE: {mae_std:.4f}")

    print(f"\n平均 MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"平均 MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"\n平均（标准化）MSE: {np.mean(mse_stds):.4f} ± {np.std(mse_stds):.4f}")
    print(f"平均（标准化）MAE: {np.mean(mae_stds):.4f} ± {np.std(mae_stds):.4f}")

    # 可视化预测
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        y_pred, y_true = [], []
        for X_test_batch, y_test_batch in test_loader:
            pred = model(X_test_batch)
            y_pred.append(pred.cpu())
            y_true.append(y_test_batch.cpu())
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        y_pred = target_scaler.inverse_transform(y_pred)
        y_true = target_scaler.inverse_transform(y_true)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:200, 0], 'b-', label='True (step 1)')
    plt.plot(y_pred[:200, 0], 'r--', label='Pred (step 1)')
    plt.legend()
    plt.grid(True)
    plt.title('First-Step Forecast for First 200 Samples')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(range(config.forecast_horizon), y_true[0], 'bo-', label='True')
    plt.plot(range(config.forecast_horizon), y_pred[0], 'rx--', label='Predicted')
    plt.legend()
    plt.grid(True)
    plt.title('Full Horizon Prediction (Sample 0)')
    plt.show()


if __name__ == '__main__':
    config_90 = Config(
        window_size=90,
        forecast_horizon=90,
        batch_size=32,
        dropout_rate=0.2,
        lr=0.001,
        epochs=100,
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5', 'Global_reactive_power', 'Voltage', 'NBJRR1', 'is_weekend']
    )

    config_365 = Config(
        window_size=90,
        forecast_horizon=365,
        batch_size=16,
        dropout_rate=0.3,
        lr=0.0005,
        epochs=150,
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5', 'Global_reactive_power', 'Voltage', 'NBJRR1', 'is_weekend']
    )

    print("预测90天：")
    train_and_evaluate(config_90)

    print("\n预测365天：")
    train_and_evaluate(config_365)