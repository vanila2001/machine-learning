import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class Config:
    def __init__(self,
                 window_size=90,
                 forecast_horizon=365,
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
    df['is_weekend'] = df['is_weekend'].astype(int)
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    df['season_num'] = df['season'].map(season_map)
    df['season_sin'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season_num'] / 4)
    df.drop(['season', 'season_num'], axis=1, inplace=True)

    df['dayofyear'] = df['Date'].dt.dayofyear
    df['day_sin'] = np.sin(2*np.pi*df['dayofyear']/365)
    df['day_cos'] = np.cos(2*np.pi*df['dayofyear']/365)
    df = df.drop(['dayofyear'], axis=1)

    no_scale_cols = ['Date', 'Global_active_power', 'season_sin', 'season_cos', 'is_weekend','day_sin','day_cos']
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


def create_sliding_windows(df, window_size, forecast_horizon, step=1):
    X, y = [], []
    features = df.drop(['Date', 'Global_active_power'], axis=1).values
    target = df['Global_active_power'].values
    for i in range(0, len(features) - window_size - forecast_horizon + 1, step):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size: i + window_size + forecast_horizon])
    return np.array(X), np.array(y)


# ============================ LSTM模型定义 ============================
class LSTMModel(nn.Module):
    def __init__(self, num_features, num_outputs, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features,hidden_size=256,num_layers=3,batch_first=True,dropout=dropout_rate,)
        self.shortcut = nn.Sequential(nn.Linear(num_features, 128),nn.ReLU(),nn.Linear(128, num_outputs))
        self.fc = nn.Sequential(nn.Linear(256, 128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128, 64),nn.ReLU(),nn.Linear(64, num_outputs))
    def forward(self, x):
        residual = self.shortcut(x[:, -1, :])
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out + residual


# ============================ 训练与预测 ============================
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


# ============================ 主函数入口 ============================
def main(config: Config):
    train_df = pd.read_csv('processed_train.csv', parse_dates=['Date'])
    test_df = pd.read_csv('processed_test.csv', parse_dates=['Date'])

    train_df, feature_scaler, target_scaler = preprocess(train_df)
    test_df, _, _ = preprocess(test_df, feature_scaler, target_scaler)

    train_df.drop(columns=config.drop_cols, inplace=True, errors='ignore')
    test_df.drop(columns=config.drop_cols, inplace=True, errors='ignore')

    X_train, y_train = create_sliding_windows(train_df, config.window_size, config.forecast_horizon)
    X_test, y_test = create_sliding_windows(test_df, config.window_size, config.forecast_horizon)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

    best_model, best_mse = None, float('inf')
    mses, maes, mse_stds, mae_stds = [], [], [], []

    for exp in range(config.num_exp):
        print(f'\n第{exp + 1}次实验：')
        model = LSTMModel(num_features=X_train.shape[2], num_outputs=config.forecast_horizon, dropout_rate=config.dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        train_losses = []
        for epoch in range(config.epochs):
            loss = train(model, train_loader, optimizer, criterion, device)
            train_losses.append(loss)
            scheduler.step(loss)
            print(f"Epoch {epoch}: Train Loss = {loss:.4f}, 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        plt.figure()
        plt.plot(train_losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()

        y_pred, y_true = predict(model, test_loader, device, target_scaler)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mses.append(mse)
        maes.append(mae)
        print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")

        y_pred_std, y_true_std = predict(model, test_loader, device, target_scaler, scale_back=False)
        test_mse_std = mean_squared_error(y_true_std, y_pred_std)
        test_mae_std = mean_absolute_error(y_true_std, y_pred_std)
        mse_stds.append(test_mse_std)
        mae_stds.append(test_mae_std)
        print(f"测试集（标准化）MSE: {test_mse_std:.4f}, MAE: {test_mae_std:.4f}")

        if mse < best_mse:
            best_model = model
            best_mse = mse

    print(f"\n平均 MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"平均 MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"\n平均（标准化）MSE: {np.mean(mse_stds):.4f} ± {np.std(mse_stds):.4f}")
    print(f"平均（标准化）MAE: {np.mean(mae_stds):.4f} ± {np.std(mae_stds):.4f}")


    # 可视化预测
    y_pred, y_true = predict(best_model, test_loader, device, target_scaler)
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
    config = Config(
        window_size=90,
        forecast_horizon=365,
        epochs=150,
        # drop_cols=['NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5', 'Global_reactive_power', 'Voltage', 'NBJRR1', 'is_weekend', 'Sub_metering_2','Sub_metering_1','Sub_metering_3']
    )
    main(config)