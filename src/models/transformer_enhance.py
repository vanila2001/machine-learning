import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
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
    features = df.drop(['Date', 'Global_active_power'], axis=1).astype(np.float32)
    target = df['Global_active_power'].values.astype(np.float32)

    X, y = [], []
    for i in range(0, len(features) - window_size - forecast_horizon + 1, step):
        X.append(features.iloc[i:i + window_size].values)
        y.append(target[i + window_size:i + window_size + forecast_horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LearnableTemporalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model_input) → only need seq_len
        B, T, _ = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        pos_embed = self.embedding(positions)  # shape: (B, T, d_model)
        return x + pos_embed

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, num_features, d_model=256, nhead=8, num_layers=3, dropout=0.2, pred_steps=90):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.temporal_embedding = LearnableTemporalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_steps)
        )

        self.shortcut = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_steps)
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.d_model = d_model

    def forward(self, x, contrastive=False):
        # x: (batch, seq_len, features)
        last_input = x[:, -1, :]  # (batch, num_features)
        x_proj = self.input_proj(x) * np.sqrt(self.d_model)
        x_proj = self.temporal_embedding(x_proj)
        x_proj = x_proj.permute(1, 0, 2)  # (seq_len, batch, d_model)
        encoded = self.transformer_encoder(x_proj)  # (seq_len, batch, d_model)
        last_step = encoded[-1]  # (batch, d_model)

        out = self.decoder(last_step) + self.shortcut(last_input)  # 残差连接

        if contrastive:
            z = self.projector(last_step)  # (batch, 64)
            z = F.normalize(z, dim=1)      # L2 normalize
            return out, z
        else:
            return out


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2B, 2B]
    sim_matrix /= temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)  # 排除自己

    positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0)
    loss = -torch.log(torch.exp(positives) / torch.exp(sim_matrix).sum(dim=1))
    return loss.mean()



def train_and_evaluate(config: Config):
    train_df = pd.read_csv('processed_train.csv', parse_dates=['Date'])
    test_df = pd.read_csv('processed_test.csv', parse_dates=['Date'])

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

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

    mses, maes, mse_stds, mae_stds = [], [], [], []

    for exp in range(config.num_exp):
        print(f"\n第{exp + 1}次实验:")
        model = TransformerTimeSeriesModel(
            num_features=X_train.shape[2],
            d_model=256,
            nhead=8,
            num_layers=3,
            dropout=config.dropout_rate,
            pred_steps=config.forecast_horizon
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        train_losses = []

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # 生成两个视图
                X1 = X_batch
                X2 = X_batch + 0.01 * torch.randn_like(X_batch)

                y_pred, z1 = model(X1, contrastive=True)
                _, z2 = model(X2, contrastive=True)

                loss_pred = criterion(y_pred, y_batch)
                loss_cl = nt_xent_loss(z1, z2)
                loss = loss_pred + 0.2 * loss_cl

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            # 验证
            model.eval()
            with torch.no_grad():
                val_preds, val_trues = [], []
                for X_val, y_val in test_loader:
                    preds = model(X_val)
                    val_preds.append(preds.cpu())
                    val_trues.append(y_val.cpu())
                val_preds = torch.cat(val_preds).numpy()
                val_trues = torch.cat(val_trues).numpy()
                val_mse = mean_squared_error(val_trues, val_preds)

            print(
                f"Epoch {epoch + 1}: Train Loss={avg_loss:.6f}, Val MSE={val_mse:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
            train_losses.append(avg_loss)

        # 画训练曲线
        plt.figure()
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.show()

        # 测试评估，直接用训练结束的模型
        model.eval()
        with torch.no_grad():
            test_preds, test_trues = [], []
            for X_batch, y_batch in test_loader:
                preds = model(X_batch)
                test_preds.append(preds.cpu())
                test_trues.append(y_batch.cpu())
            test_preds = torch.cat(test_preds).numpy()
            test_trues = torch.cat(test_trues).numpy()

            test_mse_std = mean_squared_error(test_trues, test_preds)
            test_mae_std = mean_absolute_error(test_trues, test_preds)
            mse_stds.append(test_mse_std)
            mae_stds.append(test_mae_std)
            print(f"测试集（标准化）MSE: {test_mse_std:.4f}, MAE: {test_mae_std:.4f}")
            # 反归一化
            test_preds_inv = target_scaler.inverse_transform(test_preds)
            test_trues_inv = target_scaler.inverse_transform(test_trues)

            mse = mean_squared_error(test_trues_inv, test_preds_inv)
            mae = mean_absolute_error(test_trues_inv, test_preds_inv)
            print(f"测试集 MSE: {mse:.4f}, MAE: {mae:.4f}")

            mses.append(mse)
            maes.append(mae)

        # 预测曲线示例（第0条样本第1步）
        plt.figure(figsize=(12, 6))
        plt.plot(test_trues_inv[:200, 0], label='True (step 1)')
        plt.plot(test_preds_inv[:200, 0], label='Pred (step 1)')
        plt.legend()
        plt.title('First Step Prediction (First 200 samples)')
        plt.grid(True)
        plt.show()

        # 预测完整时间步示例（第0条样本）
        plt.figure(figsize=(12, 6))
        plt.plot(range(config.forecast_horizon), test_trues_inv[0], 'b-o', label='True')
        plt.plot(range(config.forecast_horizon), test_preds_inv[0], 'r-x', label='Predicted')
        plt.title('Full Horizon Prediction (Sample 0)')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Transformer_enhance模型 {config.forecast_horizon}天预测结果：")
    print(f"平均 MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"平均 MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"平均（标准化）MSE: {np.mean(mse_stds):.4f} ± {np.std(mse_stds):.4f}")
    print(f"平均（标准化）MAE: {np.mean(mae_stds):.4f} ± {np.std(mae_stds):.4f}")

if __name__ == '__main__':
    config_90 = Config(
        window_size=90,
        forecast_horizon=90,
        batch_size=32,
        dropout_rate=0.2,
        lr=0.001,
        epochs=200,
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5',
                   'Global_reactive_power', 'Voltage', 'NBJRR1',
                   'is_weekend', 'Sub_metering_2', 'Sub_metering_1', 'Sub_metering_3']
    )

    print("开始90天预测：")
    train_and_evaluate(config_90)

    config_365 = Config(
        window_size=90,
        forecast_horizon=365,
        batch_size=32,
        dropout_rate=0.2,
        lr=0.001,
        epochs=200,
        drop_cols=['NBJBROU', 'NBJRR10', 'RR', 'NBJRR5',
                   'Global_reactive_power', 'Voltage', 'NBJRR1',
                   'is_weekend', 'Sub_metering_2', 'Sub_metering_1', 'Sub_metering_3']
    )

    print("开始365天预测：")
    train_and_evaluate(config_365)
