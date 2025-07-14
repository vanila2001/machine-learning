import pandas as pd
import numpy as np


# 读取数据函数
def read_data(file_path):
    # 读取CSV文件并返回DataFrame
    df = pd.read_csv(file_path)
    print("原始数据基本信息：")
    print(df.info())
    return df


# 填充缺失值
def fill_data(df):
    # 将DateTime列转换为日期时间类型
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # 定义需要转换为数值的列
    str_to_num_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1',
                       'Sub_metering_2']

    # 转换字符串列为数值型，非数值内容设为NaN
    for col in str_to_num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 设置DateTime列为索引
    df.set_index('DateTime', inplace=True)

    # 按时间排序
    df.sort_index(inplace=True)

    # 处理缺失值，采用时间插值填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='time')

    # 检查是否还有缺失值，若有采用均值填充
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())

    return df


# 填充缺失分钟的日期
def process_minute_data(df, missing_threshold=200):
    print("开始填充缺失分钟数据的日期")
    # 按日期分组处理
    date_groups = df.groupby(df.index.date)
    valid_dates = []
    filled_data = []

    # 过滤并填充缺失分钟
    for date, group in date_groups:
        # 计算缺失分钟数
        expected_minutes = 1440  # 每天1440分钟
        actual_minutes = len(group)
        missing_minutes = expected_minutes - actual_minutes

        if missing_minutes > missing_threshold:
            print(f"日期 {date} 缺失 {missing_minutes} 分钟，超过阈值 {missing_threshold}，已丢弃")
            continue

        # 生成完整分钟索引
        full_index = pd.date_range(
            start=f"{date} 00:00:00",
            end=f"{date} 23:59:00",
            freq='min'
        )

        # 重采样并填充缺失值
        resampled = group.reindex(full_index)
        for col in resampled.columns:
            resampled[col] = resampled[col].interpolate(method='time')

        filled_data.append(resampled)
        valid_dates.append(date)

    if not filled_data:
        print("没有找到有效日期数据")

    # 合并所有有效日期数据
    full_minute_data = pd.concat(filled_data)
    print(f"处理后数据行数: {len(full_minute_data)}，有效日期数: {len(valid_dates)}")

    return full_minute_data


def daily_merge(df):
    # 按天进行数据聚合
    df['Date'] = df.index.date

    # 分组聚合
    grouped = df.groupby('Date')

    # 按天求和的列
    sum_cols = ['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    daily_sum = grouped[sum_cols].sum().reset_index()

    # 按天求平均的列
    mean_cols = ['Voltage', 'Global_intensity']
    daily_mean = grouped[mean_cols].mean().reset_index()

    # 气候数据取每天第一个值
    climate_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    daily_climate = grouped[climate_cols].first().reset_index()

    # 合并聚合后的数据
    daily_data = pd.merge(daily_sum, daily_mean, on='Date', how='outer')
    daily_data = pd.merge(daily_data, daily_climate, on='Date', how='outer')

    # 计算未被分表记录的能耗
    daily_data['sub_metering_remainder'] = (daily_data['Global_active_power'] * 1000 / 60) - (
            daily_data['Sub_metering_1'] + daily_data['Sub_metering_2'] + daily_data['Sub_metering_3'])

    # 确保计算结果不为负
    daily_data['sub_metering_remainder'] = daily_data['sub_metering_remainder'].clip(lower=0)

    return daily_data


# 处理异常值
def handle_outliers(df):
    # 使用IQR方法处理所有列的异常值
    for col in df.columns:
        # 计算IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 截断异常值
        df[col] = df[col].clip(lower_bound, upper_bound)

    return df


def weekend_season(df):
    # 确保Date列是datetime类型
    df['Date'] = pd.to_datetime(df['Date'])

    # 添加是否为周末的列
    df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6])

    def get_season(month):
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:  # 12, 1, 2
            return "winter"

    # 添加季节列
    df['season'] = df['Date'].dt.month.apply(get_season)

    return df


def main():
    file_path = 'test.csv'
    output_path = 'processed_test.csv'

    # 读取数据
    df = read_data(file_path)

    # 填充缺失值
    df = fill_data(df)

    # 填充缺失分钟的日期
    df = process_minute_data(df)

    # 按天进行数据聚合
    df = daily_merge(df)

    # 处理异常值
    df = handle_outliers(df)

    # 标记周末和季节
    df = weekend_season(df)

    # 保存预处理后的数据
    df.to_csv(output_path, index=False)
    print(f"预处理后的数据已保存至 {output_path}")
    print("处理后的数据基本信息：")
    print(df.info())
    print(df.head())


if __name__ == "__main__":
    main()
