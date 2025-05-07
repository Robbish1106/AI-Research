import os
import pandas as pd

# === 設定資料夾（你自己的資料位置）===
data_folder = os.path.abspath(os.path.join("..", "資料", "資料集"))

# 存放所有月資料
df_all = []

# 掃描資料夾中所有檔案
for file in sorted(os.listdir(data_folder)):
    if file.endswith('.csv') and file.startswith('PI7'):
        filepath = os.path.join(data_folder, file)
        print(f"讀取：{file}")
        df = pd.read_csv(filepath, usecols=['Datetime', 'DRUM_LEVEL'])
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')
        df['DRUM_LEVEL'] = pd.to_numeric(df['DRUM_LEVEL'], errors='coerce')
        df = df.dropna().sort_values('Datetime')
        df_all.append(df)

# 合併所有資料
df = pd.concat(df_all).sort_values('Datetime').reset_index(drop=True)
df = df.set_index('Datetime')
df = df.fillna(method='ffill').fillna(method='bfill')

# === 移動標準差計算 ===
rolling_std = df['DRUM_LEVEL'].rolling('60min').std()
threshold = rolling_std.mean() + rolling_std.std() * 1  # 1倍標準差
df['high_var'] = rolling_std > threshold

# === 標記異常段（連續為 True 時視為一段）===
df['label'] = (df['high_var'] != df['high_var'].shift()).cumsum()
grouped = df[df['high_var']].groupby('label')

# === 儲存異常區段（時間+最大變化）===
results = []
for _, group in grouped:
    start = group.index.min()
    end = group.index.max()
    peak_std = rolling_std.loc[start:end].max()
    duration = (end - start).total_seconds() / 60  # 分鐘
    if duration >= 30:  # 過濾太短的雜訊段
        results.append({
            '開始時間': start,
            '結束時間': end,
            '持續時間（分鐘）': round(duration),
            '最大標準差': round(peak_std, 2)
        })

# 轉成 DataFrame 顯示
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('開始時間').reset_index(drop=True)

# 儲存輸出成 Excel 或 CSV
output_csv = os.path.join(data_folder, "疑似破管區段列表.csv")
df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ 疑似破管時段已儲存為：{output_csv}")
