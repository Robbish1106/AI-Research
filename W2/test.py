import os
import pandas as pd
import matplotlib.pyplot as plt

# === 設定資料路徑（相對於你.py檔）===
data_folder = os.path.abspath(os.path.join("..", "資料", "資料集"))
filename = "PI72022_SDK_4monthDATA.csv"
filepath = os.path.join(data_folder, filename)

# === 讀取 CSV 資料 ===
df = pd.read_csv(filepath)
df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')

# === 指定破管期間與分析欄位 ===
start_all = '2022-04-24 00:00'
end_all   = '2022-04-27 23:59'
start_event = '2022-04-25 10:30'
end_event   = '2022-04-27 18:21'

df_all = df[(df['Datetime'] >= start_all) & (df['Datetime'] <= end_all)]
df_event = df[(df['Datetime'] >= start_event) & (df['Datetime'] <= end_event)]

columns_to_plot = ['DRUM_LEVEL', 'LT_CD008_1', 'LT_CD008_2', 'FT_DW009', 'ZT_BC010']

# === 畫圖 ===
for col in columns_to_plot:
    plt.figure(figsize=(14, 5))
    plt.plot(df_all['Datetime'], df_all[col], label='正常期間', color='skyblue')
    plt.plot(df_event['Datetime'], df_event[col], label='破管期間', color='crimson')
    plt.title(f'{col} - Comparison Before/After Pipe Rupture')
    plt.legend(['Normal Period', 'Rupture Period'])
    plt.xlabel('Time')
    plt.ylabel(col)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{col}_accident_compare.png")
    plt.show()
