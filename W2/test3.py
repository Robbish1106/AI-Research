import os
import pandas as pd
import matplotlib.pyplot as plt

# === 設定資料路徑（相對於你.py檔）===
data_folder = os.path.abspath(os.path.join("..", "資料", "資料集"))
filename = "PI72021_SDK_3monthDATA.csv"  # ← 換成2021/03月的資料
filepath = os.path.join(data_folder, filename)

# === 讀取 CSV 資料 ===
df = pd.read_csv(filepath)
df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='coerce')

# === 指定推測破管期間與分析欄位 ===
start_all = '2021-03-17 00:00'  # 前一天也包進去觀察
end_all   = '2021-03-20 00:00'
start_event = '2021-03-18 00:00'
end_event   = '2021-03-18 23:59'

df_all = df[(df['Datetime'] >= start_all) & (df['Datetime'] <= end_all)]
df_event = df[(df['Datetime'] >= start_event) & (df['Datetime'] <= end_event)]

columns_to_plot = ['DRUM_LEVEL', 'LT_CD008_1', 'LT_CD008_2', 'FT_DW009', 'ZT_BC010']

# === 畫圖 ===
for col in columns_to_plot:
    plt.figure(figsize=(14, 5))
    plt.plot(df_all['Datetime'], df_all[col], label='正常期間', color='skyblue')
    plt.plot(df_event['Datetime'], df_event[col], label='推測破管期間', color='crimson')
    plt.title(f'{col} - Possible Pipe Rupture (2021/03/18)')
    plt.legend(['Normal Period', 'Rupture Period'])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{col}_rupture_20210318.png")
    plt.show()
