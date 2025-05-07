import os
import pandas as pd
import matplotlib.pyplot as plt

# === 設定資料夾路徑 ===
base_folder = os.path.abspath(os.path.join("..", "資料", "資料集"))
output_folder = os.path.abspath(os.path.join("..", "W3", "data", "PLATEN_SH_IN_HDR_P_1"))

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# === 定義需要處理的月份 ===
months = [f"PI72020_SDK_{i}monthDATA.csv" for i in range(6, 11)]  # 6月到10月

# === 設定統一的 Y 軸範圍 ===
y_min, y_max = 0, 30

# === 逐月處理 CSV 檔案 ===
for month_file in months:
    target_file = os.path.join(base_folder, month_file)
    
    if not os.path.exists(target_file):
        print(f"⚠️ 找不到檔案：{month_file}")
        continue
    
    # === 讀取CSV ===
    df = pd.read_csv(target_file, usecols=['Datetime', 'PLATEN_SH_IN_HDR_P_1'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(inplace=True)

    # === 依日期分組並繪製折線圖 ===
    df['Date'] = df['Datetime'].dt.date

    for date, group in df.groupby('Date'):
        plt.figure(figsize=(14, 6))
        plt.plot(group['Datetime'], group['PLATEN_SH_IN_HDR_P_1'], color='royalblue', linewidth=1)
        plt.title(f'PLATEN_SH_IN_HDR_P_1 on {date}')
        plt.xlabel('Time')
        plt.ylabel('PLATEN_SH_IN_HDR_P_1')
        plt.xticks(rotation=30)
        plt.grid(True)
        
        # 設定統一的Y軸範圍
        plt.ylim(y_min, y_max)

        # 保存圖片
        output_path = os.path.join(output_folder, f"{date}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已儲存 {output_path}")

print("所有圖片已儲存完成。")
