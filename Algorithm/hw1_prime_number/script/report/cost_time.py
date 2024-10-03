import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager 
matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
plt.rc('font', family='Taipei Sans TC Beta')
plt.rcParams['axes.unicode_minus'] = False

# 讀取 algorithm_params.txt
params = {}
with open('result/all_algorithm/algorithm_params.txt', 'r', encoding='utf-8') as file:
    for line in file:
        key_value = line.strip().split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            params[key] = value

print("算法參數設置：", params)

# 讀取 cost_time.csv
df = pd.read_csv('result/all_algorithm/cost_time.csv')

print(df)

# 繪製圖表
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='n', y='time_ms', hue='算法')

plt.title('各算法執行時間比較')
plt.xlabel('n 值')
plt.ylabel('執行時間 (毫秒)')
plt.legend(title='算法')

# 設置對數刻度
plt.xscale('log')
plt.yscale('log')

# 保存圖表
plt.savefig('result/all_algorithm/cost_time_comparison.png')
plt.close()

print("圖表已保存為 'result/all_algorithm/cost_time_comparison.png'")
