import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.font_manager 
matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
plt.rc('font', family='Taipei Sans TC Beta')

plt.rcParams['axes.unicode_minus'] = False

# 設定結果目錄
result_dir = 'result/fermat_vs_miller'

# 函數：讀取結果文件
def read_result_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('算法, try_time'):
            header_index = i
            break

    if header_index is None:
        print(f"無法找到表頭於文件: {file_path}")
        return None

    return pd.read_csv(file_path, skiprows=header_index, skipinitialspace=True)

# 函數：繪製並保存圖表
def plot_and_save(data, n, output_dir):
    plt.figure(figsize=(12, 6))
    for algo in data['算法'].unique():
        algo_data = data[data['算法'] == algo]
        plt.plot(algo_data['try_time'], algo_data['accuracy(%)'], label=algo, marker='o')
    
    plt.xlabel('嘗試次數')
    plt.ylabel('正確率 (%)')
    plt.title(f'不同嘗試次數下 Fermat 和 Miller-Rabin 算法的準確率比較 (n={n})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 設置 x 軸的範圍和刻度
    plt.xlim(1, data['try_time'].max())
    plt.xticks(range(1, data['try_time'].max() + 1, max(1, data['try_time'].max() // 10)))
    
    output_path = os.path.join(output_dir, f'n_{n}_accuracy_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"圖表已保存: {output_path}")

# 主程序
def main():
    # 創建輸出目錄
    output_dir = os.path.join(result_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # 讀取並處理所有結果文件
    for filename in os.listdir(result_dir):
        if filename.startswith('n_') and filename.endswith('_results.txt'):
            file_path = os.path.join(result_dir, filename)
            n = int(re.search(r'n_(\d+)_results', filename).group(1))
            
            data = read_result_file(file_path)
            if data is not None:
                plot_and_save(data, n, output_dir)
        
        elif filename.startswith('basic_') and filename.endswith('.txt'):
            file_path = os.path.join(result_dir, filename)
            n = int(re.search(r'basic_(\d+)', filename).group(1))
            
            # 讀取 basic 文件內容
            with open(file_path, 'r', encoding='utf-8') as f:
                primes = [int(line.strip()) for line in f if line.strip()]
            
            # 繪製質數分佈圖
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(primes) + 1), primes, marker='o', linestyle='')
            plt.xlabel('質數序號')
            plt.ylabel('質數值')
            plt.title(f'質數分佈 (n={n})')
            plt.grid(True)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'basic_{n}_prime_distribution.png')
            plt.savefig(output_path)
            plt.close()
            print(f"基本質數分佈圖已保存: {output_path}")

if __name__ == "__main__":
    main()