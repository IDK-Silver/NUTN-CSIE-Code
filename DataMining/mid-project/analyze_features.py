"""特徵重要性分析腳本

分析所有特徵與 traffic_volume 的相關性，用於指導特徵選擇。
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path


def analyze_feature_importance(train_path: str = "blob/process/train_processed.csv",
                               output_path: str = "blob/analysis/feature_importance.json"):
    """分析特徵重要性並輸出報告

    Args:
        train_path: 訓練資料路徑
        output_path: 輸出 JSON 路徑
    """
    print("=" * 80)
    print("📊 特徵重要性分析")
    print("=" * 80)
    print()

    # 讀取訓練資料
    df = pd.read_csv(train_path)
    print(f"✓ 載入資料: {df.shape}")
    print()

    # 分離特徵和目標
    X = df.drop(columns=['ID', 'traffic_volume'])
    y = df['traffic_volume']

    # 計算每個特徵與目標的相關性
    correlations = []
    constant_features = []

    for col in X.columns:
        # 檢查是否為常數特徵
        if X[col].std() == 0:
            constant_features.append(col)
            continue

        corr = X[col].corr(y)

        # 跳過 nan 值
        if pd.isna(corr):
            continue

        abs_corr = abs(corr)
        correlations.append({
            'feature': col,
            'correlation': float(corr),
            'abs_correlation': float(abs_corr)
        })

    if constant_features:
        print(f"⚠️ 發現 {len(constant_features)} 個常數特徵（已跳過）:")
        for feat in constant_features[:5]:
            print(f"   - {feat}")
        if len(constant_features) > 5:
            print(f"   ... 還有 {len(constant_features) - 5} 個")
        print()

    # 按絕對相關性排序
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # 顯示前 20 個最重要的特徵
    print("### 前 20 個最相關特徵（按絕對相關性）")
    print()
    print(f"{'排名':<6} {'特徵名稱':<35} {'相關性':>12} {'絕對值':>12}")
    print("-" * 70)

    for i, item in enumerate(correlations[:20], 1):
        feature = item['feature']
        corr = item['correlation']
        abs_corr = item['abs_correlation']

        # 標記不同類型的特徵
        if feature.startswith('rush_'):
            marker = "🔥"
        elif feature.startswith('temp_'):
            marker = "🌡️"
        elif feature.startswith('weather_'):
            marker = "☁️"
        elif 'hour' in feature:
            marker = "⏰"
        else:
            marker = "  "

        print(f"{i:<6} {marker} {feature:<33} {corr:>12.6f} {abs_corr:>12.6f}")

    print()
    print("-" * 70)
    print()

    # 分類統計
    print("### 特徵類別統計")
    print()

    categories = {
        'rush_': '交互作用（Rush Hour）',
        'temp_': '溫度相關',
        'weather_': '天氣相關',
        'hour': '時間相關',
        'base': '基本特徵'
    }

    for prefix, name in categories.items():
        if prefix == 'base':
            # 基本特徵：不含任何前綴
            features = [c for c in correlations
                       if not any(c['feature'].startswith(p) for p in ['rush_', 'temp_', 'weather_'])
                       and 'hour' not in c['feature']]
        else:
            features = [c for c in correlations if c['feature'].startswith(prefix) or prefix in c['feature']]

        if features:
            avg_corr = np.mean([f['abs_correlation'] for f in features])
            max_corr = max([f['abs_correlation'] for f in features])
            print(f"{name:30s}: {len(features):2d} 個特徵, 平均相關性 {avg_corr:.4f}, 最大 {max_corr:.4f}")

    print()
    print("-" * 70)
    print()

    # 推薦的 top-k 特徵
    print("### 推薦的特徵選擇")
    print()

    for k in [5, 10, 15, 20]:
        top_features = [c['feature'] for c in correlations[:k]]
        avg_corr = np.mean([c['abs_correlation'] for c in correlations[:k]])
        print(f"Top-{k:2d}: 平均相關性 {avg_corr:.4f}")
        if k == 15:
            print(f"        特徵: {', '.join(top_features[:5])}...")

    print()
    print("-" * 70)
    print()

    # 智能特徵選擇建議（考慮多項式回歸）
    print("### 💡 針對 Polynomial Regression 的智能選擇建議")
    print()
    print("⚠️  注意：單變量相關性不等於多項式回歸的特徵重要性！")
    print()

    # 過濾掉多重共線的溫度特徵
    smart_selection = []
    seen_temp_poly = False

    for c in correlations:
        feat = c['feature']

        # 避免同時選 temp, temp_squared, temp_cubed（多重共線性）
        if feat in ['temp_squared', 'temp_cubed']:
            if not seen_temp_poly:
                seen_temp_poly = True
            else:
                continue  # 跳過重複的溫度多項式

        # 必須包含的時間循環特徵（即使相關性低）
        if feat in ['hour_sin', 'hour_cos']:
            # 提升優先級
            smart_selection.insert(0, c)
            continue

        # 必須包含的基礎特徵
        if feat in ['Rush Hour', 'temp', 'clouds_all', 'is_holiday']:
            smart_selection.insert(0, c)
            continue

        smart_selection.append(c)

    print("推薦 Top-15（智能過濾，適合 Polynomial Regression）:")
    smart_top15 = [c['feature'] for c in smart_selection[:15]]
    for i, feat in enumerate(smart_top15, 1):
        corr = next((c['correlation'] for c in correlations if c['feature'] == feat), 0.0)
        print(f"  {i:2d}. {feat:35s} ({corr:>7.4f})")

    print()
    print("-" * 70)
    print()

    # 低相關性特徵（可能是雜訊）
    low_corr_features = [c for c in correlations if c['abs_correlation'] < 0.01]
    print(f"### 低相關性特徵（|相關性| < 0.01）：{len(low_corr_features)} 個")
    if low_corr_features:
        print("這些特徵對預測幫助不大，可考慮移除：")
        for item in low_corr_features[:10]:
            print(f"  - {item['feature']}: {item['correlation']:.6f}")
        if len(low_corr_features) > 10:
            print(f"  ... 還有 {len(low_corr_features) - 10} 個")

    print()
    print("=" * 80)

    # 儲存完整結果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'total_features': len(correlations),
        'analysis_date': pd.Timestamp.now().isoformat(),
        'correlations': correlations,
        'recommendations': {
            'top_5': [c['feature'] for c in correlations[:5]],
            'top_10': [c['feature'] for c in correlations[:10]],
            'top_15': [c['feature'] for c in correlations[:15]],
            'top_20': [c['feature'] for c in correlations[:20]],
            'smart_15': smart_top15,  # 智能選擇（考慮多項式回歸）
        },
        'low_correlation': [c['feature'] for c in low_corr_features]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 完整分析結果已儲存: {output_path}")
    print()

    return correlations


if __name__ == "__main__":
    analyze_feature_importance()
