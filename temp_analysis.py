import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r"C:\Users\k\EC\product_performance_detailed_results.csv"

try:
    df = pd.read_csv(file_path)

    print("--- 1. 信頼区間カバレッジ (Coverage Probability) ---")
    df['is_covered'] = (df['actual_rating'] >= df['estimated_star_lower']) & (df['actual_rating'] <= df['estimated_star_upper'])
    overall_coverage = df['is_covered'].mean() * 100
    print(f"全体のカバレッジ: {overall_coverage:.1f}% の商品で、実際の評価が95%信頼区間内に収まりました。")
    low_review_df = df[df['num_reviews'] <= 10]
    low_review_coverage = low_review_df['is_covered'].mean() * 100
    print(f"レビュー10件以下の商品でのカバレッジ: {low_review_coverage:.1f}%")
    print("\n")

    print("--- 2. 予測信頼区間の幅の分析 ---")
    bins = [0, 3, 10, 50, np.inf]
    labels = ["1-3 reviews", "4-10 reviews", "11-50 reviews", "51+ reviews"]
    df["review_bin"] = pd.cut(df["num_reviews"], bins=bins, labels=labels, right=True)
    width_analysis = df.groupby("review_bin", observed=True)["prediction_width"].mean().reset_index()
    width_analysis.columns = ["レビュー数グループ", "平均の予測区間幅"]
    print("レビュー数に応じた平均予測区間幅:")
    print(width_analysis.to_string(index=False))
    print("\n")

    print("--- 3. ランク相関 (Spearman's Rank Correlation) ---")
    rank_corr = df[['actual_rating', 'estimated_star_rating']].corr(method='spearman').iloc[0, 1]
    print(f"実績評価と予測評価のスピアマン順位相関係数: {rank_corr:.3f}")
    print("（1に近いほど、ランキングの予測が正確であることを示します）")

except FileNotFoundError:
    print(f"エラー: {file_path} が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
