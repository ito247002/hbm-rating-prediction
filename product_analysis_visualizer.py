"""
商品パフォーマンス詳細分析結果の可視化ツール

このスクリプトはproduct_performance_detailed_results.csvを読み込み、
階層ベイズモデルの予測性能を包括的に分析・可視化します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class ProductAnalysisVisualizer:
    """商品分析結果の可視化クラス"""
    
    def __init__(self, csv_path: str):
        """
        初期化
        
        Args:
            csv_path (str): CSVファイルのパス
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """CSVデータの読み込みと前処理"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"データ読み込み完了: {len(self.df)}行のデータ")
            print(f"カラム: {list(self.df.columns)}")
            
            # 基本統計量の表示
            print("\n=== 基本統計量 ===")
            print(self.df[['actual_rating', 'abs_error', 'num_reviews', 'prediction_width']].describe())
            
            # カテゴリー別商品数
            print(f"\n=== カテゴリー別商品数 ===")
            print(self.df['category'].value_counts())
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            raise
    
    def plot_prediction_accuracy(self, figsize=(15, 10)):
        """予測精度に関するグラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('予測精度分析', fontsize=16, fontweight='bold')
        
        # 1. 実際値 vs 予測値（星評価）
        ax1 = axes[0, 0]
        ax1.scatter(self.df['actual_rating'], self.df['estimated_star_rating'], alpha=0.6, s=50)
        
        # 完璧な予測ライン
        min_val = min(self.df['actual_rating'].min(), self.df['estimated_star_rating'].min())
        max_val = max(self.df['actual_rating'].max(), self.df['estimated_star_rating'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完璧な予測')
        
        ax1.set_xlabel('実際の評価値')
        ax1.set_ylabel('予測評価値')
        ax1.set_title('実際値 vs 予測値')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 絶対誤差の分布
        ax2 = axes[0, 1]
        ax2.hist(self.df['abs_error'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(self.df['abs_error'].mean(), color='red', linestyle='--', 
                   label=f'平均誤差: {self.df["abs_error"].mean():.3f}')
        ax2.axvline(0.8, color='orange', linestyle='--', 
                   label='目標誤差: 0.8')
        ax2.set_xlabel('絶対誤差')
        ax2.set_ylabel('頻度')
        ax2.set_title('絶対誤差の分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 誤差 vs レビュー数
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.df['num_reviews'], self.df['abs_error'], 
                             c=self.df['prediction_width'], cmap='viridis', alpha=0.6, s=50)
        ax3.set_xlabel('レビュー数')
        ax3.set_ylabel('絶対誤差')
        ax3.set_title('レビュー数 vs 絶対誤差')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # カラーバー追加
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('予測区間の幅')
        
        # 4. 予測区間の幅の分布
        ax4 = axes[1, 1]
        ax4.hist(self.df['prediction_width'], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(self.df['prediction_width'].mean(), color='red', linestyle='--',
                   label=f'平均幅: {self.df["prediction_width"].mean():.3f}')
        ax4.set_xlabel('予測区間の幅')
        ax4.set_ylabel('頻度')
        ax4.set_title('予測不確実性の分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_review_analysis(self, figsize=(12, 8)):
        """レビュー数による分析"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('レビュー数による性能分析', fontsize=16, fontweight='bold')
        
        # レビュー数のビニング
        bins = [0, 1, 5, 10, 50, 100, float('inf')]
        labels = ['1', '2-5', '6-10', '11-50', '51-100', '100+']
        self.df['review_bin'] = pd.cut(self.df['num_reviews'], bins=bins, labels=labels, right=False)
        
        # 1. レビュー数分布
        ax1 = axes[0, 0]
        review_counts = self.df['review_bin'].value_counts().sort_index()
        ax1.bar(range(len(review_counts)), review_counts.values)
        ax1.set_xticks(range(len(review_counts)))
        ax1.set_xticklabels(review_counts.index, rotation=45)
        ax1.set_xlabel('レビュー数グループ')
        ax1.set_ylabel('商品数')
        ax1.set_title('レビュー数分布')
        ax1.grid(True, alpha=0.3)
        
        # 2. レビュー数グループ別平均誤差
        ax2 = axes[0, 1]
        avg_error = self.df.groupby('review_bin')['abs_error'].mean().sort_index()
        ax2.bar(range(len(avg_error)), avg_error.values)
        ax2.set_xticks(range(len(avg_error)))
        ax2.set_xticklabels(avg_error.index, rotation=45)
        ax2.set_xlabel('レビュー数グループ')
        ax2.set_ylabel('平均絶対誤差')
        ax2.set_title('レビュー数グループ別平均誤差')
        ax2.axhline(y=0.8, color='red', linestyle='--', label='目標誤差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. レビュー数 vs 予測区間幅
        ax3 = axes[1, 0]
        ax3.scatter(self.df['num_reviews'], self.df['prediction_width'], alpha=0.6, s=50)
        ax3.set_xlabel('レビュー数')
        ax3.set_ylabel('予測区間の幅')
        ax3.set_title('レビュー数 vs 予測不確実性')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. レビュー数グループ別予測区間幅
        ax4 = axes[1, 1]
        avg_width = self.df.groupby('review_bin')['prediction_width'].mean().sort_index()
        ax4.bar(range(len(avg_width)), avg_width.values)
        ax4.set_xticks(range(len(avg_width)))
        ax4.set_xticklabels(avg_width.index, rotation=45)
        ax4.set_xlabel('レビュー数グループ')
        ax4.set_ylabel('平均予測区間幅')
        ax4.set_title('レビュー数グループ別不確実性')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_category_analysis(self, figsize=(15, 12)):
        """カテゴリー別分析"""
        # 商品数が多い上位10カテゴリーを選択
        top_categories = self.df['category'].value_counts().head(10).index
        df_top = self.df[self.df['category'].isin(top_categories)]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('カテゴリー別性能分析（上位10カテゴリー）', fontsize=16, fontweight='bold')
        
        # 1. カテゴリー別商品数
        ax1 = axes[0, 0]
        cat_counts = df_top['category'].value_counts()
        ax1.bar(range(len(cat_counts)), cat_counts.values)
        ax1.set_xticks(range(len(cat_counts)))
        ax1.set_xticklabels(cat_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('商品数')
        ax1.set_title('カテゴリー別商品数')
        ax1.grid(True, alpha=0.3)
        
        # 2. カテゴリー別平均誤差
        ax2 = axes[0, 1]
        cat_error = df_top.groupby('category')['abs_error'].mean().sort_values(ascending=False)
        bars = ax2.bar(range(len(cat_error)), cat_error.values)
        ax2.set_xticks(range(len(cat_error)))
        ax2.set_xticklabels(cat_error.index, rotation=45, ha='right')
        ax2.set_ylabel('平均絶対誤差')
        ax2.set_title('カテゴリー別平均誤差')
        ax2.axhline(y=0.8, color='red', linestyle='--', label='目標誤差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 誤差が高いカテゴリーを強調
        for i, (bar, error) in enumerate(zip(bars, cat_error.values)):
            if error > 3.0:
                bar.set_color('red')
            elif error > 2.0:
                bar.set_color('orange')
        
        # 3. カテゴリー別予測区間幅
        ax3 = axes[1, 0]
        cat_width = df_top.groupby('category')['prediction_width'].mean().sort_values(ascending=False)
        ax3.bar(range(len(cat_width)), cat_width.values)
        ax3.set_xticks(range(len(cat_width)))
        ax3.set_xticklabels(cat_width.index, rotation=45, ha='right')
        ax3.set_ylabel('平均予測区間幅')
        ax3.set_title('カテゴリー別予測不確実性')
        ax3.grid(True, alpha=0.3)
        
        # 4. カテゴリー別誤差分布（ボックスプロット）
        ax4 = axes[1, 1]
        category_order = cat_error.index[:8]  # 上位8カテゴリー
        df_plot = df_top[df_top['category'].isin(category_order)]
        
        sns.boxplot(data=df_plot, x='category', y='abs_error', ax=ax4, order=category_order)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_title('カテゴリー別誤差分布')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_analysis(self, figsize=(12, 10)):
        """不確実性分析"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('予測不確実性の詳細分析', fontsize=16, fontweight='bold')
        
        # 1. eta値の信頼区間
        ax1 = axes[0, 0]
        # サンプリング（可視化のため）
        sample_idx = np.random.choice(len(self.df), min(100, len(self.df)), replace=False)
        df_sample = self.df.iloc[sample_idx].copy()
        df_sample = df_sample.sort_values('predicted_eta_mean')
        
        x = range(len(df_sample))
        ax1.errorbar(x, df_sample['predicted_eta_mean'], 
                    yerr=[df_sample['predicted_eta_mean'] - df_sample['predicted_eta_hdi_lower'],
                          df_sample['predicted_eta_hdi_upper'] - df_sample['predicted_eta_mean']],
                    fmt='o', alpha=0.6, capsize=3, capthick=1)
        ax1.set_xlabel('商品インデックス（予測eta値順）')
        ax1.set_ylabel('予測eta値')
        ax1.set_title('eta値の予測信頼区間（サンプル100商品）')
        ax1.grid(True, alpha=0.3)
        
        # 2. 星評価の信頼区間
        ax2 = axes[0, 1]
        ax2.errorbar(x, df_sample['estimated_star_rating'],
                    yerr=[df_sample['estimated_star_rating'] - df_sample['estimated_star_lower'],
                          df_sample['estimated_star_upper'] - df_sample['estimated_star_rating']],
                    fmt='o', alpha=0.6, capsize=3, capthick=1)
        ax2.set_xlabel('商品インデックス（予測eta値順）')
        ax2.set_ylabel('予測星評価')
        ax2.set_title('星評価の予測信頼区間（サンプル100商品）')
        ax2.grid(True, alpha=0.3)
        
        # 3. 予測区間幅 vs 誤差の関係
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.df['prediction_width'], self.df['abs_error'], 
                             c=self.df['num_reviews'], cmap='plasma', alpha=0.6, s=50)
        ax3.set_xlabel('予測区間の幅')
        ax3.set_ylabel('絶対誤差')
        ax3.set_title('予測不確実性 vs 実際誤差')
        ax3.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('レビュー数')
        
        # 4. 標準偏差の分布
        ax4 = axes[1, 1]
        ax4.hist(self.df['predicted_eta_sd'], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(self.df['predicted_eta_sd'].mean(), color='red', linestyle='--',
                   label=f'平均標準偏差: {self.df["predicted_eta_sd"].mean():.3f}')
        ax4.set_xlabel('予測eta値の標準偏差')
        ax4.set_ylabel('頻度')
        ax4.set_title('予測不確実性（標準偏差）の分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """分析結果のサマリーレポートを生成"""
        report = []
        report.append("=== 商品パフォーマンス分析サマリーレポート ===\n")
        
        # 基本統計
        total_products = len(self.df)
        mean_error = self.df['abs_error'].mean()
        median_error = self.df['abs_error'].median()
        target_achieved = (self.df['abs_error'] <= 0.8).sum()
        target_rate = target_achieved / total_products * 100
        
        report.append(f"総商品数: {total_products}")
        report.append(f"平均絶対誤差: {mean_error:.3f}")
        report.append(f"誤差中央値: {median_error:.3f}")
        report.append(f"目標達成商品数: {target_achieved} ({target_rate:.1f}%)")
        report.append("")
        
        # レビュー数分析
        low_review = (self.df['num_reviews'] <= 5).sum()
        low_review_error = self.df[self.df['num_reviews'] <= 5]['abs_error'].mean()
        high_review = (self.df['num_reviews'] > 50).sum()
        high_review_error = self.df[self.df['num_reviews'] > 50]['abs_error'].mean()
        
        report.append("=== レビュー数による分析 ===")
        report.append(f"少数レビュー商品（≤5件）: {low_review}商品, 平均誤差: {low_review_error:.3f}")
        report.append(f"多数レビュー商品（>50件）: {high_review}商品, 平均誤差: {high_review_error:.3f}")
        report.append("")
        
        # カテゴリー分析
        worst_category = self.df.groupby('category')['abs_error'].mean().sort_values(ascending=False).head(1)
        best_category = self.df.groupby('category')['abs_error'].mean().sort_values(ascending=True).head(1)
        
        report.append("=== カテゴリー分析 ===")
        report.append(f"最も困難なカテゴリー: {worst_category.index[0]} (平均誤差: {worst_category.values[0]:.3f})")
        report.append(f"最も予測しやすいカテゴリー: {best_category.index[0]} (平均誤差: {best_category.values[0]:.3f})")
        report.append("")
        
        # 不確実性分析
        mean_width = self.df['prediction_width'].mean()
        correlation = np.corrcoef(self.df['prediction_width'], self.df['abs_error'])[0, 1]
        
        report.append("=== 不確実性分析 ===")
        report.append(f"平均予測区間幅: {mean_width:.3f}")
        report.append(f"予測区間幅と実誤差の相関: {correlation:.3f}")
        report.append("")
        
        return "\n".join(report)
    
    def create_comprehensive_analysis(self, save_plots=True, output_dir="analysis_output"):
        """包括的な分析を実行し、すべてのグラフを生成"""
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        print("商品パフォーマンス分析を開始...")
        
        # 各種グラフの生成
        fig1 = self.plot_prediction_accuracy()
        if save_plots:
            fig1.savefig(f"{output_dir}/prediction_accuracy.png", dpi=300, bbox_inches='tight')
        
        fig2 = self.plot_review_analysis()
        if save_plots:
            fig2.savefig(f"{output_dir}/review_analysis.png", dpi=300, bbox_inches='tight')
        
        fig3 = self.plot_category_analysis()
        if save_plots:
            fig3.savefig(f"{output_dir}/category_analysis.png", dpi=300, bbox_inches='tight')
        
        fig4 = self.plot_uncertainty_analysis()
        if save_plots:
            fig4.savefig(f"{output_dir}/uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        
        # サマリーレポート生成
        summary = self.generate_summary_report()
        print(summary)
        
        if save_plots:
            with open(f"{output_dir}/analysis_summary.txt", "w", encoding='utf-8') as f:
                f.write(summary)
        
        print(f"\n分析完了! グラフは {output_dir} フォルダに保存されました。")
        
        return {
            'prediction_accuracy': fig1,
            'review_analysis': fig2, 
            'category_analysis': fig3,
            'uncertainty_analysis': fig4,
            'summary': summary
        }


def main():
    """メイン実行関数"""
    csv_file = "product_performance_detailed_results.csv"
    
    try:
        # 分析器の初期化
        analyzer = ProductAnalysisVisualizer(csv_file)
        
        # 包括的分析の実行
        results = analyzer.create_comprehensive_analysis(save_plots=True)
        
        # グラフを表示
        plt.show()
        
    except FileNotFoundError:
        print(f"エラー: {csv_file} が見つかりません。")
        print("ファイルパスを確認してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()