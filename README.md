# 階層ベイズモデルを用いた商品評価予測システム

## 概要

階層ベイズモデルを学習するために作成しました。

少数のレビューしかない商品の真の評価値を予測することで、ECサイトにおけるより信頼性の高いレビューを取得することを目的としています。

また、以下のkaggleのデータセットを使用しています
[rammar-and-online-product-reviews](https://www.kaggle.com/datasets/datafiniti/grammar-and-online-product-reviews)

## プロジェクトの目的

- 階層ベイズモデルの理論と実装の理解
- 少ないデータから信頼性の高い予測を行う手法の学習
- PyMCを使ったベイズ統計モデリングの実践

## 主な特徴

- **階層構造**: カテゴリレベル → 個別商品レベルの2層階層
- **ベータ二項分布**: 1-5星評価の適切なモデリング
- **MCMC sampling**: 事後分布の推定
- **信頼区間**: 予測の不確実性を定量化

## システム構成

### 主要ファイル

- `要求定義.md` - プロジェクト要件の詳細仕様
- `計画書.md` - 階層ベイズアプローチの詳細計画
- `MLBayes_test.ipynb` - 基本的な実装例（複雑さの基準）
- `product_rating_prediction.ipynb` - メインの実装ノートブック
- `product_analysis_visualizer.py` - 結果可視化ツール

### データファイル

- `GrammarandProductReviews.csv` - サンプルレビューデータ
- `product_performance_detailed_results.csv` - 予測結果

### 出力結果

- `analysis_output/` - 分析結果の画像とサマリー
  - `analysis_summary.txt` - 分析結果のテキスト要約
  - `*.png` - 各種可視化グラフ

## 技術スタック

- **Python 3.8+** - メイン言語
- **PyMC** - ベイズ統計モデリング
- **NumPy, Pandas** - データ処理
- **Matplotlib, ArviZ** - 可視化
- **Jupyter Notebook** - 開発・実験環境

## モデル仕様

### 階層構造
```
全体レベル → カテゴリレベル → 個別商品レベル
```

### 対象データ
- **予測対象**: レビュー数10件以下の商品
- **評価スケール**: 1-5星評価
- **カテゴリ**: 家電、書籍、アパレル等

### 性能目標
- **精度**: MAE ≤ 0.8（レビュー10件以下の商品）
- **処理時間**: 単一商品予測 < 1秒、バッチ処理(1000商品) < 10分
- **学習時間**: モデル訓練 < 10分（デモ用途）

## 使い方

### 1. 環境構築
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install pymc numpy pandas matplotlib arviz jupyter
```

### 2. ノートブック実行
```bash
jupyter notebook
```

### 3. 主要ノートブック
- `MLBayes_test.ipynb` - 基本概念の理解
- `product_rating_prediction.ipynb` - メイン実装
- `kaggle_categories_experiment.ipynb` - カテゴリ実験

## 予測結果の見方

### 予測評価の表示例
```
予測評価: ⭐4.5 (95%の確率で 4.1〜4.8 の範囲)
```

### 解釈
- レビューが少ない商品でも、同カテゴリの傾向を活用
- 縮退（Shrinkage）により過度な評価を抑制
- 信頼区間で予測の確からしさを表現

## 学習ポイント

1. **階層ベイズの概念**: 個別データをグループ全体の情報で補強
2. **縮退効果**: 不安定な個別推定値をグループ平均に近づける
3. **不確実性の定量化**: 予測だけでなく信頼度も提供
4. **実データへの適用**: 理論から実践的な問題解決へ

## 制約事項

- デモ・学習用途のため、商用レベルの最適化は未実施
- 季節性、レビュアー信頼度等の高度な要因は考慮外
- 処理対象は比較的少量のデータセット

## 作成資料
※これら資料は以下のプロンプトで作成しました
- 以下の問題設計と解決方法をもとに、どのように解決していくかをブラッシュアップしたいです。また、回答はmdファイルで出力してください 
 - 問題設定
 ECサイトを使用し欲しい商品のレビューが少なく購入する際、レビューを参考にしにくい。
 - 解決方法
 階層ベイズモデルを活用することで、カテゴリごとのレビューの傾向からレビューの少ない商品の正確なレビューを得ること

- @(mdファイルの指定)このファイルから要求定義を行ってください、そして不明な点があれば質問してください。


- `要求定義.md` - 詳細な機能・性能要件
- `計画書.md` - 階層ベイズモデルの理論背景
- `kaggle_experiment_explanation.md` - 実験方法の解説
