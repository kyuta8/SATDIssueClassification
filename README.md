# SATDIssueClassification

## 使い方

### データセット
dataフォルダに含まれる各プロジェクトのフォルダ直下に存在するzipファイルを解凍

### モデルの評価
モデルの評価スクリプト：`EvaluateSingleProjectModel.py`
```
optional arguments:
  -h, --help            show this help message and exit
  -n {1,2,n,embeded}, --ngram {1,2,n,embeded} 学習時のN-gramを指定
  -m MODEL, --model MODEL
                        Select Machine Learning Algorithm
  -t, --text            Use Text Metrics
  -r, --reporter        Use Reporter Metrics
  -p, --process         Use Process Metrics
  -c, --code            Use Source Code Metrics
  -s SAMPLING, --sampling SAMPLING
                        Select Imbalanced Data Processing
  -tf, --tfidf          Use tf-idf
  --drop                Drop Data with Outliers
  --project PROJECT     Select Project
```

`single.bat`を実行すると全てのプロジェクトでの評価が実行される

## 論文
https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&page_id=13&active_action=repository_view_main_item_detail&item_id=223512&item_no=1&block_id=8

```
@article{木村2023-1,
   author	 = "木村,祐太 and 大平,雅雄",
   title	 = "技術的負債に関連する課題票分類手法の構築",
   journal	 = "情報処理学会論文誌",
   year 	 = "2023",
   volume	 = "64",
   number	 = "1",
   pages	 = "2--12",
   month	 = "jan"
}
```
