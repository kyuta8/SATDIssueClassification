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
