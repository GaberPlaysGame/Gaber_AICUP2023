## Improvement
- Version 1: Redirect Wikipedia Searching
- Version 2: TFIDF + Wikipedia Searching
- Version 3: Neg_ratio Adjusting, TFIDF Adjusting

### Document Retrieval
原本結果為：
- Precision: 0.2525661209068016
- Recall: 0.8066447523089846

訓練了136分鐘，在採用 Pandarallel 的情況下。

| Experiment  | Precision | Recall | F1-Score | Filename |
| ----------- | --------- | ------ | -------- | -------- |
| Original                                       | 0.250937 | 0.807333 | 0.382870 | **train_doc5_aicup.jsonl**                     |
| On Only Wiki-Search                        | 0.017607 | 0.907427 | 0.034544 | **train_doc5_search.jsonl**                    |
| Tf-Idf On Search Data                      | 0.330172 | 0.555870 | 0.414275 | **train_doc5_tfidf_1.jsonl**                   |
| Merge Tf-Idf and AICUP                     | 0.197161 | 0.899203 | 0.323410 | **train_doc5_new_2.jsonl**                     |
| w2v + Tf-Idf On Search Data, thres=0.65    | 0.143234 | 0.635255 | 0.233761 | **train_doc5_tfidf_3.jsonl**                   |
| Merge, On TFIDF Ver. 3                     | 0.125820 | 0.902380 | 0.220848 | **train_doc5_new_4.jsonl**                     |
| w2v + Tf-Idf, thres=0.75, wv=300           | 0.158859 | 0.579969 | 0.249404 | **train_doc5_tfidf_4.jsonl**                   |
| Merge, On TFIDF Ver. 4                     | 0.153136 | 0.900349 | 0.261752 | **train_doc5_new_5.jsonl**                     |
| w2v + Tf-Idf, thres=0.8, wv=300            | 0.176479 | 0.478771 | 0.257896 | **train_doc5_tfidf_5.jsonl**                   |
| Merge, On TFIDF Ver. 5                     | 0.186085 | 0.896547 | 0.308200 | **train_doc5_new_6.jsonl**                     |
| Sentence BERT 加權                | 0.484753 | 0.884544 | 0.626286 | **train_doc5_sbert.jsonl**                     |

還需要改善的點：
1. 需要 Hyperlink，有些沒有直接提及。
2. 初設定取值在find值越前面的地方，但是有時候主詞在後面，會被濾掉
    - 目前計畫採用直接搜尋方式，如果說有完整對到的會列在比較前面，其餘搜尋結果無條件靠後
    - 現在改用重複提及，如果有重複被搜尋到的就會被放到list內。
3. 針對引號等等，不能分割搜尋

### Sentence Retrieval
利用原本訓練出來的model50, model100與model150下去做validation。

用我的GPU不知道為甚麼要跑到10小時，Batch Size也只接受到32。改成Colab後Batch Size到64，只有一個model50，執行時間只要6分鐘。

原本的成績：
| Model | Train-F1 | Train-P | Train-R | Valid-F1 | Valid-P | Valid-R |
| ----- | -------- | ------- | ------- | -------- | ------- | ------- |
| model50-0-64         | 0.357773 | 0.237802 | 0.722047 | 0.333525 | 0.216876 | 0.721698 |
| model50-1-32         | 0.398642 | 0.267014 | 0.786220 | 0.364270 | 0.237342 | 0.783018 |
| model100-1-32        | 0.403016 | 0.269849 | 0.795669 | 0.365551 | 0.238286 | 0.784591 |
| model150-1-32        | 0.402539 | 0.269376 | 0.796062 | 0.364640 | 0.237657 | 0.783018 |
| model30-1-64         | 0.373679 | 0.249173 | 0.746875 | 0.370676 | 0.246171 | 0.750000 |
| model50-1-64         | 0.389274 | 0.259563 | 0.778125 | 0.388977 | 0.256796 | 0.801563 |
| model60-1-64         | 0.384763 | 0.256595 | 0.768750 | 0.383362 | 0.253671 | 0.784375 |
| model90-1-64         | 0.387116 | 0.258079 | 0.774218 | 0.387164 | 0.255859 | 0.795312 |
| model50-2-64         | 0.399022 | 0.266451 | 0.794140 | 0.394697 | 0.260182 | 0.817187 |
| model50-3-64-neg0.05  | 0.394725 | 0.264029 | 0.781641 | 0.387462 | 0.256119 | 0.795313 |
| model50-3-64-neg0.01  | 0.399061 | 0.266530 | 0.793750 | 0.390361 | 0.257682 | 0.804688 |
| model100-4-64-neg0.05 | 0.395390 | 0.263828 | 0.788672 | 0.380936 | 0.251875 | 0.781250 |
| model200-6-32-neg0.05 | 0.403486 | 0.269115 | 0.805859 | 0.385430 | 0.255156 | 0.787500 |
| model50-6-32-neg0.01  | 0.395786 | 0.264401 | 0.786718 | 0.381569 | 0.252266 | 0.782813 |
| model100-6-32-neg0.01 | 0.398201 | 0.265807 | 0.793359 | 0.384270 | 0.254141 | 0.787500 |
| model150-7-32-neg0.01_sent5  | 0.414659 | 0.281621 | 0.785937 | 0.394098 | 0.262291 | 0.792187 |
| model150-7-32-neg0.025_sent5 | 0.414759 | 0.282167 | 0.782421 | 0.394803 | 0.262916 | 0.792187 |
| model150-7-32-neg0.05_sent5  | 0.417682 | 0.283808 | 0.790625 | 0.394608 | 0.262916 | 0.790625 |
| model200-7-32-neg0.075_sent5_1e-5 | 0.417821 | 0.283886 | 0.791015 | 0.395859 | 0.263854 | 0.792187 |
| model200-7-32-neg0.075_sent5_2e-5 | 0.420307 | 0.285527 | 0.796093 | 0.397340 | 0.264479 | 0.798437 |
| model200-7-32-neg0.075_sent5_3e-5 | 0.419447 | 0.285136 | 0.792968 | 0.398239 | 0.265104 | 0.800000 |
| model200-7-32-neg0.075_sent5_4e-5_e1_base    | 0.419925 | 0.285527 | 0.793359 | 0.400776 | 0.266666 | 0.806250 |
| model400-7-32-neg0.075_sent5_4e-5_e2_base    | 0.420113 | 0.285449 | 0.795312 | 0.398239 | 0.265104 | 0.800000 |
| model250-7-32-neg0.075_sent5_4e-5_e1_hflbert | 0.420342 | 0.285761 | 0.794531 | 0.400230 | 0.266354 | 0.804687 |
| model250-7-32-neg0.075_sent5_3.5e-5_e1_hflbert | 0.420874 | 0.286152 | 0.795312 | 0.400423 | 0.266354 | 0.806250 |

### Claim Validation
CUDA 在 Colab 上還是容易爆 CudaOutOfMemoryError，我的GPU只能容許Batch Size=8，Colab只能24。

原本的成績與後續對照：
| Model | Val_acc | Train_step | Version | Batch Size | Base Model |
| ----- | ------- | ---------- | ------- | ---------- | ---------- |
| model.7900 | 0.4056 | 7900 | 0  | 8  | model50-0-32 |
| model.950  | 0.4170 | 950  | 0  | 8  | model50-0-32 |
| model.750  | 0.4259 | 750  | 0  | 32 | model50-0-64 |
| model.100  | 0.4471 | 100  | 1  | 32 | model90-1-64 |
| model.425  | 0.4471 | 425  | 1  | 32 | model90-1-64 |
| model.250  | 0.4559 | 250  | 1  | 32 | model90-1-64 |

### Submission History
| Model | Val_acc | Public Score | Private Score | Base Model |
| ----- | ------- | ------------ | ------------- | ---------- |
| model.950  | 0.4170 | 0.303337 | NaN | model50-0-32  |
| model.750  | 0.4259 | 0.292214 | NaN | model50-0-32  |
| model.250  | 0.4559 | 0.299292 | NaN | model90-1-64  |
| model.1500 | 0.4395 | 0.301314 | NaN | model50-2-64  |
| model.1900 | 0.4358 | 0.302326 | NaN | model50-2-64  |
| model.700  | 0.4295 | 0.314459 | NaN | model50-2-64  |

## Module Testing
### HanLP
HanLP的constituent tree parsing 中，會自動把標點符號以及錯字轉換，如「曆史」變成「歷史」、引號「變成英文引號 "，範例不勝枚舉，這邊一律只能忽略。
- 因此導致後面的mapping[term] = claim.find(new_term)出錯，產生AssertionError

### Wikipedia-api
Wikipedia-api不能在Pandarallel內處理，會造成名為 JSONDecodeError 的 bug，因此判到頁面是否存在的方法改用 requests 下去進行。
- **JSOndecodeerror __init__() missing 2 required positional arguments: 'doc' and 'pos**

### Word2Vector
處理了W2V在向量250的訓練，接下來要看 Word Embedding。
- W2V 不能處理未知字庫，這代表斷詞必須統一。

### Word Embedding
在尋找 Word Embedding 時有想要使用 Sentence Embedding 的想法，可能會改用 Doc2Vec

### ProofVer
[Proofver](https://github.com/krishnamrith12/ProoFVer)
