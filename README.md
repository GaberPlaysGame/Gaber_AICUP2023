## Improvement
### Document Retrieval
原本結果為：
- Precision: 0.2525661209068016
- Recall: 0.8066447523089846

訓練了136分鐘，在採用 Pandarallel 的情況下。

| Experiment  | Precision | Recall | Filename |
| ----------- | --------- | ------ | -------- |
| Main                          | 0.25093750000000560 | 0.8073333333333337 | **train_doc5_aicup.jsonl**                     |
| Redirection, first 8          | 0.15681787513494116 | 0.8258028967254409 | **train_doc5_with_redirection.jsonl**          |
| Fullname, Stopwords           | 0.16030084262924360 | 0.8328872795969775 | **train_doc5_with_stopwords.jsonl**            |
| Fixed English bug             | 0.16064943924673190 | 0.8338318639798489 | **train_doc5_fix_english.jsonl**               |
| Delete Replicate English page | 0.16275338850905655 | 0.8340942485306465 | **train_doc5_delete_replicate_eng_page.jsonl** |
| Repeat Mention                | 0.23450026306606990 | 0.8795759865659112 | **train_doc5_repeat_mention.jsonl**            |
| Quote Full Name               | 0.24161798354178782 | 0.8767422334172965 | **train_doc5_quote_fullname.jsonl**            |
| On New Data                   | 0.24198516414141477 | 0.8701666666666668 | **train_doc5_new_1.jsonl**                     |

還需要改善的點：
1. 需要 Hyperlink，有些沒有直接提及。
2. 初設定取值在find值越前面的地方，但是有時候主詞在後面，會被濾掉
    - 目前計畫採用直接搜尋方式，如果說有完整對到的會列在比較前面，其餘搜尋結果無條件靠後
    - 現在改用重複提及，如果有重複被搜尋到的就會被放到list內。
3. 英文的重複搜尋還沒解決，如ACM acm同時存在
4. 針對引號等等，不能分割搜尋

### Sentence Retrieval
利用原本訓練出來的model50, model100與model150下去做validation。

用我的GPU不知道為甚麼要跑到10小時，Batch Size也只接受到32。改成Colab後Batch Size到64，只有一個model50，執行時間只要6分鐘。

原本的成績：
| Model | Train-F1 | Train-P | Train-R | Valid-F1 | Valid-P | Valid-R |
| ----- | -------- | ------- | ------- | -------- | ------- | ------- |
| model50-0      | 0.357773 | 0.237802 | 0.722047 | 0.333525 | 0.216876 | 0.721698 |
| model50-1-32   | 0.398642 | 0.267014 | 0.786220 | 0.364270 | 0.237342 | 0.783018 |
| model100-1-32  | 0.403016 | 0.269849 | 0.795669 | 0.365551 | 0.238286 | 0.784591 |
| model150-1-32  | 0.402539 | 0.269376 | 0.796062 | 0.364640 | 0.237657 | 0.783018 |
| model30-1-64   | 0.373679 | 0.249173 | 0.746875 | 0.370676 | 0.246171 | 0.750000 |
| model50-1-64   | 0.389274 | 0.259563 | 0.778125 | 0.388977 | 0.256796 | 0.801563 |
| model60-1-64   | 0.384763 | 0.256595 | 0.768750 | 0.383362 | 0.253671 | 0.784375 |
| model90-1-64   | 0.387116 | 0.258079 | 0.774218 | 0.387164 | 0.255859 | 0.795312 |

### Claim Validation
CUDA 在 Colab 上還是容易爆 CudaOutOfMemoryError，我的GPU只能容許Batch Size=8，Colab只能24。

原本的成績與後續對照：
| Model | Val_acc | Train_step | Version | Batch Size |
| ----- | ------- | ---------- | ------- | ---------- |
| model.7900 | 0.4056 | 7900 | 0  | 8  |
| model.950  | 0.4170 | 950  | 0  | 8  |
| model.750  | 0.4259 | 750  | 0  | 32 |

### Submission History
| Model | Val_acc | Public Score | Private Score |
| ----- | ------- | ------------ | ------------- |
| model.950  | 0.4170 | 0.303337 | NaN |
| model.750  | 0.4259 | 0.292214 | NaN |

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