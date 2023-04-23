# Document Retrieval
原本結果為：
- Precision: 0.2525661209068016
- Recall: 0.8066447523089846

## Improvement
### Redirection
訓練了136分鐘，在採用 Pandarallel 的情況下。

| Experiment  | Precision | Recall | Filename |
| ----------- | --------- | ------ | -------- |
| Main                          | 0.25256612090680160 | 0.8062447523089846 | **train_doc5_aicup.jsonl**                     |
| Redirection, first 8          | 0.15681787513494116 | 0.8258028967254409 | **train_doc5_with_redirection.jsonl**          |
| Fullname, Stopwords           | 0.16030084262924360 | 0.8328872795969775 | **train_doc5_with_stopwords.jsonl**            |
| Fixed English bug             | 0.16064943924673190 | 0.8338318639798489 | **train_doc5_fix_english.jsonl**               |
| Delete Replicate English page | 0.16275338850905655 | 0.8340942485306465 | **train_doc5_delete_replicate_eng_page.jsonl** |
| Repeat Mention                | 0.23450026306606990 | 0.8795759865659112 | **train_doc5_repeat_mention.jsonl**            |
| Quote Full Name               | 0.24161798354178782 | 0.8767422334172965 | **train_doc5_quote_fullname.jsonl**            |

還需要改善的點：
1. 需要 Hyperlink，有些沒有直接提及。
2. 初設定取值在find值越前面的地方，但是有時候主詞在後面，會被濾掉
    - 目前計畫採用直接搜尋方式，如果說有完整對到的會列在比較前面，其餘搜尋結果無條件靠後
    - 現在改用重複提及，如果有重複被搜尋到的就會被放到list內。
3. 英文的重複搜尋還沒解決，如ACM acm同時存在
4. 針對引號等等，不能分割搜尋

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