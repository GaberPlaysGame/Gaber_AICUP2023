## 安裝環境環境設置
本專案運行於以下硬體設備規格：
- Linux 版本 **5.4.0-147-generic**
- RAM 64 GB
- CPU 雙核12顆
- GPU 雙核記憶體 24 GB, 3090 (大部分情況只有用12GB)，第一階段在雙核執行，二三階段可雙核可單核執行，請檢查 DataParallel 部分去做設定

## 程式碼
本專案程式碼共分為四部分。
- document_retrieval.py：第一階段 Document Retrieval 文檔，有牽涉到三部分的文檔檢索，建議分段使用。
- sentence_retrieval.py：第二階段 Sentence Retrieval 文檔，參數皆已經調好，可以調參測試。
- claim_verification.py：第三階段 Claim Verification 文檔，參數皆已經調好，可以調參測試。請注意這個階段目的主要是生成模型，後續若沒有 model ensemble 的需求，可以把下方註解拿掉。
- claim_verification_ensemble.py：第三階段 Model Ensemble 的部分，需要用到三個模型。我們將最高分數的三個模型附上，可以做測試。

### document_retrieval.py
#### Search
從第185行開始為 Search 的部分，當生成完後，應該會要有個名字類似於 train_doc5_search_0316.jsonl 的文檔，欄位必須要有 "predicted_pages" 和 "direct_match"，否則後續兩階段無法進行，請檢查是否有此欄位。

#### TF-IDF
從第406行開始為 TFIDF 部分，請先做上一階段從第355行開始的前置設定。生成出來會有三個檔案：
- train_doc5_tfidf_0316.jsonl: 最初始的檔案，純粹的TOP50
- train_doc5_tfidf_0316_union.jsonl: TOP50 與 Search 的 "direct_match" 欄位合併後的檔案
- train_doc5_tfidf_0316_with_d.jsonl: union 的檔案加上 "direct_match" 的欄位，也是我們的最後產出，第三階段會用到此檔案，請同樣確認欄位必須要有 "predicted_pages" 和 "direct_match"，否則後續階段無法進行。

#### SBERT
從第527行開始為 SBERT 部分，請先做從第355行開始的前置設定。生成出來會有一個 train_doc5_sbert_0316.jsonl 以及數個 logging 檔。logging 檔部分為方便調參，紀錄了 SBERT 的中間產出資料，可以不用理會。請注意 SBERT 的產生時間非常久，8000筆大約需花費近20個小時，因此務必請分階段執行。

#### Merge
若是有需要合併檔案的需求，可以把從第728行的程式碼註解移除，合併兩階段的訓練集。