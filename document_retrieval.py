import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
from functools import partial

# 3rd party libs
import hanlp
import jieba
import numpy as np
import opencc
import pandas as pd
import scipy
from hanlp.components.pipeline import Pipeline
from pandarallel import pandarallel
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from TCSP import read_stopwords_list
from tqdm.notebook import tqdm

from utils import load_json
from hw3_utils import jsonl_dir_to_df

# Set Jieba Dictionary for Tokenization()
jieba.set_dictionary('data/jieba_dict/dict.txt.big')
jieba.initialize()
stopwords = read_stopwords_list()

# Set Up Pandarallel, TQDM for Search, TFIDF & SBERT
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
tqdm.pandas()

# Strings of document path
file_train_0316 = "data/public_train_0316.jsonl"
file_train_0522 = "data/public_train_0522.jsonl"
file_test_public = "data/public_test_data.jsonl"
file_test_private = "data/private_test_data.jsonl"

# Length of document, pre-defined
len_train_0316 = 3969
len_train_0522 = 7678
len_test_public = 989
len_test_private = 8049

# OPENCC for Tradition to Simplified/Simplified to Traditional Chinese
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

# Dataclasses
@dataclass
class Claim:
    data: str
@dataclass
class AnnotationID:
    id: int
@dataclass
class EvidenceID:
    id: int
@dataclass
class PageTitle:
    title: str
@dataclass
class SentenceID:
    id: int
@dataclass
class Evidence:
    data: List[List[Tuple[AnnotationID, EvidenceID, PageTitle, SentenceID]]]

# Functions
def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)
def get_nps_hanlp(
    predictor: Pipeline,
    d: Dict[str, Union[int, Claim, Evidence]],
) -> List[str]:
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]

    return nps
def calculate_precision(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> float:
    precision = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])

        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)

        count += 1

    # Macro precision
    print(f"Precision: {precision / count}")
    return precision / count
def calculate_recall(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> float:
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1

    print(f"Recall: {recall / count}")
    return recall / count
def calculate_f1(precision: float, recall: float) -> float:
    f1 = 2*(precision*recall)/(precision+recall)
    print(f"F1-Score: {f1}")
    return f1
def save_doc(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
    mode: str = "train",
    suffix: str = "",
    num_pred_doc: int = 5,
    col_name = "predicted_pages"
) -> None:
    with open(
        f"data/{mode}_doc{num_pred_doc}{suffix}.jsonl",
        "w",
        encoding="utf8",
    ) as f:
        for i, d in enumerate(data):
            d[col_name] = list(predictions.iloc[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
def tokenize(text: str, stopwords: list) -> str:
    tokens = jieba.cut(text)

    return " ".join([w for w in tokens if w not in stopwords])
def merge_separate_data(
    len: int,                   # Total len for tfidf
    suffix: str,                # Determine save doc suffix
    phase: str,                 # Determine phase, tfidf/search/sbert
    batch_size: int = 500,      # batch
    save_mode: str = "train",   # Determine save doc mode
):
    import math
    rounds = math.ceil(len / batch_size)

    start = 0
    with open(f'data/{save_mode}_doc5_{phase}_{suffix}_{start}.jsonl') as fp:
        data = fp.read()
        fp.close()
    for i in range(1, rounds):
        with open(f'data/{save_mode}_doc5_{phase}_{suffix}_{start+batch_size}.jsonl') as fp:
            data2 = fp.read()
            data += data2
            fp.close()
        start += batch_size
    
    with open (f'data/{save_mode}_doc5_{phase}_{suffix}.jsonl', 'w') as fp:
        fp.write(data)

# 1. Search
def get_pred_pages_search(
        series_data: pd.Series, 
        ):
    import wikipedia
    import re
    import opencc
    import pandas as pd

    import numpy as np

    wikipedia.set_lang("zh")
    CONVERTER_T2S = opencc.OpenCC("t2s.json")
    CONVERTER_S2T = opencc.OpenCC("s2t.json")
    
    def do_st_corrections(text: str) -> str:
        simplified = CONVERTER_T2S.convert(text)
        return CONVERTER_S2T.convert(simplified)
    
    def if_page_exists(page: str) -> bool:
        import requests
        url_base = "https://zh.wikipedia.org/wiki/"
        new_url = [url_base + page, url_base + page.upper()]
        for url in new_url:
            r = requests.head(url)
            if r.status_code == 200:
                return True
            else:
                continue
        return False

    claim = series_data["claim"]
    results = []
    direct_results = []
    nps = series_data["hanlp_results"]
    nps.append(claim)

    def post_processing(page):
        page = do_st_corrections(page)
        page = page.replace(" ", "_")
        page = page.replace("-", "")

    for i, np in enumerate(nps):
        if (if_page_exists(np)):
            try:
                page = do_st_corrections(wikipedia.page(title=np).title)
                if page == np:
                    post_processing(page)
                    direct_results.append(page)
                else:
                    post_processing(page)
                    direct_results.append(page)
            except wikipedia.DisambiguationError as diserr:
                for option in diserr.options:
                    option = do_st_corrections(option)
                    if new_option := re.sub(r"\s\(\S+\)", "", option) in claim:
                        post_processing(option)
                        direct_results.append(option)
                    post_processing(option)
                    results.append(option)
                page = do_st_corrections(wikipedia.search(np)[0])
                if page == np:
                    post_processing(page)
                    direct_results.append(page)
            except wikipedia.PageError as pageerr:
                pass

        # Simplified Traditional Chinese Correction
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        for term in wiki_search_results:
            if (((new_term := term) in claim) or
                ((new_term := term) in claim.replace(" ", "")) or
                ((new_term := term.replace("·", "")) in claim) or                                   # 過濾人名
                ((new_term := re.sub(r"\s\(\S+\)", "", term)) in claim) or                          # 過濾空格 / 消歧義
                ((new_term := term.replace("(", "").replace(")", "").split()[0]) in claim and       # 消歧義與括號內皆有在裡面
                    (new_term := term.replace("(", "").replace(")", "").split()[1]) in claim) or
                ((new_term := term.replace("-", " ")) in claim) or                                  # 過濾槓號
                ((new_term := term.lower()) in claim) or                                            # 過濾大小寫
                ((new_term := term.lower().replace("-", "")) in claim) or                           # 過濾大小寫及槓號
                ((new_term := re.sub(r"\s\(\S+\)", "", term.lower().replace("-", ""))) in claim)    # 過濾大小寫、槓號及消歧義
                ):
                post_processing(term)
                direct_results.append(term)
            post_processing(term)
            results.append(term)

    direct_results = list(set(direct_results))
    results = list(set(results))            # remove duplicates
    series_data["predicted_pages"] = results
    series_data["direct_match"] = direct_results

    return series_data
def dr_search(
    data_name: str,
    data_len: int,
    hanlp_results: list,
    suffix: str,
    batch: int = 2000,
    nb_workers: int = 32,
    save_mode: str = "train",
    start_round: int = 0,
):
    import math
    rounds = math.ceil(data_len / batch)
    pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=nb_workers)
    DATA = load_json(data_name)

    for i in range(start_round, rounds):
        start = i*batch
        df = pd.DataFrame(DATA[start:start+batch])
        df.loc[:, "hanlp_results"] = hanlp_results[start:start+batch]
        df_search = df.parallel_apply(
            get_pred_pages_search, axis=1)
        res_p = df_search["predicted_pages"]
        res_d = df_search["direct_match"]
        save_doc(DATA[start:start+batch], res_p, mode=save_mode, suffix=f"_search_{suffix}_{start}", col_name="predicted_pages")
        DATA_SEARCH = load_json(f"data/{save_mode}_doc5_search_{suffix}_{start}.jsonl")
        save_doc(DATA_SEARCH, res_d, mode=save_mode, suffix=f"_search_{suffix}_{start}", col_name="direct_match")

## Data Param (Choose the data you want, we use first training data for example)
data_name = file_train_0316
data_len = len_train_0316
hanlp_file = f"data/hanlp_con_results_0316.pkl"
save_mode = "train"
'''
    Choose save_mode, "train" or "test"
'''
suffix = "0316"
'''
    Choose the suffix to represent a data, we have for pre-defined suffix:
        "0316": First training data
        "0522": Second training data
        "public": Public test data
        "private": Private test data
    You can choose other than these 4 as the suffix.
'''
JSON_DATA = load_json(data_name)

## Generate HanLP files
predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
).append(
    hanlp.load("CTB9_CON_ELECTRA_SMALL"),
    output_key="con",
    input_key="tok",
))
if Path(hanlp_file).exists():
    with open(hanlp_file, "rb") as f:
        hanlp_results = pickle.load(f)
else:
    hanlp_results = [get_nps_hanlp(predictor, d) for d in JSON_DATA]
    with open(hanlp_file, "wb") as f:
        pickle.dump(hanlp_results, f)

## Main Function
dr_search(data_name=data_name, data_len=data_len, hanlp_results=hanlp_results, suffix=suffix, save_mode=save_mode)
merge_separate_data(len=data_len, suffix=suffix, phase="search", batch_size=2000, save_mode=save_mode)

# 2. Setting for TFIDF and SBERT
## HYPERPARAMS of TFIDF
WIKI_PATH = "data/wiki-pages"
MIN_WIKI_LEN = 10
TOPK_TFIDF = 50
MIN_DF = 1
MAX_DF = 0.8

## Generate wiki.pkl for TFIDF and SBERT
wiki_cache_path = Path(f"data/wiki.pkl")
if wiki_cache_path.exists():
    wiki_pages = pd.read_pickle(wiki_cache_path)
else:
    def text_split(line: str) -> list:
        import re
        line = re.sub(r"[0-9]+\t", "", line)
        lines = line.split("\n")
        lines = list(filter(None, lines))
        return lines
    # You need to download `wiki-pages.zip` from the AICUP website
    wiki_pages = jsonl_dir_to_df(WIKI_PATH)
    # wiki_pages are combined into one dataframe, so we need to reset the index
    wiki_pages = wiki_pages.reset_index(drop=True)

    # tokenize the text and keep the result in a new column `processed_text`
    wiki_pages["lines"] = wiki_pages["lines"].parallel_apply(text_split)
    wiki_pages["processed_text"] = wiki_pages["text"].parallel_apply(
        partial(tokenize, stopwords=stopwords)
    )
    # save the result to a pickle file
    wiki_pages.to_pickle(wiki_cache_path, protocol=4)

wiki_pages = wiki_pages[
    wiki_pages['processed_text'].str.len() > MIN_WIKI_LEN
]

## TFIDF Initialize
corpus = wiki_pages["processed_text"].tolist()
vectorizer = TfidfVectorizer(
    min_df=MIN_DF,
    max_df=MAX_DF,
    use_idf=True,
    sublinear_tf=True,
    ngram_range=(1,2),
)
X = vectorizer.fit_transform(corpus)

## SBERT Initialize
sbert_model = SentenceTransformer('uer/sbert-base-chinese-nli', device='cuda')
pool = sbert_model.start_multi_process_pool()

# 3. TFIDF
def get_pred_pages_tfidf(
    series_data: pd.Series, 
    tokenizing_method: callable,
    vectorizer: TfidfVectorizer,
    tf_idf_matrix: scipy.sparse.csr_matrix,
    wiki_pages: pd.DataFrame,
    topk: int,
) -> set:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    claim = series_data["claim"]
    results = []

    tokens = tokenizing_method(claim)
    claim_vector = vectorizer.transform([tokens])
    sim_scores = cosine_similarity(tf_idf_matrix, claim_vector)
    sim_scores = sim_scores[:, 0]  # flatten the array
    sorted_indices = np.argsort(sim_scores)[::-1]
    topk_sorted_indices = sorted_indices[:topk]
    results = wiki_pages.iloc[topk_sorted_indices]["id"]

    return set(results)
def dr_tfidf(
    data_len: int,                   # Total len for tfidf
    data_name: str,             # Data File name
    suffix: str,       # Determine save doc suffix, 0316/0522/public/private
    save_mode: str = "train",   # Determine save doc mode, train/test
    batch_size: int = 500,      # batch
    start_round: int = 0,       # Start i
):
    import math

    pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=2)
    rounds = math.ceil(data_len / batch_size)
    DATA = load_json(data_name)

    for i in range(start_round, rounds):
        start = i*batch_size
        df_batch = pd.DataFrame(DATA[start:start+batch_size])
        predicted_results = df_batch.parallel_apply(
            partial(
                get_pred_pages_tfidf,
                tokenizing_method=partial(tokenize, stopwords=stopwords),
                vectorizer=vectorizer,
                tf_idf_matrix=X,
                wiki_pages=wiki_pages,
                topk=50,
                threshold=0.0
            ), axis=1
        )
        save_doc(DATA[start:start+batch_size], predicted_results, mode=save_mode, suffix=f"_tfidf_{suffix}_{start}")
def union_result(series_data: pd.Series,) -> set:
    tfidf = series_data["tfidf"]
    search = series_data["search"]
    return set(set(tfidf).union(set(search)))
def union_tfidf_search(
    data_name: str,
    suffix: str,
    save_mode: str = "train",
):
    DATA = load_json(data_name)

    with open(f"data/{save_mode}_doc5_search_{suffix}.jsonl", "r", encoding="utf8") as f:
        predicted_results_search = pd.Series([
            set(json.loads(line)["direct_match"])
            for line in f
        ], name="search")
    with open(f"data/{save_mode}_doc5_tfidf_{suffix}.jsonl", "r", encoding="utf8") as f:
        predicted_results_tfidf = pd.Series([
            set(json.loads(line)["predicted_pages"])
            for line in f
        ], name="tfidf")

    results_df = pd.merge(pd.Series([line for line in predicted_results_tfidf], name="tfidf"), 
                        pd.Series([line for line in predicted_results_search], name="search"), right_index=True, left_index=True)
    predicted_results = results_df.apply(union_result, axis=1)
    save_doc(DATA, predicted_results, mode=save_mode, suffix=f"_tfidf_{suffix}_union", col_name="predicted_pages")

    total = 0
    for data in predicted_results:
        total += len(data)
    print(f"Total Predicted pages: {total}")

    if save_mode == "train":
        recall = calculate_recall(DATA, predicted_results)
def append_tfidf(
    suffix: str,
    save_mode: str = "train",
):
    DATA = load_json(f"data/{save_mode}_doc5_tfidf_{suffix}_union.jsonl")
    with open(f"data/{save_mode}_doc5_search_{suffix}.jsonl", "r", encoding="utf8") as f:
        direct_match = pd.Series([
            set(json.loads(line)["direct_match"])
            for line in f
        ], name="direct_match")
    save_doc(DATA, direct_match, mode=save_mode, suffix=f"_tfidf_{suffix}_with_d", col_name="direct_match")

## Data Param (Choose the data you want, we use first training data for example)
data_name = file_train_0316
data_len = len_train_0316
hanlp_file = f"data/hanlp_con_results_0316.pkl"
num_of_samples = 500
save_mode = "train"
'''
    Choose save_mode, "train" or "test"
'''
suffix = "0316"
'''
    Choose the suffix to represent a data, we have for pre-defined suffix:
        "0316": First training data
        "0522": Second training data
        "public": Public test data
        "private": Private test data
    You can choose other than these 4 as the suffix.
'''
JSON_DATA = load_json(data_name)

## Main Function
dr_tfidf(data_len=data_len, batch_size=num_of_samples, data_name=data_name, suffix=suffix, save_mode=save_mode)
merge_separate_data(len=data_len, suffix=suffix, phase="tfidf", save_mode=save_mode)
union_tfidf_search(data_name=data_name, suffix=suffix, save_mode=save_mode)
append_tfidf(suffix=suffix, save_mode=save_mode)

# 4. SBERT
def get_pred_pages_sbert(
    series_data: pd.Series, 
    tokenizing_method: callable,
    sbert_model: SentenceTransformer,
    # wiki_pages: pd.DataFrame,
    topk: int,
    threshold: float,
    i: int,
    mode: str = "train",
    suffix: str = "0316",
) -> set:
    # Disable huggingface tokenizor parallelism warning
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

    import torch.cuda as cuda
    cuda.empty_cache()
    
    # Parameters:
    THRESHOLD_LOWEST = 0.6
    THRESHOLD_SIM_LINE = threshold
    WEIGHT_SIM_ID = 0.05    # The lower it is, the higher sim_id is when it directly matches claim.
    
    def sim_score_eval(sim_line, sim_id):
        if len(claim) > 15:
            if sim_line > THRESHOLD_SIM_LINE:
                res = 2*(1.1*sim_line*1.1*sim_id)/(1.1*sim_line+1.1*sim_id)
            else:
                res = 0
        else:
            res = sim_id
        
        return res
    
    def post_processing(page) -> str:
        import opencc
        CONVERTER_T2S = opencc.OpenCC("t2s.json")
        CONVERTER_S2T = opencc.OpenCC("s2t.json")
    
        simplified = CONVERTER_T2S.convert(page)
        page = CONVERTER_S2T.convert(simplified)
        page = page.replace(" ", "_")
        page = page.replace("-", "")
        return page

    claim = series_data["claim"]
    search_list = series_data["predicted_pages"]
    direct_search = series_data["direct_match"]
    results = []
    mapping = {}
    df_res = []

    tokens = tokenizing_method(claim)
    emb_claim_tok = sbert_model.encode(tokens)
    emb_claim = sbert_model.encode(claim)

    search_list = [post_processing(id) for id in search_list]

    for search_id in search_list:
        # print(search_id)
        search_series = wiki_pages.loc[wiki_pages['id'] == search_id]
        if search_series.empty:
            continue
        try:
            for temp in search_series["lines"]:
                search_lines = temp
        except:
            continue

        if len(search_lines) == 0:
             continue
        search_id_tok = tokenizing_method(search_id)
        emb_id = sbert_model.encode(search_id_tok)
        sim_id = util.pytorch_cos_sim(emb_id, emb_claim).numpy()
        sim_id = sim_id[0][0]
        new_sim_id = 0
        if search_id in direct_search:
            if sim_id > 0:
                new_sim_id = 1-((1-sim_id)*WEIGHT_SIM_ID)
            else:
                sim_id = 0
                new_sim_id = 1-((1-sim_id)*WEIGHT_SIM_ID)
        else:
            new_sim_id = sim_id

        sim_score = 0
        sim_line = 0
        sim_line_b = 0

        embs = sbert_model.encode_multi_process(search_lines, pool=pool)
        for emb in embs:
            sim = util.pytorch_cos_sim(emb, emb_claim).numpy()
            sim = sim[0][0]
            sim_line = max(sim, sim_line)

        search_lines_tok = [tokenizing_method(line) for line in search_lines]
        embs = sbert_model.encode_multi_process(search_lines_tok, pool=pool)
        for emb in embs:
            sim = util.pytorch_cos_sim(emb, emb_claim_tok).numpy()
            sim = sim[0][0]
            sim_line = max(sim, sim_line)

        if sim_line > THRESHOLD_SIM_LINE:
            sim_line = max(sim_line, sim_line_b)
            sim_line_b = sim_line
            sim_score = sim_score_eval(sim_line, new_sim_id)
            sim_score = max(sim_score, sim_line_b)
            if sim_score > THRESHOLD_LOWEST:
                search_id = post_processing(search_id)
                if search_id in mapping:
                    mapping[search_id] = max(sim_score, mapping[search_id])
                else:
                    mapping[search_id] = sim_score
        data = (claim, search_id, sim_id, new_sim_id, sim_line, sim_score)
        df_res.append(data)

    mapping_sorted = sorted(mapping.items(), key=lambda x:x[1], reverse=True)
    DIFF = 0.125
    for k, v in mapping_sorted:
        THRESHOLD_TOP = v
        break
    if len(mapping_sorted) >= topk:
        results = [k for k, v in mapping_sorted if v > THRESHOLD_TOP-DIFF][:topk]
    else:
        results = [k for k, v in mapping_sorted if v > THRESHOLD_LOWEST][:topk]
    if not results:
        results = [k for k, v in mapping_sorted][:topk]
    if not results:
        results = series_data["direct_match"]
    if not results:
        results = series_data["predicted_pages"][:topk]

    df = pd.DataFrame(df_res, columns=['Claim', 'Search_ID', 'Sim_ID', 'Sim_ID_Adjusted', 'Sim_Line', 'Sim_Score'])

    with open(f"data/{mode}_doc5_logging_{suffix}_{i}.jsonl", "a", encoding="utf8") as f:
        f.write(df.to_json(orient='records', lines=True, force_ascii=False))
    
    return set(results)
def dr_sbert(
    suffix: str,
    data_len: int,
    compare_data: str,          # Compare Data Filename
    end_round: int, 
    num_of_samples: int = 500,
    start_round: int = 0,
    save_mode: str = "train"
):
    DATA = load_json(f"data/{save_mode}_doc5_tfidf_{suffix}_with_d.jsonl")
    COMPARE = load_json(compare_data)

    for i in range(start_round, end_round):
        start = i*num_of_samples
        df_tfidf = pd.DataFrame(DATA[start:start+num_of_samples])
        results_sbert = df_tfidf.progress_apply(
            partial(
                get_pred_pages_sbert,
                tokenizing_method=partial(tokenize, stopwords=stopwords),
                sbert_model = sbert_model,
                topk=5,
                threshold=0.375,
                i = i,
                suffix = suffix,
                mode = save_mode,
            ), axis=1)
        save_doc(COMPARE[start:start+num_of_samples], results_sbert, mode=save_mode, suffix=f"_sbert_{suffix}_{start}")

## Data Param (Choose the data you want, we use first training data for example)
data_name = file_train_0316
data_len = len_train_0316
hanlp_file = f"data/hanlp_con_results_0316.pkl"
num_of_samples = 500
save_mode = "train"
'''
    Choose save_mode, "train" or "test"
'''
suffix = "0316"
'''
    Choose the suffix to represent a data, we have for pre-defined suffix:
        "0316": First training data
        "0522": Second training data
        "public": Public test data
        "private": Private test data
    You can choose other than these 4 as the suffix.
'''
end_round = math.ceil(data_len / num_of_samples)

## Main Function
dr_sbert(suffix="0316", data_len=data_len, compare_data=data_name, start_round=1, end_round=end_round, save_mode=save_mode)
merge_separate_data(len=data_len, suffix=suffix, phase="sbert", save_mode=save_mode)

print(f"On Original Data, batch")
with open(f"data/{save_mode}_doc5_sbert_{suffix}.jsonl", "r", encoding="utf8") as f:
    predicted_results_original = pd.Series([
        set(json.loads(line)["predicted_pages"])
        for line in f
    ], name="sbert")
old_precision = calculate_precision(JSON_DATA, predicted_results_original)
old_recall = calculate_recall(JSON_DATA, predicted_results_original)
old_f1 = calculate_f1(old_precision, old_recall)

# Merge Two Data (if needed)
# with open(f'data/train_doc5_sbert_0316.jsonl') as fp:
#     data = fp.read()
#     fp.close()
# with open(f'data/train_doc5_sbert_0522.jsonl') as fp:
#     data2 = fp.read()
#     data += data2
#     fp.close()

# with open (f'data/train_doc5_sbert.jsonl', 'w') as fp:
#     fp.write(data)
#     fp.close()