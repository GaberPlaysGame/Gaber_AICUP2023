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

# our own libs
from utils import load_json
from hw3_utils import jsonl_dir_to_df

jieba.set_dictionary('data/jieba_dict/dict.txt.big')
jieba.initialize()
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
).append(
    hanlp.load("CTB9_CON_ELECTRA_SMALL"),
    output_key="con",
    input_key="tok",
))
stopwords = read_stopwords_list()
tqdm.pandas()

doc_path = f"data/train_doc5.jsonl"
doc_path_sbert_0316 = f"data/train_doc5_sbert_0316.jsonl"
doc_path_sbert_0522 = f"data/train_doc5_sbert_0522.jsonl"
doc_path_search_0316 = f"data/train_doc5_search_0316.jsonl"
doc_path_search_0522 = f"data/train_doc5_search_0522.jsonl"
doc_path_tfidf_0316 = f"data/train_doc5_tfidf_0316.jsonl"
doc_path_tfidf_0522 = f"data/train_doc5_tfidf_0522.jsonl"
file_test_private = "data/private_test_data.jsonl"
file_train_0316 = "data/public_train_0316.jsonl"
file_train_0522 = "data/public_train_0522.jsonl"
len_public_0316 = 3969
len_public_0522 = 7678
len_test_private = 8049

TRAIN_DATA_1 = load_json(file_train_0316)
TRAIN_DATA_2 = load_json(file_train_0522)
TEST_DATA_PUBLIC = load_json("data/public_test.jsonl")
TEST_DATA_PRIVATE = load_json("data/private_test_data.jsonl")
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

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

wiki_path = "data/wiki-pages"
min_wiki_length = 10
topk = 50
min_df = 1
max_df = 0.8
use_idf = True
sublinear_tf = True
wiki_cache = "wiki"
target_column = "text"

wiki_cache_path = Path(f"data/{wiki_cache}.pkl")
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
    wiki_pages = jsonl_dir_to_df(wiki_path)
    # wiki_pages are combined into one dataframe, so we need to reset the index
    wiki_pages = wiki_pages.reset_index(drop=True)

    # tokenize the text and keep the result in a new column `processed_text`
    wiki_pages["lines"] = wiki_pages["lines"].parallel_apply(text_split)
    wiki_pages["processed_text"] = wiki_pages[target_column].parallel_apply(
        partial(tokenize, stopwords=stopwords)
    )
    # save the result to a pickle file
    wiki_pages.to_pickle(wiki_cache_path, protocol=4)

wiki_pages = wiki_pages[
    wiki_pages['processed_text'].str.len() > min_wiki_length
]

# TFIDF Initialize
corpus = wiki_pages["processed_text"].tolist()
vectorizer = TfidfVectorizer(
    min_df=min_df,
    max_df=max_df,
    use_idf=use_idf,
    sublinear_tf=sublinear_tf,
    # dtype=np.float64,
    ngram_range=(1,2),
    # norm=None,
)
X = vectorizer.fit_transform(corpus)

# SBERT Initialize
sbert_model = SentenceTransformer('uer/sbert-base-chinese-nli', device='cuda')
pool = sbert_model.start_multi_process_pool()

# Main Function
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
        # print(f"searching {np}")

        if (if_page_exists(np)):
            try:
                page = do_st_corrections(wikipedia.page(title=np).title)
                if page == np:
                    # print(f"Found, np={np}, page={page}, claim={claim}")
                    post_processing(page)
                    direct_results.append(page)
                else:
                    # print(f"Redirect, np={np}, page={page}, claim={claim}")
                    post_processing(page)
                    direct_results.append(page)
            except wikipedia.DisambiguationError as diserr:
                for option in diserr.options:
                    option = do_st_corrections(option)
                    if new_option := re.sub(r"\s\(\S+\)", "", option) in claim:
                        # print(f"Disambig, np={np}, page={option}, claim={claim}")
                        post_processing(option)
                        direct_results.append(option)
                    post_processing(option)
                    results.append(option)
                page = do_st_corrections(wikipedia.search(np)[0])
                if page == np:
                    # print(f"Disambig, np={np}, page={page}, claim={claim}")
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
            # if prefix not in tmp_muji:  #忽略掉括號，如果括號有重複的話。假設如果有" 1 (數字)", 則"1 (符號)" 會被忽略
            post_processing(term)
            results.append(term)

    direct_results = list(set(direct_results))
    results = list(set(results))            # remove duplicates
    series_data["predicted_pages"] = results
    series_data["direct_match"] = direct_results

    return series_data
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
    '''
    if series_data["label"] != "NOT ENOUGH INFO":
        gt_pages = set([
            evidence[2]
            for evidence_set in series_data["evidence"]
            for evidence in evidence_set
        ])
    else:
        gt_pages = set([])
    '''

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

        # Multiple GPU Support:
            # embs = sbert_model.encode_multi_process(search_lines, pool=pool)
            # for emb in embs:
            #     sim = util.pytorch_cos_sim(emb, emb_claim).numpy()
            #     sim = sim[0][0]
            #     sim_line = max(sim, sim_line)

            # search_lines_tok = [tokenizing_method(line) for line in search_lines]
            # embs = sbert_model.encode_multi_process(search_lines_tok, pool=pool)
            # for emb in embs:
            #     sim = util.pytorch_cos_sim(emb, emb_claim_tok).numpy()
            #     sim = sim[0][0]
            #     sim_line = max(sim, sim_line)

        # Single GPU Support:
        for search_line in search_lines:
            emb = sbert_model.encode(search_line)
            sim = util.pytorch_cos_sim(emb, emb_claim).numpy()
            sim = sim[0][0]
            sim_line = max(sim, sim_line)

            search_lines_tok = tokenizing_method(search_line)
            emb = sbert_model.encode(search_lines_tok)
            sim = util.pytorch_cos_sim(emb, emb_claim).numpy()
            sim = sim[0][0]
            sim_line = max(sim, sim_line)

        if sim_line > THRESHOLD_SIM_LINE:
            sim_line = max(sim_line, sim_line_b)
            sim_line_b = sim_line
            sim_score = sim_score_eval(sim_line, new_sim_id)
            sim_score = max(sim_score, sim_line_b)
            # print(sim_score, sim_line, search_id)
            if sim_score > THRESHOLD_LOWEST:
                search_id = post_processing(search_id)
                if search_id in mapping:
                    mapping[search_id] = max(sim_score, mapping[search_id])
                else:
                    mapping[search_id] = sim_score
        data = (claim, search_id, sim_id, new_sim_id, sim_line, sim_score)
        df_res.append(data)

    mapping_sorted = sorted(mapping.items(), key=lambda x:x[1], reverse=True)
    # print(mapping_sorted[:topk])
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
    # print(results)

    # Analysis on missed pages
    '''
    if series_data["label"] != "NOT ENOUGH INFO":
        for page in gt_pages:
            if page in mapping:
                if page not in results:
                    print(f"Missed: ID={page}, score={mapping[page]}")
                else:
                    continue
            else:
                if page not in search_list:
                    print(f"Missed: ID={page}, not in search_list")
                else:
                    print(f"Missed: ID={page}, score < {THRESHOLD_LOWEST}")
    '''
    df = pd.DataFrame(df_res, columns=['Claim', 'Search_ID', 'Sim_ID', 'Sim_ID_Adjusted', 'Sim_Line', 'Sim_Score'])

    with open(f"data/{mode}_doc5_logging_{suffix}_{i}.jsonl", "a", encoding="utf8") as f:
        f.write(df.to_json(orient='records', lines=True, force_ascii=False))
    

    return set(results)
def get_pred_pages_tfidf(
    series_data: pd.Series, 
    tokenizing_method: callable,
    vectorizer: TfidfVectorizer,
    tf_idf_matrix: scipy.sparse.csr_matrix,
    wiki_pages: pd.DataFrame,
    topk: int,
    threshold: float
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

    # for search_id in search_list:
    #     search_tokens = wiki_pages.loc[wiki_pages['id'] == search_id]
    #     if search_tokens.empty:
    #         continue
    #     search_processed_text = search_tokens["processed_text"]
    #     search_vector = vectorizer.transform(search_processed_text)
    #     sim_scores = cosine_similarity(search_vector, claim_vector)
    #     sim_scores = sim_scores[0][0]
    #     if sim_scores > threshold:
    #         mapping[search_id] = sim_scores
            # print(sim_scores, search_id)

    # print(mapping)
    # results = sorted(mapping, key=mapping.get, reverse=True)[:topk]
    # print(results)
    return set(results)
def get_pred_pages_log(
    data: pd.DataFrame, 
    topk: int,
    threshold: float,
    progress_bar,
    
):
    # Parameters:
    THRESHOLD_LOWEST = 0.6
    THRESHOLD_MID = 0.7
    THRESHOLD_HIGHEST = 0.885
    THRESHOLD_SIM_LINE = threshold
    WEIGHT_SIM_ID = 0.2    # The lower it is, the higher sim_id is when it directly matches claim.
    
    def sim_score_eval(sim_line, sim_id, claim):
        if len(claim) <= 15:
            res = sim_id
        else:
            w_line = 1.1
            w_id = 1.1
            if sim_line > THRESHOLD_SIM_LINE:
                res = 2*(w_line*sim_line*w_id*sim_id)/(w_line*sim_line+w_id*sim_id)
            else:
                res = 0
        
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

    results = []
    doc_res = []
    mapping = {}
    claim_prev = ""
    claim_count = 0
    claim_comma = 0
    direct_match = []
    predicted_pages = []

    for index, series_data in data.iterrows():
        claim = series_data["Claim"]
        search_id = series_data["Search_ID"]
        sim_id = series_data["Sim_ID"]
        sim_id_new = series_data["Sim_ID_Adjusted"]
        sim_line = series_data["Sim_Line"]

        if index == 0:  
            claim_prev = claim
            claim_comma = claim.count("，")
        elif claim != claim_prev:
            mapping_sorted = sorted(mapping.items(), key=lambda x:x[1], reverse=True)
            DIFF = 0.125
            for k, v in mapping_sorted:
                THRESHOLD_TOP = v
                break
            # print(mapping_sorted[:topk])
            if len(mapping_sorted) >= topk:
                doc_res = [k for k, v in mapping_sorted if v > THRESHOLD_TOP-DIFF][:topk]
            else:
                doc_res= [k for k, v in mapping_sorted if v > THRESHOLD_LOWEST][:topk]
            if not doc_res:
                doc_res = direct_match[:topk]
            if not doc_res:
                doc_res = predicted_pages[:topk]
            
            results.append(doc_res)
            #print(claim_count, mapping)
            doc_res = []
            mapping = {}
            claim_prev = claim
            claim_comma = claim.count("，")
            claim_count = claim_count + 1
            progress_bar.update(1)

        if sim_id != sim_id_new:
            direct_match.append(search_id)
            if sim_id > 0:
                # print(f"{search_id}: sim_id={sim_id}")
                sim_id_new = 1-((1-sim_id)*WEIGHT_SIM_ID)
            else:
                sim_id = 0
                sim_id_new = 1-((1-sim_id)*WEIGHT_SIM_ID)
        else:
            sim_id_new = sim_id

        predicted_pages.append(search_id)
        sim_score = sim_score_eval(sim_line=sim_line, sim_id=sim_id_new, claim=claim)
        if sim_score > 0:
            sim_score = max(sim_score, sim_line)
            # print(sim_score, search_id)
            if sim_score > THRESHOLD_LOWEST:
                search_id = post_processing(search_id)
                if search_id in mapping:
                    mapping[search_id] = max(sim_score, mapping[search_id])
                else:
                    mapping[search_id] = sim_score

    mapping_sorted = sorted(mapping.items(), key=lambda x:x[1], reverse=True)

    if len(mapping_sorted) >= topk:
        doc_res = [k for k, v in mapping_sorted if v > THRESHOLD_TOP-DIFF][:topk]
    else:
        doc_res= [k for k, v in mapping_sorted if v > THRESHOLD_LOWEST][:topk]
    if not doc_res:
        doc_res = direct_match[:topk]
    if not doc_res:
        doc_res = predicted_pages[:topk]

    results.append(doc_res)

    return results

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
        save_doc(DATA[start:start+batch], res_p, mode=save_mode, suffix=f"_search_{suffix}_{i}p", col_name="predicted_pages")
        DATA_SEARCH = load_json(f"data/{save_mode}_doc5_search_{suffix}_{i}p.jsonl")
        save_doc(DATA_SEARCH, res_d, mode=save_mode, suffix=f"_search_{suffix}_{i}d", col_name="direct_match")
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
        if save_mode == "train" and i <= 13:
            print(f"On TFIDF top 50 Data, batch = {i}:")
            precision = calculate_precision(DATA[start:start+batch_size], predicted_results)
            recall = calculate_recall(DATA[start:start+batch_size], predicted_results)
            f1 = calculate_f1(precision, recall)
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
        try:
            if i < 13 and save_mode == "train":
                print(f"On Sbert Data, batch = {i}:")
                precision = calculate_precision(COMPARE[start:start+num_of_samples], results_sbert)
                recall = calculate_recall(COMPARE[start:start+num_of_samples], results_sbert)
                f1 = calculate_f1(precision, recall)
        except ZeroDivisionError:
            pass


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
def merge_log(series_data: pd.Series, topk: int = 5) -> set:
    result = []
    for i in range(0, topk):
        if series_data.iloc[i] != None:
            result.append(series_data.iloc[i])
    # print(set(result))
    return set(result)

# On Train Files:
hanlp_file = f"data/hanlp_con_results_0522.pkl"
if Path(hanlp_file).exists():
    with open(hanlp_file, "rb") as f:
        hanlp_results = pickle.load(f)
else:
    hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA_1]
    with open(hanlp_file, "wb") as f:
        pickle.dump(hanlp_results, f)

data_name = file_train_0316
suffix = "0316"
num_of_samples = 500
end_round = math.ceil(len_public_0316 / num_of_samples)

dr_tfidf(data_len=len_public_0316, batch_size=num_of_samples, data_name=data_name, suffix=suffix, start_round=3)
merge_separate_data(len=len_public_0316, suffix=suffix, phase="tfidf")
union_tfidf_search(data_name=file_train_0316, suffix=suffix)
append_tfidf(suffix=suffix)

dr_sbert(suffix="0316", data_len=len_public_0316, compare_data=file_train_0316, start_round=1, end_round=end_round)
merge_separate_data(len=len_public_0316, suffix=suffix, phase="sbert")

print(f"On Original Data, batch")
with open(f"data/train_doc5_sbert_0522.jsonl", "r", encoding="utf8") as f:
    predicted_results_original = pd.Series([
        set(json.loads(line)["predicted_pages"])
        for line in f
    ], name="sbert")
old_precision = calculate_precision(TRAIN_DATA_2, predicted_results_original)
old_recall = calculate_recall(TRAIN_DATA_2, predicted_results_original)
old_f1 = calculate_f1(old_precision, old_recall)

# Log File Operation, doesn't needed
# for i in range(14, 16):
#     start = i*num_of_samples
#     doc_log = f"data/train_doc5_logging_0522_{i}.jsonl"
#     TRAIN_DATA_LOG = load_json(doc_log)
#     train_df_log = pd.DataFrame(TRAIN_DATA_LOG)

#     progress_bar = tqdm(range(500))
    
#     predicted_results_log = get_pred_pages_log(
#         data=train_df_log, 
#         topk=topk, 
#         threshold=0.375, 
#         progress_bar=progress_bar
#     )
#     predicted_results_log_df = pd.DataFrame(predicted_results_log)
#     predicted_results_log_df_b = predicted_results_log_df.apply(merge, axis=1)
#     save_doc(TRAIN_DATA[start:start+num_of_samples], predicted_results_log_df_b, mode="train", suffix=f"_log_0522_{i}")

#     with open(f"data/train_doc5_sbert_0522.jsonl", "r", encoding="utf8") as f:
#         predicted_results_original = pd.Series([
#             set(json.loads(line)["predicted_pages"])
#             for line in f
#         ], name="sbert")

#     if i < 13:
#         print(f"On Original Data, batch = {i}")
#         old_precision = calculate_precision(TRAIN_DATA[start:start+num_of_samples], predicted_results_original[start:start+num_of_samples])
#         old_recall = calculate_recall(TRAIN_DATA[start:start+num_of_samples], predicted_results_original[start:start+num_of_samples])
#         old_f1 = calculate_f1(precision, recall)

#         print(f"\nOn Log Data, batch = {i}")
#         precision = calculate_precision(TRAIN_DATA[start:start+num_of_samples], predicted_results_log_df_b)
#         print(f"(Diff: {precision-old_precision})")
#         recall = calculate_recall(TRAIN_DATA[start:start+num_of_samples], predicted_results_log_df_b)
#         print(f"(Diff: {recall-old_recall})")
#         f1 = calculate_f1(precision, recall)
#         print(f"F1-Score: {f1}")
#         print(f"(Diff: {f1-old_f1})")

# Merge Two Training Data
with open(f'data/train_doc5_sbert_0316.jsonl') as fp:
    data = fp.read()
    fp.close()
with open(f'data/train_doc5_sbert_0522.jsonl') as fp:
    data2 = fp.read()
    data += data2
    fp.close()

with open (f'data/train_doc5_sbert.jsonl', 'w') as fp:
    fp.write(data)
    fp.close()

# On Test File
test_doc_path = f"data/test_doc5.jsonl"
test_doc_path_aicup = f"data/test_doc5_aicup.jsonl"
test_doc_path_search = f"data/test_doc5_search.jsonl"
test_doc_path_tfidf = f"data/test_doc5_tfidf.jsonl"
test_doc_path_sbert = f"data/test_doc5_sbert_0522.jsonl"

data_name = file_test_private
suffix = "private"
num_of_samples = 500
end_round = math.ceil(len_test_private / num_of_samples)

hanlp_test_file = f"data/hanlp_con_test_results_private.pkl"
if Path(hanlp_test_file).exists():
    with open(hanlp_test_file, "rb") as f:
        hanlp_test_results = pickle.load(f)
else:
    hanlp_test_results = [get_nps_hanlp(predictor, d) for d in TEST_DATA_PRIVATE]
    with open(hanlp_test_file, "wb") as f:
        pickle.dump(hanlp_test_results, f)

dr_search(
    data_name=file_test_private, 
    data_len=len_test_private, 
    hanlp_results= hanlp_test_results, 
    suffix="private", 
    save_mode="test",
)
dr_tfidf(data_len=len_test_private, batch_size=500, data_name=data_name, suffix=suffix, start_round=11, save_mode="test")
merge_separate_data(len=len_test_private, suffix=suffix, phase="tfidf", save_mode="test")
union_tfidf_search(data_name=data_name, suffix=suffix, save_mode="test")
append_tfidf(suffix=suffix, save_mode="test")
dr_sbert(
    suffix="private", 
    data_len=len_test_private, 
    compare_data=file_test_private, 
    start_round=14, 
    save_mode="test", 
    end_round=end_round,
)



