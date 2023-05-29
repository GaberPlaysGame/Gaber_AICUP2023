import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
from functools import partial

# 3rd party libs
import hanlp
import opencc
import pandas as pd
import torch
import swifter
from sentence_transformers import SentenceTransformer, util
from hanlp.components.pipeline import Pipeline
from pandarallel import pandarallel
from TCSP import read_stopwords_list
from tqdm.notebook import tqdm

# our own libs
from utils import load_json
from hw3_utils import jsonl_dir_to_df
from dr_util import Claim, AnnotationID, Evidence, EvidenceID, PageTitle, SentenceID
from dr_util import calculate_precision, calculate_f1, calculate_recall, save_doc

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=16)
tqdm.pandas()

stopwords = read_stopwords_list()

# data
TRAIN_DATA = load_json("data/public_train_0522.jsonl")
TEST_DATA = load_json("data/public_test.jsonl")
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

def tokenize(text: str, stopwords: list) -> str:
    import jieba
    tokens = jieba.cut(text)

    return " ".join([w for w in tokens if w not in stopwords])

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
    model: SentenceTransformer,
    wiki_pages: pd.DataFrame,
    topk: int,
    threshold: float
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

    tokens = tokenizing_method(claim)
    emb_claim_tok = model.encode(tokens)
    emb_claim = model.encode(claim)

    search_list = [post_processing(id) for id in search_list]

    for search_id in search_list:
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
        emb_id = model.encode(search_id_tok)
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

        embs = model.encode_multi_process(search_lines, pool=pool)
        for emb in embs:
            sim = util.pytorch_cos_sim(emb, emb_claim).numpy()
            sim = sim[0][0]
            sim_line = max(sim, sim_line)

        search_lines_tok = [tokenizing_method(line) for line in search_lines]
        embs = model.encode_multi_process(search_lines_tok, pool=pool)
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

    return set(results)

if __name__ == '__main__':
    sbert_model = SentenceTransformer('uer/sbert-base-chinese-nli', device='cpu')
    pool = sbert_model.start_multi_process_pool()
    print(pool)

    wiki_path = "data/wiki-pages"
    topk = 5

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

    doc_path = f"data/train_doc5.jsonl"
    doc_path_sbert = f"data/train_doc5_sbert.jsonl"
    doc_path_search = f"data/train_doc5_search.jsonl"

    predictor = (hanlp.pipeline().append(
        hanlp.load("FINE_ELECTRA_SMALL_ZH"),
        output_key="tok",
    ).append(
        hanlp.load("CTB9_CON_ELECTRA_SMALL"),
        output_key="con",
        input_key="tok",
    ))

    hanlp_file = f"data/hanlp_con_results.pkl"
    if Path(hanlp_file).exists():
        with open(hanlp_file, "rb") as f:
            hanlp_results = pickle.load(f)
    else:
        hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA]
        with open(hanlp_file, "wb") as f:
            pickle.dump(hanlp_results, f)

    if Path(doc_path_search).exists():
        with open(doc_path_search, "r", encoding="utf8") as f:
            predicted_results_search = pd.Series([
                set(json.loads(line)["predicted_pages"])
                for line in f
            ], name="search")
    else:
        pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=16)
        train_df = pd.DataFrame(TRAIN_DATA)
        train_df.loc[:, "hanlp_results"] = hanlp_results
        # predicted_results = train_df.progress_apply(get_pred_pages, axis=1)
        train_df_search = train_df.parallel_apply(
            get_pred_pages_search, axis=1)
        predicted_results_search_p = train_df_search["predicted_pages"]
        predicted_results_search_d = train_df_search["direct_match"]
        save_doc(TRAIN_DATA, predicted_results_search_p, mode="train", suffix="_search", col_name="predicted_pages")
        TRAIN_DATA_SEARCH = load_json(doc_path_search)
        save_doc(TRAIN_DATA_SEARCH, predicted_results_search_d, mode="train", suffix="_search", col_name="direct_match")

    # num_of_samples = 3969
    # num_of_samples = 500
    # TRAIN_DATA_SEARCH = load_json(doc_path_search)
    # train_df_search = pd.DataFrame(TRAIN_DATA_SEARCH[:num_of_samples])

    # if Path(doc_path_sbert).exists():
    #     with open(doc_path_sbert, "r", encoding="utf8") as f:
    #         predicted_results_sbert = pd.Series([
    #             set(json.loads(line)["predicted_pages"])
    #             for line in f
    #         ], name="sbert")
    # else:
    #     pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
    #     print("Start predicting documents:")
    #     predicted_results_sbert = train_df_search.progress_apply(
    #         partial(
    #             get_pred_pages_sbert,
    #             tokenizing_method=partial(tokenize, stopwords=stopwords),
    #             model=sbert_model,
    #             wiki_pages=wiki_pages,
    #             pool=pool,
    #             topk=topk,
    #             threshold=0.375
    #         ), axis=1)
    #     save_doc(TRAIN_DATA[:num_of_samples], predicted_results_sbert, mode="train", suffix="")