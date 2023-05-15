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

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
tqdm.pandas()

stopwords = read_stopwords_list()

# data
TRAIN_DATA = load_json("data/public_train.jsonl")
TEST_DATA = load_json("data/public_test.jsonl")
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

def tokenize(text: str, stopwords: list) -> str:
    import jieba
    tokens = jieba.cut(text)

    return " ".join([w for w in tokens if w not in stopwords])

def get_pred_pages_sbert(
    series_data: pd.Series, 
    tokenizing_method: callable,
    model: SentenceTransformer,
    wiki_pages: pd.DataFrame,
    topk: int,
    threshold: float
) -> set:

    # Parameters:
    THRESHOLD_LOWEST = 0.55
    THRESHOLD_HIGHEST = 0.7
    THRESHOLD_SIM_LINE = threshold
    WEIGHT_SIM_ID = 0.05    # The lower it is, the higher sim_id is when it directly matches claim.
    
    def sim_score_eval(s1, s2):
        weight_id = 1
        weight_line = 1
        
        return (weight_id + weight_line)*(s1*s2)/(weight_line*s1+weight_id*s2)
    
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
    claim_sentences_count = claim.count('，')+1
    search_list = series_data["predicted_pages"]
    direct_search = series_data["direct_match"]
    results = []
    mapping = {}

    tokens = tokenizing_method(claim)
    emb_claim_tok = model.encode(tokens)
    emb_claim = model.encode(claim)

    search_list = [post_processing(id) for id in search_list]
    if series_data["label"] != "NOT ENOUGH INFO":
        gt_pages = set([
            evidence[2]
            for evidence_set in series_data["evidence"]
            for evidence in evidence_set
        ])
    else:
        gt_pages = set([])

    for search_id in search_list:
        # print(search_id)
        search_series = wiki_pages.loc[wiki_pages['id'] == search_id]
        if search_series.empty:
            continue
        try:
            for temp in search_series["lines"]:
                search_lines = temp
            # search_tokens = tokenizing_method(search_lines)
        except:
            continue
        # search_processed_text = str(search_tokens["processed_text"].astype("string"))
        search_id_tok = tokenizing_method(search_id)
        emb_id = model.encode(search_id_tok)
        sim_id = util.pytorch_cos_sim(emb_id, emb_claim).numpy()
        sim_id = sim_id[0][0]
        new_sim_id = 0
        if search_id in direct_search:
            if sim_id > 0:
                # print(f"{search_id}: sim_id={sim_id}")
                new_sim_id = 1-((1-sim_id)*WEIGHT_SIM_ID)
            else:
                sim_id = 0
                new_sim_id = 1-((1-sim_id)*WEIGHT_SIM_ID)
        else:
            new_sim_id = sim_id

        sim_score = 0
        sim_line_b = 0
        for search_line in search_lines:
            # print(search_line)
            search_line_count = search_line.count('，')+1
            if search_line_count > claim_sentences_count and claim_sentences_count > 1:
                search_line_list = search_line.split('，')
                sim_line = 0
                for i in range(0, search_line_count-claim_sentences_count+1):
                    line = "，".join(search_line_list[i:i+claim_sentences_count])
                    # print(line)
                    search_token = tokenizing_method(line)
                    emb_search = model.encode(search_token)
                    sim = util.pytorch_cos_sim(emb_search, emb_claim_tok).numpy()
                    sim = sim[0][0]
                    sim_line = max(sim, sim_line)

                    emb_search = model.encode(line)
                    sim = util.pytorch_cos_sim(emb_search, emb_claim).numpy()
                    sim = sim[0][0]
                    sim_line = max(sim, sim_line)
            else:
                search_token = tokenizing_method(search_line)
                emb_search = model.encode(search_token)
                sim_line = util.pytorch_cos_sim(emb_search, emb_claim_tok).numpy()
                sim_line = sim_line[0][0]

                emb_search = model.encode(search_line)
                sim = util.pytorch_cos_sim(emb_search, emb_claim).numpy()
                sim = sim[0][0]
                sim_line = max(sim, sim_line)
            
            # print(sim_scores)
            if sim_line > THRESHOLD_SIM_LINE:
                sim_line = max(sim_line, sim_line_b)
                sim_line_b = sim_line
                sim_score = sim_score_eval(sim_line, new_sim_id)
                # print(sim_score, search_id)
                if sim_score > THRESHOLD_LOWEST:
                    search_id = post_processing(search_id)
                    if search_id in mapping:
                        mapping[search_id] = max(sim_score, mapping[search_id])
                    else:
                        mapping[search_id] = sim_score
        if search_id in gt_pages:
            print(f"Analysis on GT pages={search_id}: origin sim_id={sim_id}, new sim_id={new_sim_id}, sim_line={sim_line_b}, sim_score={sim_score}")

    # print(mapping)
    mapping_sorted = sorted(mapping.items(), key=lambda x:x[1], reverse=True)
    # print(mapping_sorted)
    results = [k for k, v in mapping_sorted if v > THRESHOLD_HIGHEST][:topk]
    # print(results)
    if not results:     # List is empty
        results= [k for k, v in mapping_sorted if v > THRESHOLD_LOWEST][:1]
        # print(f"Empty, new_results={results}")   

    # Analysis on missed pages
    
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

    return set(results)

if __name__ == '__main__':
    # print(torch.cuda.device_count())
    sbert_model = SentenceTransformer('uer/sbert-base-chinese-nli', device='cpu')

    wiki_path = "data/wiki-pages"
    min_wiki_length = 25
    topk = 5
    min_df = 1
    max_df = 0.5
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

    doc_path = f"data/train_doc5.jsonl"
    doc_path_aicup = f"data/train_doc5_aicup.jsonl"
    doc_path_sbert = f"data/train_doc5_sbert.jsonl"
    doc_path_search = f"data/train_doc5_search.jsonl"
    doc_path_tfidf = f"data/train_doc5_tfidf.jsonl"

    # num_of_samples = 3969
    num_of_samples = 3969
    TRAIN_DATA_SEARCH = load_json(doc_path_search)
    train_df_search = pd.DataFrame(TRAIN_DATA_SEARCH[:num_of_samples])

    if Path(doc_path_sbert).exists():
        with open(doc_path_sbert, "r", encoding="utf8") as f:
            predicted_results_sbert = pd.Series([
                set(json.loads(line)["predicted_pages"])
                for line in f
            ], name="sbert")
    else:
        pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=3)
        print("Start predicting documents:")
        predicted_results_sbert = train_df_search.parallel_apply(
            partial(
                get_pred_pages_sbert,
                tokenizing_method=partial(tokenize, stopwords=stopwords),
                model=sbert_model,
                wiki_pages=wiki_pages,
                topk=5,
                threshold=0.375
            ), axis=1)
        save_doc(TRAIN_DATA[:num_of_samples], predicted_results_sbert, mode="train", suffix="")

    old_precision = 0.3879699248120298
    old_recall = 0.8124895572263992
    old_f1 = 0.5247852770334176

    print("On Search-TFIDF Data:")
    precision = calculate_precision(TRAIN_DATA[:num_of_samples], predicted_results_sbert)
    print(f"(Diff: {precision-old_precision})")
    recall = calculate_recall(TRAIN_DATA[:num_of_samples], predicted_results_sbert)
    print(f"(Diff: {recall-old_recall})")
    f1 = calculate_f1(precision, recall)
    print(f"F1-Score: {f1}")
    print(f"(Diff: {f1-old_f1})")