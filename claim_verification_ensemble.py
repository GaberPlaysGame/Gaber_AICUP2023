import pickle
from pathlib import Path
from typing import Dict, Tuple

# import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
from functools import partial

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
# import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import BERTDataset
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)

class AicupTopkEvidenceBERTDataset(BERTDataset):
    """AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        claim = item["claim"]
        evidence = item["evidence_list"]

        # In case there are less than topk evidence sentences
        pad = ["[PAD]"] * (self.topk - len(evidence))
        evidence += pad
        concat_claim_evidence = " [SEP] ".join([*claim, *evidence])

        concat = self.tokenizer(
            concat_claim_evidence,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        label = LABEL2ID[item["label"]] if "label" in item else -1
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}

        if "label" in item:
            concat_ten["labels"] = torch.tensor(label)

        return concat_ten

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=4)

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
SR_THRESHOLD = 0.0
TRAIN_TEST_SPLIT = 0.1

TRAIN_DATA = load_json(f"data/sent_retrieval/split_{TRAIN_TEST_SPLIT}/train_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}.jsonl")
DEV_DATA = load_json(f"data/sent_retrieval/split_{TRAIN_TEST_SPLIT}/dev_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}.jsonl")

wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
del wiki_pages

def join_with_topk_evidence(
    df: pd.DataFrame,
    mapping: dict,
    mode: str = "train",
    topk: int = 5,
) -> pd.DataFrame:
    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns:
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)

    print(f"Extracting evidence_list for the {mode} mode ...")
    df["evidence_list"] = df["predicted_evidence"].parallel_map(lambda x: [
        mapping.get(evi_id, {}).get(str(evi_idx), "")
        for evi_id, evi_idx in x  # for each evidence list
    ][:topk] if isinstance(x, list) else [])
    print(df["evidence_list"][:topk])

    return df

def run_evaluation_ensemble(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    model3: torch.nn.Module,
    dataloader: DataLoader,
    device
) -> list:
    model1.eval()
    model2.eval()
    model3.eval()

    y_true = []
    y_pred = []
    y_pred1 = []
    y_pred2 = []
    y_pred3 = []

    loss1 = 0
    loss2 = 0
    loss3 = 0
    # weight1 = 1
    # weight2 = 1
    # weight3 = 1

    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_true.extend(batch["labels"].tolist())

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs1 = model1(**batch)
            outputs2 = model2(**batch)
            outputs3 = model3(**batch)
            loss1 += outputs1.loss.sum().item()
            loss2 += outputs2.loss.sum().item()
            loss3 += outputs3.loss.sum().item()
            logits1 = outputs1.logits
            logits2 = outputs2.logits
            logits3 = outputs3.logits
            y_pred1.extend(logits1.tolist())
            y_pred2.extend(logits2.tolist())
            y_pred3.extend(logits3.tolist())
            
    for i in range(0, len(y_pred1)):
        logits1 = y_pred1[i]
        # logits1 = [j * weight1 for j in logits1]
        logits2 = y_pred2[i]
        # logits2 = [j * weight2 for j in logits2]
        logits3 = y_pred3[i]
        # logits3 = [j * weight3 for j in logits3]
        all_logits = [logits1, logits2, logits3]

        ave_logits = [sum(x) / 3 for x in zip(*all_logits)]
        y_pred.append(ave_logits.index(max(ave_logits)))

    acc_pred = accuracy_score(y_true, y_pred)
    loss = (loss1 + loss2 + loss3) / 3

    return {"val_loss": loss / len(dataloader), 
            "val_acc_pred": acc_pred}

def run_predict_ensemble(
    model1: torch.nn.Module, 
    model2: torch.nn.Module, 
    model3: torch.nn.Module, 
    test_dl: DataLoader, 
    device) -> list:

    model1.eval()
    model2.eval()
    model3.eval()

    weight1 = 1
    weight2 = 1
    weight3 = 1

    preds = []
    preds1 = []
    preds2 = []
    preds3 = []

    for batch in tqdm(test_dl,
                      total=len(test_dl),
                      leave=False,
                      desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred1 = model1(**batch).logits
        pred2 = model2(**batch).logits
        pred3 = model3(**batch).logits
        preds1.extend(pred1.tolist())
        preds2.extend(pred2.tolist())
        preds3.extend(pred3.tolist())

    for i in range(0, len(preds1)):
        logits1 = preds1[i]
        logits2 = preds2[i]
        logits3 = preds3[i]
        all_logits = [logits1, logits2, logits3]
        ave_logits = [sum(x) / 3 for x in zip(*all_logits)]
        preds.append(ave_logits.index(max(ave_logits)))
    return preds

MODEL_NAME = "hfl/chinese-lert-base" #@param {type:"string"}
MODEL_LARGE_NAME = "hfl/chinese-lert-large"
MODEL_SHORT = "hfl-lert-base-1"
MODEL_LARGE_SHORT = "hfl-lert-large-1"

EVAL_VERSION = 2
TRAIN_BATCH_SIZE = 32  #@param {type:"integer"}
EVAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
LR = 5.625e-5  #@param {type:"number"}
NUM_EPOCHS = 20  #@param {type:"integer"}
MAX_SEQ_LEN = 256  #@param {type:"integer"}
EVIDENCE_TOPK = 5  #@param {type:"integer"}

OUTPUT_FILENAME = "submission.jsonl"

filename_train = f"data/claim_verification/split_{TRAIN_TEST_SPLIT}/train_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}_top{EVIDENCE_TOPK}"
filename_dev = f"data/claim_verification/split_{TRAIN_TEST_SPLIT}/dev_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}_top{EVIDENCE_TOPK}"
TRAIN_PKL_FILE = Path(filename_train + f".pkl")
DEV_PKL_FILE = Path(filename_dev + f".pkl")

with open(TRAIN_PKL_FILE, "rb") as f:
        train_df = pickle.load(f)

with open(DEV_PKL_FILE, "rb") as f:
        dev_df = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_large = AutoTokenizer.from_pretrained(MODEL_NAME)
val_dataset = AicupTopkEvidenceBERTDataset(
    dev_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
eval_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=0,)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

ckpt_name1 = "val_acc=0.6944_model.2460.pt"  #@param {type:"string"}
# ckpt_name2 = "val_acc=0.6798_model.1175.pt"  #@param {type:"string"}
ckpt_name2 = "val_acc=0.6901_model.1250.pt"  #@param {type:"string"}
ckpt_name3 = "val_acc=0.7004_model.960.pt"  #@param {type:"string"}
EXP_DIR_HEAD_8 = f"claim_verification/e{NUM_EPOCHS}_bs{8}_"
EXP_DIR_HEAD_16 = f"claim_verification/e{NUM_EPOCHS}_bs{16}_"
EXP_DIR_HEAD_32 = f"claim_verification/e{NUM_EPOCHS}_bs{32}_"
EXP_DIR_HEAD_48 = f"claim_verification/e{NUM_EPOCHS}_bs{48}_"
EXP_DIR_TAIL_1 = f"split={0.1}_{LR}_top{5}_{MODEL_LARGE_SHORT}_eval{EVAL_VERSION}_maxlen{MAX_SEQ_LEN}_sr_threshold{0.0}"
# EXP_DIR_TAIL_2 = f"split={0.1}_{LR}_top{5}_{MODEL_SHORT}_eval{EVAL_VERSION}_maxlen{MAX_SEQ_LEN}_sr_threshold{0.05}"
EXP_DIR_TAIL_2 = f"split={0.1}_{LR}_top{5}_{MODEL_LARGE_SHORT}_eval{EVAL_VERSION}_maxlen{MAX_SEQ_LEN}_sr_threshold{0.05}"
EXP_DIR_TAIL_3 = f"split={0.1}_{LR}_top{5}_{MODEL_SHORT}_eval{EVAL_VERSION}_maxlen{MAX_SEQ_LEN}_sr_threshold{0.0}"

CKPT_DIR_1 = "checkpoints/" + EXP_DIR_HEAD_16 + EXP_DIR_TAIL_1  
# CKPT_DIR_2 = "checkpoints/" + EXP_DIR_HEAD_32 + EXP_DIR_TAIL_2  
CKPT_DIR_2 = "checkpoints/" + EXP_DIR_HEAD_16 + EXP_DIR_TAIL_2  
CKPT_DIR_3 = "checkpoints/" + EXP_DIR_HEAD_32 + EXP_DIR_TAIL_3  

model1 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_LARGE_NAME,
    num_labels=len(LABEL2ID),
)
model2 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_LARGE_NAME,
    num_labels=len(LABEL2ID),
)
model3 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
)
model1.to(device)
model2.to(device)
model3.to(device)

model1 = load_model(model1, ckpt_name1, CKPT_DIR_1)
model2 = load_model(model2, ckpt_name2, CKPT_DIR_2)
model3 = load_model(model3, ckpt_name3, CKPT_DIR_3)

# val_results = run_evaluation_ensemble(model1, model2, model3, eval_dataloader, device)
# for metric_name, metric_value in val_results.items():
#     print(f"{metric_name}: {metric_value}")

#### On Private Data
suffix = "private"
test_data_name = f"split_{TRAIN_TEST_SPLIT}/test_doc5sent5_{suffix}_neg0.1_2e-05_e1_hfl_bert_split={TRAIN_TEST_SPLIT}"
TEST_DATA = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}.jsonl")
TEST_DATA_NO_THRESHOLD = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.jsonl")
TEST_PKL_FILE = Path(f"data/claim_verification/{test_data_name}_threshold{SR_THRESHOLD}.pkl")
TEST_PKL_FILE_NO_THRESHOLD = Path(f"data/claim_verification/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.pkl")

if not TEST_PKL_FILE.exists():
    test_df = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df.to_pickle(TEST_PKL_FILE, protocol=4)
else:
    with open(TEST_PKL_FILE, "rb") as f:
        test_df = pickle.load(f)
if not TEST_PKL_FILE_NO_THRESHOLD.exists():
    test_df_nothreshold = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA_NO_THRESHOLD),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df_nothreshold.to_pickle(TEST_PKL_FILE_NO_THRESHOLD, protocol=4)
else:
    with open(TEST_PKL_FILE_NO_THRESHOLD, "rb") as f:
        test_df_nothreshold = pickle.load(f)

test_dataset = AicupTopkEvidenceBERTDataset(
    test_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

torch.cuda.empty_cache()
predicted_label = run_predict_ensemble(model1, model2, model3, test_dataloader, device)

predict_dataset = test_df_nothreshold.copy()
predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
    f"submission/{suffix}_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}",
    orient="records",
    lines=True,
    force_ascii=False,
)

TEST_DATA_OUTPUT = load_json(f"submission/{suffix}_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}")
TEST_DATA_ANS = load_json(f"data/{suffix}_test_data_ans.jsonl")
acc = 0
sample = 100

output_df = pd.DataFrame(TEST_DATA_OUTPUT[:sample])
ans_df = pd.DataFrame(TEST_DATA_ANS)

for i in range(0, sample):
    predict_label = output_df["predicted_label"].iloc[i]
    ans_label = ans_df["label"].iloc[i]
    if predict_label == ans_label:
        acc += 1
print(acc / sample)

#### On Public Data
suffix = "public"
test_data_name = f"split_{TRAIN_TEST_SPLIT}/test_doc5sent5_{suffix}_neg0.1_2e-05_e1_hfl_bert_split={TRAIN_TEST_SPLIT}"
TEST_DATA = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}.jsonl")
TEST_DATA_NO_THRESHOLD = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.jsonl")
TEST_PKL_FILE = Path(f"data/claim_verification/{test_data_name}_threshold{SR_THRESHOLD}.pkl")
TEST_PKL_FILE_NO_THRESHOLD = Path(f"data/claim_verification/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.pkl")

if not TEST_PKL_FILE.exists():
    test_df = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df.to_pickle(TEST_PKL_FILE, protocol=4)
else:
    with open(TEST_PKL_FILE, "rb") as f:
        test_df = pickle.load(f)
if not TEST_PKL_FILE_NO_THRESHOLD.exists():
    test_df_nothreshold = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA_NO_THRESHOLD),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df_nothreshold.to_pickle(TEST_PKL_FILE_NO_THRESHOLD, protocol=4)
else:
    with open(TEST_PKL_FILE_NO_THRESHOLD, "rb") as f:
        test_df_nothreshold = pickle.load(f)

test_dataset = AicupTopkEvidenceBERTDataset(
    test_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

torch.cuda.empty_cache()
predicted_label = run_predict_ensemble(model1, model2, model3, test_dataloader, device)

predict_dataset = test_df_nothreshold.copy()
predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
    f"submission/{suffix}_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}",
    orient="records",
    lines=True,
    force_ascii=False,
)

TEST_DATA_OUTPUT = load_json(f"submission/{suffix}_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}")
TEST_DATA_ANS = load_json(f"data/{suffix}_test_data_ans.jsonl")
acc = 0
sample = 100

output_df = pd.DataFrame(TEST_DATA_OUTPUT[:sample])
ans_df = pd.DataFrame(TEST_DATA_ANS)

for i in range(0, sample):
    predict_label = output_df["predicted_label"].iloc[i]
    ans_label = ans_df["label"].iloc[i]
    if predict_label == ans_label:
        acc += 1
print(acc / sample)

#### Merge Public & Private Data
all_data_name = Path(f"submission/all_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}")
public_data_name = Path(f"submission/public_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}")
private_data_name = Path(f"submission/private_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}")

with open(public_data_name, "r", encoding="utf8") as f:
    data = f.read()
    f.close()
with open(private_data_name, "r", encoding="utf8") as f:
    data2 = f.read()
    data += data2
    f.close()
with open(all_data_name, "w", encoding="utf8") as f:
    f.write(data)
    f.close()

ANSWER_DATA = load_json(all_data_name)
answer_df = pd.DataFrame(ANSWER_DATA)
answer_df = answer_df.sort_values("id")
print(answer_df)
answer_df.to_json(
    f"submission/allsorted_ensemble_{ckpt_name1[8:14]}_{ckpt_name2[8:14]}_{ckpt_name3[8:14]}_{OUTPUT_FILENAME}",
    orient="records",
    lines=True,
    force_ascii=False,
)
