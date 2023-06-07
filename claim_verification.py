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
SR_THRESHOLD = 0.05
TRAIN_TEST_SPLIT = 0.1

TRAIN_DATA = load_json(f"data/sent_retrieval/split_{TRAIN_TEST_SPLIT}/train_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}.jsonl")
DEV_DATA = load_json(f"data/sent_retrieval/split_{TRAIN_TEST_SPLIT}/dev_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}.jsonl")

wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
del wiki_pages

def run_predict(model: torch.nn.Module, test_dl: DataLoader, device) -> list:
    model.eval()

    preds = []
    for batch in tqdm(test_dl,
                      total=len(test_dl),
                      leave=False,
                      desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch).logits
        pred = torch.argmax(pred, dim=1)
        preds.extend(pred.tolist())
    return preds

def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, device):
    model.eval()

    loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_true.extend(batch["labels"].tolist())

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss.sum().item()
            logits = outputs.logits
            y_pred.extend(torch.argmax(logits, dim=1).tolist())

    acc = accuracy_score(y_true, y_pred)

    return {"val_loss": loss / len(dataloader), "val_acc": acc}

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

MODEL_NAME = "hfl/chinese-lert-base" #@param {type:"string"}
# MODEL_NAME = "hfl/chinese-lert-large" #@param {type:"string"}

MODEL_SHORT = "hfl-lert-base-1"
EVAL_VERSION = 2
TRAIN_BATCH_SIZE = 16  #@param {type:"integer"}
TEST_BATCH_SIZE = 32  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
LR = 5.625e-5  #@param {type:"number"}
NUM_EPOCHS = 20  #@param {type:"integer"}
REAL_EPOCHS = 7
MAX_SEQ_LEN = 256  #@param {type:"integer"}
EVIDENCE_TOPK = 5  #@param {type:"integer"}
VALIDATION_STEP = 50  #@param {type:"integer"}

OUTPUT_FILENAME = "submission.jsonl"

EXP_DIR = f"claim_verification/e{NUM_EPOCHS}_bs{TRAIN_BATCH_SIZE}_" + f"split=0.1_{LR}_top{EVIDENCE_TOPK}_{MODEL_SHORT}_eval{EVAL_VERSION}_maxlen{MAX_SEQ_LEN}_sr_threshold{SR_THRESHOLD}"
LOG_DIR = "logs/" + EXP_DIR
CKPT_DIR = "checkpoints/" + EXP_DIR

if not Path(LOG_DIR).exists():
    Path(LOG_DIR).mkdir(parents=True)

if not Path(CKPT_DIR).exists():
    Path(CKPT_DIR).mkdir(parents=True)

# Option 1:
filename_train = f"data/claim_verification/split_{TRAIN_TEST_SPLIT}/train_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}_top{EVIDENCE_TOPK}"
filename_dev = f"data/claim_verification/split_{TRAIN_TEST_SPLIT}/dev_doc5sent5_neg0.1_2e-05_e1_hfl_bert_split=0.1_threshold{SR_THRESHOLD}_top{EVIDENCE_TOPK}"
TRAIN_PKL_FILE = Path(filename_train + f".pkl")
TRAIN_JSONL_FILE = filename_train + f".jsonl"
DEV_PKL_FILE = Path(filename_dev + f".pkl")
DEV_JSONL_FILE = filename_dev + f".jsonl"

if not TRAIN_PKL_FILE.exists():
    train_df = join_with_topk_evidence(
        pd.DataFrame(TRAIN_DATA),
        mapping,
        topk=EVIDENCE_TOPK,
    )
    train_df.to_pickle(TRAIN_PKL_FILE, protocol=4)
    train_df.to_json(
        TRAIN_JSONL_FILE,
        orient="records",
        lines=True,
        force_ascii=False,)
else:
    with open(TRAIN_PKL_FILE, "rb") as f:
        train_df = pickle.load(f)

if not DEV_PKL_FILE.exists():
    dev_df = join_with_topk_evidence(
        pd.DataFrame(DEV_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    dev_df.to_pickle(DEV_PKL_FILE, protocol=4)
    dev_df.to_json(
        DEV_JSONL_FILE,
        orient="records",
        lines=True,
        force_ascii=False,)
else:
    with open(DEV_PKL_FILE, "rb") as f:
        dev_df = pickle.load(f)

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = AicupTopkEvidenceBERTDataset(
    train_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
val_dataset = AicupTopkEvidenceBERTDataset(
    dev_df,
    tokenizer=tokenizer,
    max_length=MAX_SEQ_LEN,
)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=0,
)
eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, num_workers=0,)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)
writer = SummaryWriter(LOG_DIR)

### Step 3: Training

progress_bar = tqdm(range(num_training_steps))
current_steps = 0

for epoch in range(REAL_EPOCHS):
    model.train()

    for batch in train_dataloader:
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.sum().backward()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        writer.add_scalar("training_loss", loss.sum().item(), current_steps)

        y_pred = torch.argmax(outputs.logits, dim=1).tolist()
        y_true = batch["labels"].tolist()

        current_steps += 1

        if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
            print(f"Start validation: current_steps={current_steps}, epoch={epoch}")
            val_results = run_evaluation(model, eval_dataloader, device)

            # log each metric separately to TensorBoard
            for metric_name, metric_value in val_results.items():
                print(f"{metric_name}: {metric_value}")
                writer.add_scalar(f"{metric_name}", metric_value, current_steps)

            val_acc = val_results['val_acc']
            if val_acc > 0.65:
                save_checkpoint(
                    model,
                    CKPT_DIR,
                    current_steps,
                    mark=f"val_acc={val_acc:.4f}",
                )

print("Finished training!")

# ### Step 4: Generate Result On Test Data

# ckpt_name = "val_acc=0.6725_model.575.pt"  #@param {type:"string"}

# #### On Private Data
# suffix = "private"
# test_data_name = f"test_doc5sent5_{suffix}_neg0.1_2e-05_e1_hfl_bert"
# TEST_DATA = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}.jsonl")
# TEST_DATA_NO_THRESHOLD = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.jsonl")
# TEST_PKL_FILE = Path(f"data/test_doc5sent5_{suffix}_threshold{SR_THRESHOLD}.pkl")
# TEST_PKL_FILE_NO_THRESHOLD = Path(f"data/test_doc5sent5_{suffix}_threshold{SR_THRESHOLD}_no_threshold.pkl")

# if not TEST_PKL_FILE.exists():
#     test_df = join_with_topk_evidence(
#         pd.DataFrame(TEST_DATA),
#         mapping,
#         mode="eval",
#         topk=EVIDENCE_TOPK,
#     )
#     test_df.to_pickle(TEST_PKL_FILE, protocol=4)
# else:
#     with open(TEST_PKL_FILE, "rb") as f:
#         test_df = pickle.load(f)
# if not TEST_PKL_FILE_NO_THRESHOLD.exists():
#     test_df_nothreshold = join_with_topk_evidence(
#         pd.DataFrame(TEST_DATA_NO_THRESHOLD),
#         mapping,
#         mode="eval",
#         topk=EVIDENCE_TOPK,
#     )
#     test_df_nothreshold.to_pickle(TEST_PKL_FILE_NO_THRESHOLD, protocol=4)
# else:
#     with open(TEST_PKL_FILE_NO_THRESHOLD, "rb") as f:
#         test_df_nothreshold = pickle.load(f)

# test_dataset = AicupTopkEvidenceBERTDataset(
#     test_df,
#     tokenizer=tokenizer,
#     max_length=MAX_SEQ_LEN,
# )
# test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

# torch.cuda.empty_cache()
# model = load_model(model, ckpt_name, CKPT_DIR)
# predicted_label = run_predict(model, test_dataloader, device)

# predict_dataset = test_df_nothreshold.copy()
# predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
# predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
#     f"submission/{suffix}_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}",
#     orient="records",
#     lines=True,
#     force_ascii=False,
# )

# #### On Public Data
# suffix = "public"
# test_data_name = f"test_doc5sent5_{suffix}_neg0.1_2e-05_e1_hfl_bert"
# TEST_DATA = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}.jsonl")
# TEST_DATA_NO_THRESHOLD = load_json(f"data/sent_retrieval/{test_data_name}_threshold{SR_THRESHOLD}_no_threshold.jsonl")
# TEST_PKL_FILE = Path(f"data/test_doc5sent5_{suffix}_threshold{SR_THRESHOLD}.pkl")
# TEST_PKL_FILE_NO_THRESHOLD = Path(f"data/test_doc5sent5_{suffix}_threshold{SR_THRESHOLD}_no_threshold.pkl")

# if not TEST_PKL_FILE.exists():
#     test_df = join_with_topk_evidence(
#         pd.DataFrame(TEST_DATA),
#         mapping,
#         mode="eval",
#         topk=EVIDENCE_TOPK,
#     )
#     test_df.to_pickle(TEST_PKL_FILE, protocol=4)
# else:
#     with open(TEST_PKL_FILE, "rb") as f:
#         test_df = pickle.load(f)
# if not TEST_PKL_FILE_NO_THRESHOLD.exists():
#     test_df_nothreshold = join_with_topk_evidence(
#         pd.DataFrame(TEST_DATA_NO_THRESHOLD),
#         mapping,
#         mode="eval",
#         topk=EVIDENCE_TOPK,
#     )
#     test_df_nothreshold.to_pickle(TEST_PKL_FILE_NO_THRESHOLD, protocol=4)
# else:
#     with open(TEST_PKL_FILE_NO_THRESHOLD, "rb") as f:
#         test_df_nothreshold = pickle.load(f)

# test_dataset = AicupTopkEvidenceBERTDataset(
#     test_df,
#     tokenizer=tokenizer,
#     max_length=MAX_SEQ_LEN,
# )
# test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

# torch.cuda.empty_cache()
# model = load_model(model, ckpt_name, CKPT_DIR)
# predicted_label = run_predict(model, test_dataloader, device)

# predict_dataset = test_df_nothreshold.copy()
# predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
# predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
#     f"submission/{suffix}_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}",
#     orient="records",
#     lines=True,
#     force_ascii=False,
# )

# #### Merge Public & Private Data
# all_data_name = f"submission/all_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}"
# public_data_name = f"submission/public_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}"
# private_data_name = f"submission/private_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}"

# with open(public_data_name, "r", encoding="utf8") as f:
#     data = f.read()
#     f.close()
# with open(private_data_name, "r", encoding="utf8") as f:
#     data2 = f.read()
#     data += data2
#     f.close()
# with open(all_data_name, "w", encoding="utf8") as f:
#     f.write(data)
#     f.close()

# ANSWER_DATA = load_json(all_data_name)
# answer_df = pd.DataFrame(ANSWER_DATA)
# answer_df = answer_df.sort_values("id")
# print(answer_df)
# answer_df.to_json(
#     f"submission/allsorted_{ckpt_name[:14]}_{MODEL_SHORT}_{LR}_{EVAL_VERSION}_e{NUM_EPOCHS}_data3_maxlen{MAX_SEQ_LEN}_{OUTPUT_FILENAME}",
#     orient="records",
#     lines=True,
#     force_ascii=False,
# )
