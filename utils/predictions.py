import evaluate
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

seqeval = evaluate.load("seqeval")

def clean_pred_label(lbl):
    return lbl.split(";")[0]


def align_predictions(predictions, label_ids, id2label):
    preds = np.argmax(predictions, axis=-1)

    batch_preds = []
    batch_labels = []

    for pred_seq, gold_seq in zip(preds, label_ids):
        p_list = []
        l_list = []

        for p, g in zip(pred_seq, gold_seq):
            if g == -100:
                continue
            p_lbl = clean_pred_label(id2label[int(p)])
            g_lbl = clean_pred_label(id2label[int(g)])
            p_list.append(p_lbl)
            l_list.append(g_lbl)

        batch_preds.append(p_list)
        batch_labels.append(l_list)

    return batch_preds, batch_labels

def bio_to_parseme_tags(bio_tags):
    """
    Convert BIO tags (e.g. B-LVC.full, I-LVC.full, O)
    into official PARSEME MWE tags:
      - O            → "*"
      - B-TYPE       → "1:TYPE"
      - I-TYPE       → "1"
    Supports multiple MWEs via incremental numbering.
    """

    mwe_id_counter = 1
    active_mwes = {}  # mwe_type -> assigned ID
    result = []

    for tag in bio_tags:
        if tag == "O":
            result.append("*")
            continue

        bio, mwe_type = tag.split("-", 1)

        if bio == "B":
            # Start new MWE
            active_mwes[mwe_type] = mwe_id_counter
            result.append(f"{mwe_id_counter}:{mwe_type}")
            mwe_id_counter += 1

        elif bio == "I":
            # Continue existing MWE
            if mwe_type in active_mwes:
                result.append(str(active_mwes[mwe_type]))
            else:
                # Inconsistent segmentation: treat as new MWE
                active_mwes[mwe_type] = mwe_id_counter
                result.append(f"{mwe_id_counter}:{mwe_type}")
                mwe_id_counter += 1

    return result




def fill_cupt_with_predictions(
    model_dir: str,
    input_cupt_path: str,
    output_cupt_path: str,
    id2label: dict
):
    """
    Loads a trained Parseme MWE model and fills the last column of a .cupt file
    with predicted PARSEME-compliant MWE labels (using "*" for O tags).
    """

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Utilities ---------------------------------------------------------

    def read_cupt(path):
        sents = []
        cur = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                if line.strip() == "":
                    if cur:
                        sents.append(cur)
                        cur = []
                elif line.startswith("#"):
                    cur.append(line)
                else:
                    fields = line.rstrip("\n").split("\t")
                    cur.append(fields)
        if cur:
            sents.append(cur)
        return sents

    def write_cupt(sents, path):
        with open(path, "w", encoding="utf8") as f:
            for sent in sents:
                for line in sent:
                    if isinstance(line, str):
                        f.write(line)
                    else:
                        f.write("\t".join(line) + "\n")
                f.write("\n")

    # ------------------------------------------------------------------------

    sents = read_cupt(input_cupt_path)
    print(f"Loaded {len(sents)} sentences.")

    for sent in sents:

        tokens = [fields[1] for fields in sent if isinstance(fields, list)]

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,  # used only for alignment
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        word_ids = encoded.word_ids()

        # convert subword predictions → per-token predictions
        token_preds = []
        last_word = None
        for pred_id, word_idx in zip(pred_ids, word_ids):
            if word_idx is None:
                continue
            if word_idx != last_word:
                token_preds.append(id2label[pred_id])
                last_word = word_idx

        assert len(token_preds) == len(tokens), "Alignment mismatch!"

        # ---- Convert to PARSEME format ------------------------------------
        parseme_tags = bio_to_parseme_tags(token_preds)

        # ---- Write into last column ---------------------------------------
        idx = 0
        for row in sent:
            if isinstance(row, list):
                row[-1] = parseme_tags[idx]
                idx += 1

    write_cupt(sents, output_cupt_path)
    print(f"Wrote predictions to: {output_cupt_path}")
