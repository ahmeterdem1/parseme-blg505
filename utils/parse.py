
def read_cupt(path):
    """
    Returns a list of sentences.
    Each sentence is a list of token dicts:
    {
        "id": "3",
        "form": "ran",
        "lemma": "_",
        ...
        "mwe": "1:VID"
    }
    """
    sentences = []
    current = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if "-" in cols[0]:  # multi-word token line
                continue

            tok = {
                "id": cols[0],
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "feats": cols[5],
                "head": cols[6],
                "deprel": cols[7],
                "deps": cols[8],
                "misc": cols[9],
                "mwe": cols[10]
            }
            current.append(tok)

    if current:
        sentences.append(current)

    return sentences


def parse_mwe_column(col):
    """
    Handle:
        "_", "*"           → empty
        "1"                → continuation of MWE 1 (no type)
        "1:VID"            → start of MWE 1 type VID
        "1;2"              → continuation of both
        "1;2:VID"          → mix
    """
    if col in ("_", "*", ""):
        return []

    entries = []
    for part in col.split(";"):
        if ":" in part:
            num, typ = part.split(":")
            entries.append((int(num), typ))
        else:
            entries.append((int(part), None))
    return entries


def normalize_label(lbl):
    """
    Enforce single label:
    "B-LVC.full;B-VID" → "B-LVC.full"
    """
    return lbl.split(";")[0]


def cupt_to_bio(sent):
    """
    Converts one sentence (list of tokens) → list of BIO labels.
    Handles multi-MWE, continuations, etc.
    Always returns ONE label per token (normalized).
    """

    # collect known types (first non-None determines type)
    mwe_types = {}
    for tok in sent:
        for mwe_id, typ in parse_mwe_column(tok["mwe"]):
            if typ is not None:
                mwe_types[mwe_id] = typ

    tags = []
    started = {}

    for tok in sent:
        entries = parse_mwe_column(tok["mwe"])

        if not entries:
            tags.append("O")
            continue

        token_labels = []
        for mwe_id, typ in entries:
            the_type = mwe_types.get(mwe_id, "UNK")

            if mwe_id not in started:
                token_labels.append(f"B-{the_type}")
                started[mwe_id] = True
            else:
                token_labels.append(f"I-{the_type}")

        # enforce single selection
        final = normalize_label(";".join(token_labels))
        tags.append(final)

    return tags

def extract_label_set(train_sents):
    labels = set()
    for sent in train_sents:
        bio = cupt_to_bio(sent)
        for t in bio:
            labels.add(normalize_label(t))
    return sorted(labels)

def extract_label_set_from(*sentence_lists):
    """
    Build the sorted set of labels (BIO single-label normalized) found
    across all provided sentence lists.

    Each argument is a list of sentences (as returned by read_cupt).
    """
    labels = set()
    for sents in sentence_lists:
        for sent in sents:
            bio = cupt_to_bio(sent)
            for t in bio:
                labels.add(normalize_label(t))
    return sorted(labels)

