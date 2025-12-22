from .parse import normalize_label, cupt_to_bio

class ParsemeDataset:
    def __init__(self, sentences, tokenizer, label2id, max_length=256):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.max_length = max_length
        self._unknown_label_warned = False  # single-time warning

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words = [tok["form"] for tok in sent]
        labels = [normalize_label(t) for t in cupt_to_bio(sent)]

        encoding = self.tokenizer(
            words,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # align BIO labels to wordpieces
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word = None

        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            else:
                if wid != prev_word:
                    lab = labels[wid]
                    if lab not in self.label2id:
                        # fallback: map unseen label to "O"
                        if not self._unknown_label_warned:
                            print(f"Warning: unseen label '{lab}' encountered — mapping to 'O'.")
                            self._unknown_label_warned = True
                        lab = "O"
                    label_ids.append(self.label2id[lab])
                else:
                    label_ids.append(-100)
            prev_word = wid

        encoding.pop("offset_mapping")
        encoding["labels"] = label_ids
        return encoding
