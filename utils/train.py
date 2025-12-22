from .parse import read_cupt, extract_label_set_from
from .data import ParsemeDataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def train_model(train_file,
                dev_file,
                output_dir,
                compute_metrics,
                model_name="xlm-roberta-base"):
    print("Reading data...")
    train_sents = read_cupt(train_file)
    dev_sents = read_cupt(dev_file)

    print("Extracting labels from train+dev (to avoid unseen labels)...")
    labels = extract_label_set_from(train_sents, dev_sents)
    if "O" not in labels:
        labels = ["O"] + labels  # ensure 'O' present
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print("Loading model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    global model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    print("Preparing datasets...")
    train_dataset = ParsemeDataset(train_sents, tokenizer, label2id)
    dev_dataset = ParsemeDataset(dev_sents, tokenizer, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()
    trainer.save_model(output_dir)
    print("Done!")


