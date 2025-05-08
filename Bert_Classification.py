import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from evaluate import load
import matplotlib.pyplot as plt
import os
import random

# === Reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === Device Detection ===
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# === Load Dataset ===
df = pd.read_csv("diverse_se_conversations_2000.csv")
df = df.dropna(subset=['conversation', 'label'])
df["label"] = df["label"].str.strip().str.lower()
df["label"] = df["label"].replace({"fail": "failure", "success": "successful"})

# === Encode Labels ===
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
label_list = list(label_encoder.classes_)
print("Classes:", label_list)

# === Compute Class Weights ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label_id"]),
    y=df["label_id"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights)

# === Train/Test Split ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["conversation"].tolist(),
    df["label_id"].tolist(),
    stratify=df["label_id"],
    test_size=0.2,
    random_state=42
)

# === Tokenize ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# === Dataset Class ===
class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ChatDataset(train_encodings, train_labels)
val_dataset = ChatDataset(val_encodings, val_labels)

# === Load Model ===
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_list))
model.to(device)

# === Custom Trainer ===
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# === Evaluation Metrics ===
accuracy_metric = load("accuracy")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    do_train=True,
    do_eval=True
)

# === Train Model ===
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

train_output = trainer.train()
eval_results = trainer.evaluate()
print("📊 Final Evaluation Metrics:", eval_results)

# === Plot Training Loss ===
losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Logged Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# === Make Predictions ===
predictions = trainer.predict(val_dataset)
logits = predictions.predictions
predicted_classes = np.argmax(logits, axis=1)

# === Show Sample Predictions ===
print("\n🔎 Sample Predictions on Validation Set:")
for i in range(10):
    print(f"Text: {val_texts[i][:100]}...")
    print(f"True Label: {label_list[val_labels[i]]}")
    print(f"Predicted : {label_list[predicted_classes[i]]}")
    print("—" * 50)

# === Save Model and Tokenizer ===
save_dir = "./saved_roberta_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\n✅ Model and tokenizer saved to {save_dir}")