import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split

# 1. Load the raw .txt file
with open("synthetic_se_conversations_100.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 2. Split the conversations (double newlines separate them)
raw_conversations = raw_text.strip().split("\n\n")

# 3. Parse conversations and labels
conversations = []
labels = []

for convo in raw_conversations:
    # Extract the label 
    match = re.search(r'\[(successful|partial|failure)\]', convo)
    if match:
        label = match.group(1)
        # Remove the label from the text
        cleaned_convo = re.sub(r'\[(successful|partial|failure)\]', '', convo).strip()
        # Optional cleaning: Fix extra spaces if any
        cleaned_convo = re.sub(' +', ' ', cleaned_convo)
        conversations.append(cleaned_convo)
        labels.append(label)
    else:
        print("Warning: No label found in a conversation. Skipping.")

# 4. Create a DataFrame
df = pd.DataFrame({
    "conversation": conversations,
    "label": labels
})

# 5. Shuffle the dataset (optional but good practice)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Split into Train and Test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 7. Save cleaned files
train_df.to_csv("train_conversations.csv", index=False)
test_df.to_csv("test_conversations.csv", index=False)

print(f"✅ Done! {len(train_df)} training samples and {len(test_df)} test samples saved.")