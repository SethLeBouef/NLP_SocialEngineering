import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv("diverse_se_conversations_2000.csv")
df = df.dropna(subset=['conversation', 'label'])

# Merge rare labels
label_mapping = {
    'fail': 'failure',
    'success': 'successful'
}
df['label'] = df['label'].replace(label_mapping)

# Balance the dataset by downsampling each class to the same size
X = df['conversation']
y = df['label']
df_bal = pd.concat([X, y], axis=1)

# Group by class
failure = df_bal[df_bal['label'] == 'failure']
partial = df_bal[df_bal['label'] == 'partial']
successful = df_bal[df_bal['label'] == 'successful']

# Downsample each class to the same size
min_count = min(len(failure), len(partial), len(successful))
failure_down = resample(failure, replace=False, n_samples=min_count, random_state=42)
partial_down = resample(partial, replace=False, n_samples=min_count, random_state=42)
successful_down = resample(successful, replace=False, n_samples=min_count, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([failure_down, partial_down, successful_down])
df_balanced = df_balanced.sample(frac=1, random_state=42)

X = df_balanced['conversation']
y = df_balanced['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorizer with n-grams (unigram + bigram)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n==== {name} ====")
    
    # Train
    model.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Classification report
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    
    # Cross-validation
    print("Cross-validation scores (accuracy):")
    scores = cross_val_score(model, vectorizer.transform(X), y, cv=5, scoring='accuracy')
    print(scores)
    print("Mean CV Accuracy: {:.2f}".format(scores.mean()))