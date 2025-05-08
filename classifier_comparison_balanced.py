import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("diverse_se_conversations_2000.csv")
df = df.dropna(subset=['conversation', 'label'])

# Merge label variations
label_mapping = {
    'fail': 'failure',
    'success': 'successful'
}
df['label'] = df['label'].replace(label_mapping)

# Downsample to balance
X = df['conversation']
y = df['label']
df_bal = pd.concat([X, y], axis=1)

failure = df_bal[df_bal['label'] == 'failure']
partial = df_bal[df_bal['label'] == 'partial']
successful = df_bal[df_bal['label'] == 'successful']
min_count = min(len(failure), len(partial), len(successful))

failure_down = resample(failure, replace=False, n_samples=min_count, random_state=42)
partial_down = resample(partial, replace=False, n_samples=min_count, random_state=42)
successful_down = resample(successful, replace=False, n_samples=min_count, random_state=42)

df_balanced = pd.concat([failure_down, partial_down, successful_down])
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Features and split
X = df_balanced['conversation']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# GridSearch for Gradient Boosting
param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train_vec, y_train)

# Best model evaluation
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

y_pred = best_model.predict(X_test_vec)
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized Gradient Boosting Confusion Matrix")
plt.tight_layout()
plt.show()