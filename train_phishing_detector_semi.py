import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ml_projects', 'enron_auto_labeled.csv')
print(f"Loading data from: {data_path}")
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit(1)
emails = pd.read_csv(data_path)
print("Data loaded. Columns:", emails.columns.tolist())
if 'text' not in emails.columns or 'label' not in emails.columns:
    print("Error: 'text' or 'label' column missing")
    exit(1)
emails['text'] = emails['text'].fillna('')

# Feature Engineering
def extract_subject(text):
    match = re.search(r'Subject: (.*?)\n', text)
    return match.group(1) if match else ''
emails['subject'] = emails['text'].apply(extract_subject)
emails['urgency'] = emails['text'].apply(lambda x: 1 if any(word in x.lower() for word in ['urgent', 'now', 'immediately']) else 0) * 100
emails['url'] = emails['text'].apply(lambda x: 1 if re.search(r'http[s]?://|www\.', x.lower()) else 0) * 100
emails['phish_keywords'] = emails['text'].apply(lambda x: sum(x.lower().count(word) for word in ['urgent', 'act', 'quick']) > 0) * 100  # New feature
emails['word_count'] = emails['text'].apply(lambda x: len(x.split()))
emails['combined'] = emails['text'] + ' ' + emails['subject']

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(emails['combined'])
X = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
X['urgency'] = emails['urgency']
X['url'] = emails['url']
X['phish_keywords'] = emails['phish_keywords']
X['word_count'] = emails['word_count']
y = emails['label']

# Split for single-run metrics and prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate with cross-validation
results = {}
threshold = 0.1
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{name} Results:")
    print(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, list(model.classes_).index('phishing')]
    y_pred = ['phishing' if p >= threshold else 'safe' for p in y_prob]
    results[name] = {
        'Accuracy': (y_pred == y_test).mean(),
        'Precision': precision_score(y_test, y_pred, pos_label='phishing', zero_division=0),
        'Recall': recall_score(y_test, y_pred, pos_label='phishing', zero_division=0),
        'F1': f1_score(y_test, y_pred, pos_label='phishing', zero_division=0),
        'Confusion Matrix': confusion_matrix(y_test, y_pred, labels=['safe', 'phishing']),
        'Predictions': y_pred
    }
    print(f"Single-Run Accuracy: {results[name]['Accuracy']:.2f}")
    print(f"Precision: {results[name]['Precision']:.2f}")
    print(f"Recall: {results[name]['Recall']:.2f}")
    print(f"F1-Score: {results[name]['F1']:.2f}")
    print(f"Confusion Matrix:\n{results[name]['Confusion Matrix']}")

# Predict a new email
# Predict a new email
new_email_text = 'urgent act now get rich quick http://example.com'
new_email = vectorizer.transform([new_email_text]).toarray()
new_features = pd.DataFrame(new_email, columns=vectorizer.get_feature_names_out())
new_features['urgency'] = 1 * 100
new_features['url'] = 1 * 100
new_features['phish_keywords'] = 1 * 100
new_features['word_count'] = len(new_email_text.split())
for name, model in models.items():
    prob = model.predict_proba(new_features)[0, list(model.classes_).index('phishing')]
    pred = 'phishing' if prob >= 0.05 else 'safe'  # Use 0.05 threshold
    print(f"{name} Prediction for '{new_email_text}': {pred} (Phishing prob: {prob:.4f})")

# Save predictions
predictions_df = pd.DataFrame({
    'text': emails['text'].iloc[X_test.index],
    'label': y_test,
    'Naive Bayes': results['Naive Bayes']['Predictions'],
    'Logistic Regression': results['Logistic Regression']['Predictions'],
    'Random Forest': results['Random Forest']['Predictions']
})
output_dir = os.path.expanduser('~/Documents/ml_projects')
predictions_file = os.path.join(output_dir, 'enron_predictions_v3_semi.csv')
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved to: {predictions_file}")

# Visualizations
plt.figure(figsize=(15, 5))
for i, (name, metrics) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['safe', 'phishing'], yticklabels=['safe', 'phishing'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices_v3_semi.png'))
print("Confusion matrices saved to: confusion_matrices_v3_semi.png")

counts = emails['label'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color=['blue', 'red'])
plt.title("Phishing vs. Safe Emails")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, 'email_counts_v3_semi.png'))
print("Email counts saved to: email_counts_v3_semi.png")