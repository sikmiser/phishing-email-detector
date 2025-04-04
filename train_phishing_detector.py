import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ml_projects', 'enron_labeled.csv')
print(f"Loading data from: {data_path}")

if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit(1)

emails = pd.read_csv(data_path)
print("Data loaded. Columns:", emails.columns.tolist())

# Check required columns
if 'text' not in emails.columns or 'label' not in emails.columns:
    print("Error: 'text' or 'label' column missing")
    exit(1)

# Handle NaN
emails['text'] = emails['text'].fillna('')

# Feature Engineering
def extract_subject(text):
    match = re.search(r'Subject: (.*?)\n', text)
    return match.group(1) if match else ''

emails['subject'] = emails['text'].apply(extract_subject)
emails['urgency'] = emails['text'].apply(lambda x: 1 if any(word in x.lower() for word in ['urgent', 'now', 'immediately']) else 0)
emails['url'] = emails['text'].apply(lambda x: 1 if re.search(r'http[s]?://|www\.', x.lower()) else 0)
emails['word_count'] = emails['text'].apply(lambda x: len(x.split()))
emails['combined'] = emails['text'] + ' ' + emails['subject']

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(emails['combined'])
X = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
X['urgency'] = emails['urgency'] * 100  # Stronger amplification
X['url'] = emails['url'] * 100          # Stronger amplification
X['word_count'] = emails['word_count']
y = emails['label']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate with threshold adjustment
results = {}
threshold = 0.3  # Lower threshold for phishing
for name, model in models.items():
    model.fit(X_train, y_train)
    if name == 'Naive Bayes':
        y_pred = model.predict(X_test)
    else:
        # Use predict_proba for threshold adjustment
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

# Print and save results
output_dir = os.path.expanduser('~/Documents/ml_projects')
results_file = os.path.join(output_dir, 'results_v3.txt')
with open(results_file, 'w') as f:
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['Accuracy']:.2f}")
        print(f"Precision: {metrics['Precision']:.2f}")
        print(f"Recall: {metrics['Recall']:.2f}")
        print(f"F1-Score: {metrics['F1']:.2f}")
        print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
        f.write(f"{name} Results:\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.2f}\n")
        f.write(f"Precision: {metrics['Precision']:.2f}\n")
        f.write(f"Recall: {metrics['Recall']:.2f}\n")
        f.write(f"F1-Score: {metrics['F1']:.2f}\n")
        f.write(f"Confusion Matrix:\n{metrics['Confusion Matrix']}\n\n")

# Predict a new email
new_email_text = 'urgent act now get rich quick http://example.com'
new_email = vectorizer.transform([new_email_text]).toarray()
new_features = pd.DataFrame(new_email, columns=vectorizer.get_feature_names_out())
new_features['urgency'] = 1 * 100  # Match amplification
new_features['url'] = 1 * 100      # Match amplification
new_features['word_count'] = len(new_email_text.split())
for name, model in models.items():
    if name == 'Naive Bayes':
        pred = model.predict(new_features)[0]
    else:
        prob = model.predict_proba(new_features)[0, list(model.classes_).index('phishing')]
        pred = 'phishing' if prob >= threshold else 'safe'
    print(f"{name} Prediction for '{new_email_text}': {pred}")

# Save predictions
predictions_df = pd.DataFrame({
    'text': emails['text'].iloc[X_test.index],
    'label': y_test,
    'Naive Bayes': results['Naive Bayes']['Predictions'],
    'Logistic Regression': results['Logistic Regression']['Predictions'],
    'Random Forest': results['Random Forest']['Predictions']
})
predictions_file = os.path.join(output_dir, 'enron_predictions_v3.csv')
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
plt.savefig(os.path.join(output_dir, 'confusion_matrices_v3.png'))
print("Confusion matrices saved to: confusion_matrices_v3.png")

counts = emails['label'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color=['blue', 'red'])
plt.title("Phishing vs. Safe Emails")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, 'email_counts_v3.png'))
print("Email counts saved to: email_counts_v3.png")