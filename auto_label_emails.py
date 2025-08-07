import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
labeled_path = os.path.join(os.path.dirname(__file__), 'enron_labeled.csv')
sample_path = os.path.join(os.path.dirname(__file__), 'enron_sample.csv')
output_path = os.path.join(os.path.dirname(__file__), 'enron_auto_labeled.csv')

# Check files exist
if not os.path.exists(labeled_path):
    print(f"Error: {labeled_path} not found.")
    exit(1)
if not os.path.exists(sample_path):
    print(f"Error: {sample_path} not found.")
    exit(1)

# Load data
try:
    labeled = pd.read_csv(labeled_path)
    sample = pd.read_csv(sample_path)
    print(f"Labeled data: {len(labeled)} emails, Sample data: {len(sample)} emails")
    print("Labeled columns:", labeled.columns.tolist())
    print("Sample columns:", sample.columns.tolist())
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Feature engineering
try:
    labeled['text'] = labeled['text'].fillna('')
    sample['message'] = sample['message'].fillna('')
except KeyError as e:
    print(f"KeyError: {e}")
    exit(1)

vectorizer = CountVectorizer(stop_words='english')
X_labeled = vectorizer.fit_transform(labeled['text'])
X_sample = vectorizer.transform(sample['message'])

labeled['urgency'] = labeled['text'].apply(lambda x: 1 if any(word in x.lower() for word in ['urgent', 'now', 'immediately']) else 0) * 100
labeled['url'] = labeled['text'].apply(lambda x: 1 if re.search(r'http[s]?://|www\.', x.lower()) else 0) * 100
sample['urgency'] = sample['message'].apply(lambda x: 1 if any(word in x.lower() for word in ['urgent', 'now', 'immediately']) else 0) * 100
sample['url'] = sample['message'].apply(lambda x: 1 if re.search(r'http[s]?://|www\.', x.lower()) else 0) * 100

X_labeled_df = pd.DataFrame(X_labeled.toarray(), columns=vectorizer.get_feature_names_out())
X_labeled_df['urgency'] = labeled['urgency']
X_labeled_df['url'] = labeled['url']
X_sample_df = pd.DataFrame(X_sample.toarray(), columns=vectorizer.get_feature_names_out())
X_sample_df['urgency'] = sample['urgency']
X_sample_df['url'] = sample['url']

# Train on labeled data
model = LogisticRegression(max_iter=1000)
model.fit(X_labeled_df, labeled['label'])

# Predict with threshold
threshold = 0.5
probs = model.predict_proba(X_sample_df)[:, list(model.classes_).index('phishing')]
predictions = ['phishing' if p >= threshold else 'safe' for p in probs]

# Save auto-labeled dataset
auto_labeled = pd.DataFrame({'text': sample['message'], 'label': predictions})
auto_labeled.to_csv(output_path, index=False)
print(f"Auto-labeled {len(auto_labeled)} emails saved to: {output_path}")
print("Label distribution:\n", auto_labeled['label'].value_counts())
print("First 5 predictions:")
print(auto_labeled.head())