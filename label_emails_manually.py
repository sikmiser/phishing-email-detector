import os
import pandas as pd

file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ml_projects', 'enron_sample.csv')
print(f"Looking for file at: {file_path}")

try:
    df = pd.read_csv(file_path, engine='python')
    print("CSV loaded successfully. Columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: Could not find {file_path}")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

if 'message' not in df.columns:
    print("Error: 'message' column not found")
    exit(1)

sample_emails = df['message'].iloc[:50].fillna('').tolist()
labels = ['safe'] * 35 + ['phishing'] * 15

print("Number of emails:", len(sample_emails))
print("First email (raw):", sample_emails[0][:100])

emails = pd.DataFrame({'text': sample_emails, 'label': labels})
emails.to_csv(os.path.join(os.path.expanduser('~'), 'Documents', 'ml_projects', 'enron_labeled.csv'), index=False)
print("First 5 emails and labels:")
print(emails.head())