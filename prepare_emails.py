import os
import csv

input_file = os.path.expanduser('~/Documents/ml_projects/emails.csv')
output_file = os.path.expanduser('~/Documents/ml_projects/enron_sample.csv')

emails = []
current_email = {'file': '', 'message': ''}

with open(input_file, 'r', encoding='utf-8') as f:
    next(f)  # Skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',', 1)
        if len(parts) != 2:
            continue
        file_part, msg_part = parts
        if file_part.startswith('"') and not file_part == '""':
            if current_email['message']:
                emails.append([current_email['file'], current_email['message'].strip()])
                if len(emails) >= 100:
                    break
            current_email['file'] = file_part
            current_email['message'] = msg_part.strip('"')
        else:
            current_email['message'] += '\n' + msg_part.strip('"')

    if current_email['message'] and len(emails) < 100:
        emails.append([current_email['file'], current_email['message'].strip()])

# Write with minimal escaping
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['file', 'message'])
    writer.writerows(emails)

print("Sample CSV created at:", output_file)
print("Number of emails:", len(emails))
print("First email (raw):", emails[0][1][:100])