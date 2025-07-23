import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv('AI_Human.csv')

# Shuffle and stratified split
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
dev, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

# Save
train.to_csv('train.csv', index=False)
dev.to_csv('dev.csv', index=False)
test.to_csv('test.csv', index=False)

print("Files created: train.csv, dev.csv, test.csv")