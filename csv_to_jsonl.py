import pandas as pd

for fname in ['train.csv', 'dev.csv', 'test.csv']:
    df = pd.read_csv(fname)
    df.to_json(fname.replace('.csv', '.jsonl'), orient='records', lines=True)
    print(f"Converted {fname} to {fname.replace('.csv', '.jsonl')}")
    