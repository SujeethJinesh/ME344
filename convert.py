import json
import pandas as pd

# Read the JSON file
with open('gen_z_slangs_translation.json', 'r') as f:
    data = json.load(f)

# Flatten the nested structure
flattened_data = []
for item in data:
    human = item[0]['value']
    gpt = item[1]['value']
    flattened_data.append({'human': human, 'gpt': gpt})

# Create a DataFrame
df = pd.DataFrame(flattened_data)

# Save as parquet
df.to_parquet('gen_z_slangs_translation.parquet')

print("Parquet file created successfully.")
