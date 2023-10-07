import pandas as pd

# Read the CSV file
df = pd.read_csv('normalized_file.csv')

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataframe back to CSV
df.to_csv('shuffled_csv_file.csv', index=False)
