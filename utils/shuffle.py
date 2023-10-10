import pandas as pd
#Script for shuffling training data, because windows decided to sort all of training data alpahabetically :))))))))) 
df = pd.read_csv('normalized_file.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('shuffled_csv_file.csv', index=False)
