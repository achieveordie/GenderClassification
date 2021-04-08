import pandas as pd
import numpy as np
from pathlib import Path

np.set_printoptions(precision=3)

csv_location = str(Path(__file__).resolve().parents[2] / r'data/gender_final.csv')
small_csv_location = (r"gender_final_small.csv",  # save here for remote reference and one in data directory
                      str(Path(__file__).resolve().parents[2] / r'data/gender_final_small.csv'))
new_data_size = 1000  # only have 1000 names, 50% of each type.

complete_df = pd.read_csv(csv_location)
complete_df.drop(columns=['Score'], inplace=True)
males_df = complete_df[complete_df.Gender == 1]
females_df = complete_df[complete_df.Gender == 0]
num_males = males_df.Gender.count()
num_females = females_df.Gender.count()

assert num_males > new_data_size/2 and num_females > new_data_size/2

new_df = pd.concat([
    males_df.sample(n=int(new_data_size/2)),
    females_df.sample(n=int(new_data_size/2))
], axis=0, ignore_index=True)

new_df = new_df.sample(frac=1).reset_index(drop=True)  # reset index and drop original index
[new_df.to_csv(i, index=False) for i in small_csv_location]
