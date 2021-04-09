import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import multiprocessing as mp

description = \
    """ Mostly duplicate of `make_subset` with difference of haveing multiple small randomized files for EDA/HP tuning.
Make `N`(default=10) subsets each of `num_size`(default=1000) where both are optional command line arguments,
save each as `gender_final_small_i.csv` for i in N. Do this using all present cores to ensure faster processing.
"""

np.set_printoptions(precision=3)

csv_location = str(Path(__file__).resolve().parents[2] / r'data/gender_final.csv')
small_csv_location = str(Path(__file__).resolve().parents[2] / r'data/subsets/gender_final_small')  # no csv yet.

complete_df = pd.read_csv(csv_location)
complete_df.drop(columns=['Score'], inplace=True)
males_df = complete_df[complete_df.Gender == 1]
females_df = complete_df[complete_df.Gender == 0]
num_males = males_df.Gender.count()
num_females = females_df.Gender.count()


def make_subset(location_small, nums_size):
    print("started ", location_small)
    print(males_df.head(1))
    new_df = pd.concat([
        males_df.sample(n=int(nums_size / 2)),
        females_df.sample(n=int(nums_size / 2))
    ])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    new_df.to_csv(str(location_small), index=False)
    print("ended ", location_small)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('N', nargs='?', default='10', type=int,
                        help='Number of subset csv(s) to be produced, defaults to 10')
    parser.add_argument('num_size', nargs='?', default='1000', type=int,
                        help='Number of rows that each subset will have, defaults to 1000')
    args = parser.parse_args()

    assert num_males > args.num_size / 2 and num_females > args.num_size / 2
    pool = mp.Pool(mp.cpu_count())
    pool_data = ([str(small_csv_location) + str(i) + '.csv', args.num_size]
                 for i in range(1, args.N))
    mapping = pool.starmap_async(make_subset, pool_data)
    mapping.wait()
