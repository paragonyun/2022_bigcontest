import pandas as pd   
from typing import Tuple
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


LABEL_COL_NAME = 'is_applied'


def down_sampling(train_valid_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print('Down Sampling 중...')
    # Separate majority and minority classes
    majority_df = train_valid_df[train_valid_df[LABEL_COL_NAME]==0]
    minority_df = train_valid_df[train_valid_df[LABEL_COL_NAME]==1]
    # Downsample majority class
    majority_downsampled_df = resample(
        majority_df, 
        replace=False,
        n_samples=len(minority_df),  # to match minority class
        )
    # Combine minority class with downsampled majority class
    downsampled_df = pd.concat([majority_downsampled_df, minority_df])
    return downsampled_df


def split_train_valid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print('Train과 Valid로 나누는 중...')
    X = df.drop(columns=[LABEL_COL_NAME])
    Y = df[LABEL_COL_NAME]
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def check_imbalanced_label(train_valid_df: pd.DataFrame) -> None:
    n_label_1 = int(train_valid_df[LABEL_COL_NAME].sum())
    n_label_0 = int(len(train_valid_df) - n_label_1)
    print(f'✅ Check imbalanced label ')
    print(f'\t🔹 The number of label_0 : {n_label_0} ({n_label_0/(n_label_0+n_label_1)*100:.2f}%)')
    print(f'\t🔹 The number of label_1 : {n_label_1} ({n_label_1/(n_label_0+n_label_1)*100:.2f}%)')
    return None