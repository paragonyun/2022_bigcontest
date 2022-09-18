import pandas as pd
import os

## Encoders
from sklearn.preprocessing import  OneHotEncoder

class Preprocessor():
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        # TODO
        self.save_dir = prep_config.get('save_dir')
        self.raw_df = dataset
    
    
    def _drop_columns(self, input_df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        print('필요 없는 열 삭제 중...')
        output_df = input_df.copy()
        if len(drop_cols) != 0:
            output_df = output_df.drop(columns=drop_cols)
            output_df.reset_index(drop=True, inplace=True)
        return output_df
    
    
    def _to_datetime(self, input_df: pd.DataFrame, time_cols: list) -> pd.DataFrame:
        print('datetime으로 바꾸는 중...')
        output_df = input_df.copy()
        if len(time_cols) != 0:
            for time_col in time_cols: 
                output_df[time_col] = pd.to_datetime(output_df[time_col])
        return output_df
    
    
    def _to_one_hot(self, input_df: pd.DataFrame, onehot_cols: list) -> pd.DataFrame:
        print('원핫인코딩 중...')
        output_df = input_df.copy()
        if len(onehot_cols) != 0:
            for col in self.onehot_cols :
                ohe = OneHotEncoder(sparse=False)
                ohe_df = pd.DataFrame(ohe.fit_transform(output_df[[col]]),
                                    columns = [f'{col}_{str(col)}' for col in ohe.categories_[0]])
                output_df = pd.concat([output_df.drop(columns=[col]),
                                    ohe_df], axis=1)
        return output_df
    
    
    def _drop_missing_rows(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print('결측치를 가지는 행 삭제 중...')
        output_df = input_df.copy()
        output_df = input_df.dropna(axis=0)
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    
         
    def _save_preprocessed_df(self, prep_df:pd.DataFrame, save_file_name: str) -> None:
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_file_name)
        prep_df.to_feather(save_path)
        print(f'✅ prep dataset saved at ({save_path})')
    
    
    def _preprocess(self):
        # empty function for override
        ...
    
    
    def run(self, save_file_name: str, save_mode=True):
        prep_df = self._preprocess()
        if save_mode:
            self._save_preprocessed_df(prep_df, save_file_name)
        return prep_df