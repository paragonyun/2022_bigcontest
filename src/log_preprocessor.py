from .preprocessor import Preprocessor
import pandas as pd

class Log_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        self.drop_cols = ['mp_os', 'mp_app_version', 'date_cd']
        self.time_cols = ['timestamp']
    
    
    def _to_categorical(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print('카테고리화 시키는 중...')
        output_df = input_df.astype({
            'user_id' : 'int32', # int64 -> int32
            'event' : 'category', # object -> category
            #'timestamp' : 'datetime64[ns]'
            })
        return output_df
    
    
    def _finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    

    def _preprocess(self) -> pd.DataFrame:
        prep_df = super()._drop_columns(self.raw_df, self.drop_cols)
        prep_df = super()._to_datetime(prep_df, self.time_cols)
        prep_df = self._to_categorical(prep_df)
        prep_df = self._finalize_df(prep_df)
        return prep_df