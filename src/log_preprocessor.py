from .preprocessor import Preprocessor
import pandas as pd

class Log_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        self.drop_cols = ['mp_os', 'mp_app_version', 'date_cd']
        self.time_cols = ['timestamp']
        self.onehot_cols = ['event']
    
    
    # def _remove_outlier(self, input_df: pd.DataFrame) -> pd.DataFrame:
    #     output_df = input_df.copy()
    #     col_list = output_df.columns.to_list()
    #     for col in col_list:
    #         q1 = output_df[col].quantile(0.25)
    #         q3 = output_df[col].quantile(0.75)
    #         IQR = (q3 - q1)
    #         rev_range = 3  # 제거 범위 조절 변수
    #         output_df = output_df[(output_df[col] <= q3 + (rev_range * IQR)) & (output_df[col] >= q1 - (rev_range * IQR))]
    #         output_df = output_df.reset_index(drop=True)
    #     return output_df
    
    
    def _finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    

    def _preprocess(self) -> pd.DataFrame:
        prep_df = super()._drop_columns(self.raw_df, self.drop_cols)
        prep_df = super()._to_datetime(prep_df, self.time_cols)
        prep_df = self._finalize_df(prep_df)
        return prep_df