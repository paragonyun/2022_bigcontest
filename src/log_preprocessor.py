from .preprocessor import Preprocessor
import pandas as pd

class Log_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        self.timestamp_col_str = 'timestamp'
        self.drop_cols_list = ['mp_os', 'mp_app_version', 'date_cd']
        
        
    def __drop_columns(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df = output_df.drop(columns=self.drop_cols_list)
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    
    
    def __to_datetime(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df[self.timestamp_col_str] = pd.to_datetime(output_df[self.timestamp_col_str])
        return output_df
    
    
    # def __remove_outlier(self, input_df: pd.DataFrame) -> pd.DataFrame:
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
    
    
    def __finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    
    
    def _preprocess(self) -> pd.DataFrame:
        prep_df = self.__drop_columns(self.raw_df)
        prep_df = self.__to_datetime(prep_df)
        prep_df = self.__finalize_df(prep_df)
        return prep_df