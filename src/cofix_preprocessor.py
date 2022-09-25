from .preprocessor import Preprocessor
import pandas as pd

class Cofix_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        self.target_date_range_col = '대상기간'
        self.target_date_range_separator = '~'
        self.split_target_date_range_cols = ['대상기간_시작', '대상기간_끝']
        self.time_cols = ['공시일'] + self.split_target_date_range_cols

    
    def _sort_by_timestamp(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print('시간축을 기준으로 정렬 중...')
        output_df = input_df.copy()
        output_df = output_df.sort_values(by=self.time_cols[0])
        return output_df
    
    
    def _split_target_date_range(self, input_df:pd.DataFrame) -> pd.DataFrame:
        print('대상기간 열 나누는 중...')
        output_df = input_df.copy()
        output_df[self.split_target_date_range_cols] = output_df[self.target_date_range_col].str.split(self.target_date_range_separator, expand=True)
        output_df = output_df.drop(columns=self.target_date_range_col)
        return output_df
    
    
    def _rename_cofix_rate_col(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print('COFIX 금리 열 이름 변경 중...')
        output_df = input_df.copy()
        output_df = output_df.rename(columns = {'단기 COFIX': 'cofix_rate'})
        output_df['cofix_rate'] = output_df['cofix_rate'].astype(float)
        return output_df
    
    
    def _finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    

    def _preprocess(self) -> pd.DataFrame:
        prep_df = self._split_target_date_range(self.raw_df)
        prep_df = super()._to_datetime(prep_df, self.time_cols)
        prep_df = self._sort_by_timestamp(prep_df)
        prep_df = self._rename_cofix_rate_col(prep_df)
        prep_df = self._finalize_df(prep_df)
        return prep_df