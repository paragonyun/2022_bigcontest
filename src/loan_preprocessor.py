from .preprocessor import Preprocessor
import pandas as pd

class Loan_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        self.time_cols = ['loanapply_insert_time']
        #self.drop_cols = ['product_id']
        self.drop_cols = []
    
    
    def _finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df
    
    
    def _preprocess(self) -> pd.DataFrame:
        prep_df = super()._drop_columns(self.raw_df, self.drop_cols)
        prep_df = super()._to_datetime(prep_df, self.time_cols)
        #prep_df = super()._drop_missing_rows(prep_df)
        prep_df = self._finalize_df(prep_df)
        return prep_df