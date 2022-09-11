from .preprocessor import Preprocessor
import pandas as pd

class User_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        
        
    def __sample_prep_1(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO
        print('loan prep1')
        return df
    
    
    def __sample_prep_2(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO
        print('loan prep2')
        return df
    
    
    def __sample_prep_3(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO
        print('loan prep3')
        return df
    
    
    def _preprocess(self) -> pd.DataFrame:
        prep_df = self.__sample_prep_1(self.raw_df)
        prep_df = self.__sample_prep_2(prep_df)
        prep_df = self.__sample_prep_3(prep_df)
        # 여기서 index sort & reset 필수
        return prep_df