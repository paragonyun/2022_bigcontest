import pandas as pd
import os

class Preprocessor():
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        # TODO
        self.save_dir = prep_config.get('save_dir')
        self.raw_df = dataset
        
         
    def _save_preprocessed_df(self, prep_df:pd.DataFrame, save_file_name: str) -> None:
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_file_name)
        print(f'🚫 Error may be in here')
        prep_df.to_feather(save_path)
        print(f'✅ prep dataset saved ({save_path})')
    
    
    def _preprocess(self):
        # empty function for override
        ...
    
    
    def run(self, save_file_name: str, save_mode=True):
        prep_df = self._preprocess()
        if save_mode:
            self._save_preprocessed_df(prep_df, save_file_name)
        return prep_df