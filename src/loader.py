import pandas as pd
import os

class Loader():
    def __init__(self, loader_config):
        self.data_dir = loader_config.get('data_dir')
        self.ext2func = {
            '.csv' : self._load_csv,
            '.ftr' : self._load_feather,
            '.feather' : self._load_feather,
            '.pkl' : self._load_pickle,
            '.pickle' : self._load_pickle,
        }
        
    def _load_csv(self, data_path) -> pd.DataFrame:
        return pd.read_csv(data_path, index_col=None)
    
    def _load_feather(self, data_path) -> pd.DataFrame:
        return pd.read_feather(data_path)
    
    def _load_pickle(self, data_path) -> dict:
        return pd.read_pickle(data_path)
    
    def _get_extension(self, file_name: str) -> str:
        _, extension = os.path.splitext(file_name)
        return extension
    
    def run(self, data_file_name: str):
        data_ext = self._get_extension(data_file_name)
        data_path = os.path.join(self.data_dir, data_file_name)
        load_df = self.ext2func[data_ext](data_path)
        return load_df
            