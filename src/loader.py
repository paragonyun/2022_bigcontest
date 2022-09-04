import pandas as pd
import pickle
import os

class Loader():
    def __init__(self, data_file_path):
        self.path = data_file_path
        _, self.extension = os.path.splitext(self.path)
        self.ext2func = {
            '.csv' : self._load_csv,
            '.ftr' : self._load_feather,
            '.feather' : self._load_feather,
            '.pkl' : self._load_pickle,
            '.pickle' : self._load_pickle,
        }
        
    def _load_csv(self):
        return pd.read_csv(self.path, index_col=None)
    
    def _load_feather(self):
        return pd.read_feather(self.path)
    
    def _load_pickle(self):
        return pd.read_pickle(self.path)
    
    def load(self):
        return self.ext2func[self.extension]()
            