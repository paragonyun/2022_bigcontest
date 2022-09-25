import pandas as pd

class Matcher():
    
    def __init__(
        self,
        prep_user_df: pd.DataFrame,
        prep_loan_df: pd.DataFrame,
        prep_log_df: pd.DataFrame,
        matcher_config: dict,
        ):
        self.split_timestamp = pd.Timestamp(
            year=matcher_config.get('split_year'),
            month=matcher_config.get('split_month'),
            day=matcher_config.get('split_day')
        )
        self.user_df = prep_user_df
        self.loan_df = prep_loan_df
        self.log_df = prep_log_df
    
    
    def _match(self):
        ml_df = self.loan_df.merge(self.user_df, on='application_id')
        #clustering_df = self.user_df
        return ml_df#, clustering_df
    
        
    def _split_dataset(self, input_df: pd.DataFrame, time_col: str) -> tuple:
        print('Train(Valid)과 Test로 나누는 중...')
        train_valid_df = input_df[input_df[time_col] < self.split_timestamp].reset_index(drop=True)
        test_df = input_df[input_df[time_col] >= self.split_timestamp].reset_index(drop=True)
        return train_valid_df, test_df
    
    
    def run(self):
        ml_df = self._match()
        ml_train_valid_df, ml_test_df = self._split_dataset(ml_df, 'loanapply_insert_time')
        #cl_train_valid_df, cl_test_df = self._split_dataset(clustering_df, '')
        return ml_train_valid_df, ml_test_df#, cl_train_valid_df, cl_test_df