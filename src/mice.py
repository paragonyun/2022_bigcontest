from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

class mice():
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def mice_1(self):
        mice_df = self.df[['credit_score','연령대', '소득분위','personal_rehabilitation_yn_0.0', 'personal_rehabilitation_yn_1.0',
       'personal_rehabilitation_yn_nan','existing_loan_cnt', 'existing_loan_amt']]
        
        imputer = IterativeImputer(imputation_order='ascending',max_iter=5,random_state=42)
        imputed_dataset_train = imputer.fit_transform(mice_df)
        train_mice = pd.DataFrame(imputed_dataset_train, columns=mice_df.columns)
        
        credit_score = list(train_mice['credit_score'].apply(lambda x : min(1000, int(x))))
        existing_loan_amt = list(train_mice['existing_loan_amt'])
        
        self.df['mice_credit_score'] = credit_score
        self.df['mice_existing_loan_amt'] = existing_loan_amt
        
        self.df = self.df.drop(['credit_score','existing_loan_amt','income_per_credit','existing_loan_percent'], axis=1)
        
        self.df['loanamt_per_income'] = self.df['mice_existing_loan_amt'] / (1 + self.df['yearly_income'])
        self.df['loanamt_per_income'].fillna(0, inplace=True)
        
        self.df['income_per_credit'] = self.df['yearly_income'] / self.df['credit_score']
    
        return self.df