from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

class mice():
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def mice_1(self):
        df2 = self.df
        
        mice_df = df2[['credit_score','연령대', '근속정도' ,'yearly_income','personal_rehabilitation_yn_0.0', 'personal_rehabilitation_yn_1.0',
       'personal_rehabilitation_yn_nan','existing_loan_cnt', 'existing_loan_amt']]
        
        imputer = IterativeImputer(imputation_order='ascending',max_iter=5,random_state=42)
        imputed_dataset_train = imputer.fit_transform(mice_df)
        train_mice = pd.DataFrame(imputed_dataset_train, columns=mice_df.columns)
        
        credit_score = list(train_mice['credit_score'].apply(lambda x : min(1000, int(x))))
        existing_loan_amt = list(train_mice['existing_loan_amt'])
        age_group = list(train_mice['연령대'])
        work_period = list(train_mice['근속정도'])
        
        df2['mice_credit_score'] = credit_score
        df2['mice_existing_loan_amt'] = existing_loan_amt
        df2['age_group'] = age_group
        df2['work_period'] = work_period
        
        df2 = df2.drop(['근속정도','연령대','credit_score','existing_loan_amt','income_per_credit','existing_loan_percent'], axis=1)
        
        df2['loanamt_per_income'] = df2['mice_existing_loan_amt'] / (1 + df2['yearly_income'])
        df2['loanamt_per_income'].fillna(0, inplace=True)
        
        df2['income_per_credit'] = df2['yearly_income'] / df2['mice_credit_score']
    
        return df2