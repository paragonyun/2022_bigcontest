from .preprocessor import Preprocessor
import pandas as pd

## Emcoders
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder 

## datetime
from datetime import datetime

class User_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        ## onehot encoder로 바뀔 cols
        self.onehot_cols = ['gender','income_type','employment_type','houseown_type','purpose',
                            'personal_rehabilitation_yn','personal_rehabilitation_complete_yn']

        ## ordinal encoder로 바뀔 cols
        self.ordinal_cols = ['birth_year','yearly_income']

        ## datetime type으로 바뀔 cols
        self.time_cols = ['insert_time', 'company_enter_month', 'birth_year']

        ## 범주화될 cols
        self.categorical_cols = ['birth_year','yearly_income']

    '''
    전처리 순서 : 
    1. datetime type 대상 cols 전처리
    2. 파생변수 만들기
    3. 범주화 대상 cols 범주화 전처리
    4. OneHot Encoder 대상 cols 전처리
    5. Ordinal Encoder 대상 cols 전처리
    '''
    def _to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        print('datetime으로 바꾸는 중...')
        output_df = df.copy()

        def _cuting (x) :
            return str(x)[:6]

        ## datetime 대상이 될 cols를 돌면서
        for i in self.time_cols :
            
            if i == 'birth_year' :

                output_df[i] = pd.to_datetime(output_df[i], format='%Y', errors='ignore')
                continue
            
            elif i == 'company_enter_month' :
                output_df['company_enter_month'] = output_df['company_enter_month'].apply(lambda x : _cuting(x))
                output_df['company_enter_month'] = pd.to_datetime(output_df['company_enter_month'], format='%Y%m')
                continue


            ## 해당 column을 datetime type으로 바꿔줌
            output_df[i] = pd.to_datetime(output_df[i])

        ## 그렇게 바뀐 df return 
        return output_df

    def _derived_variable_maker(self, df: pd.DataFrame) :
        print('파생변수 생성 중...')
        output_df = df.copy()
        ## 신용점수 대비 연소득 : 연소득/신용점수
        output_df['income_per_credit'] = output_df['yearly_income'] / output_df['credit_score']

        ## 기대출비율 : 기대출금액 / 연소득
        output_df['existing_loan_percent'] = output_df['existing_loan_amt'] / output_df['yearly_income']

        return output_df
    
    
    def _to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        print('카테고리화 시키는 중...')
        ## Birth 카테고리컬 화
        # 이미 생년월일은 datetime type으로 변환된 상태임
        def _calculate_age(x) :
            this_yaer = datetime.now().year
            return this_yaer - x.year

        def _birth_category(x) :
            if 0 <= x < 10 :
                return '0대'
            elif 10<= x <20 :
                return '10대'
            elif 20<= x <30 :
                return '20대'
            elif 30<= x <40 :
                return '30대'
            elif 40<= x <50 :
                return '40대' 
            elif 50<= x <60 :
                return '50대'
            elif 60<= x <70 :
                return '60대'
            else :
                return '70대 이상'

        output_df = df.copy()
        # 나이로 일단 변환
        output_df['age'] = output_df['birth_year'].apply(lambda x : _calculate_age(x))

        # 연령대로 변환
        output_df['age_cat'] = output_df['age'].apply(lambda x : _birth_category(x))

        ## yearly_income 카테고리 화 
        def _income_category(x) :
            
            if x < 5510000 :
                return '1'
            elif 5510000 <= x <18440000 :
                return '2'
            elif 18440000 <= x <32260000 :
                return '3'
            elif 32260000 <= x < 42970000 :
                return '4'
            elif 42970000 <= x < 50880000 :
                return '5'
            elif 50880000 <= x < 61690000 :
                return '6'
            elif 61690000 <= x < 75790000 :
                return '7'
            elif 75790000 <= x < 96130000 :
                return '8'
            elif 96130000 <= x < 128850000 :
                return '9'
            elif 128850000 <= x  :
                return '10'

            else :
                return '0'


        output_df['yearly_income_cat'] = output_df['yearly_income'].apply(lambda x : _income_category(x))

        return output_df

    def _to_one_hot(self, df) :
        print('원핫인코딩 중...')
        output_df = df.copy()
        for i in self.onehot_cols :
            ohe = OneHotEncoder(sparse=False)
            ohe_df = pd.DataFrame(ohe.fit_transform(output_df[[i]]),
                                columns = [f'{i}_'+ str(col) for col in ohe.categories_[0]])
            output_df = pd.concat([output_df.drop(columns=[i]),
                                ohe_df], axis=1)


        return output_df

    def _to_ordinal(self, df) :
        print('순서형인코딩 중...')
        output_df = df.copy()

        ## 그냥 sklearn의 Ordinal Encoder를 쓰면 순서 상관없이 바꿔버립니다..
        ## 그래서 수동으로 그냥 함수 작성해서 apply 해줬습니다.

        def _birth_ordinal(x) :
            if x == '0대' :
                return 0
            elif x == '10대' :
                return 1
            elif x == '20대' :
                return 2
            elif x == '30대' :
                return 3
            elif x == '40대' :
                return 4
            elif x == '50대' :
                return 5
            elif x == '60대' :
                return 6
            else :
                return 7

        def _income_ordinal(x) :
            if x == '1' :
                return 1 
            elif x == '2' :
                return 2
            elif x == '3' :
                return 3
            elif x == '4' :
                return 4
            elif x == '5' :
                return 5
            elif x == '6' :
                return 6
            elif x == '7' :
                return 7 
            elif x == '8' :
                return 8
            elif x == '9' :
                return 9
            elif x == '10' :
                return 10
            else :
                return 0

        output_df['연령대'] = output_df['age_cat'].apply(lambda x : _birth_ordinal(x))
        output_df['소득분위'] = output_df['yearly_income_cat'].apply(lambda x : _income_ordinal(x))

        ## 필요 없는 columns 삭제
        output_df = output_df.drop(['birth_year', 'age','age_cat'], axis=1)

        return output_df

    def _finalize_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df.copy()
        output_df.reset_index(drop=True, inplace=True)
        return output_df

    def _preprocess(self) -> pd.DataFrame:
        prep_df = self._to_datetime(self.raw_df) # self.raw_df 는 preprocessor.py 에서 상속받음
        prep_df = self._derived_variable_maker(prep_df) # 파생변수 생성
        prep_df = self._to_categorical(prep_df) # 범주화
        prep_df = self._to_one_hot(prep_df) # 원핫
        prep_df = self._to_ordinal(prep_df) # ordinal
        prep_df = self._finalize_df(prep_df) # 인덱스 리셋

        return prep_df