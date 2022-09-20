from tkinter import Grid
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import seaborn as sns



class RF():
# {'n_estimators': list(np.linspace(100, 5000, 50)),
#                          'max_depth': list(range(3,11)),
#                          'min_samples_leaf': [10,15,20]}
    
    def __init__(self, train_X, train_Y, val_X, val_Y, test_X, test_Y, random_state):
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.params = {'n_estimators': [100, 200, 300],
                         'max_depth': [3,4,5],
                         'min_samples_leaf': [4,6,8]}
        self.random_state = random_state
        
        
    def grid_search(self):
        train_X = self.train_X
        train_Y = self.train_Y
        val_X = self.val_X
        val_Y = self.val_Y
        params = self.params
        
        rf_model = RandomForestClassifier()
        rf_grid = RandomizedSearchCV(rf_model, param_distributions= params, n_iter=5, scoring="f1", cv=3, refit=True, random_state=self.random_state)
        rf_grid.fit(train_X, train_Y)
        
        print('best parameters : ', rf_grid.best_params_)
        print('best val score : ', round(rf_grid.best_score_, 4))
        
        best_model = rf_grid.best_estimator_
        
        return best_model
    
    def test_score(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        pred = model.predict(test_X)
        F1_score = f1_score(test_Y, pred)
        print("f1_score: ", F1_score)
        print(classification_report(test_Y, pred))
        
    def confusion_matrix(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        label=['0', '1']
        plot = plot_confusion_matrix(model,
                                    test_X, test_Y, 
                                    display_labels=label,
                                    # cmap=plt.cm.Blue,  (plt.cm.Reds, plt.cm.rainbow)
                                    normalize=None) # 'true', 'pred', 'all', default=None
        plot.ax_.set_title('Confusion Matrix')
        plt.show()
        
    def feature_importance(self, model):
        ftr_importances_values = model.feature_importances_
        ftr_importances = pd.Series(ftr_importances_values, index = self.test_X.columns)
        ftr_importances = ftr_importances.sort_values(ascending=False)
        
        plt.figure(figsize=(8,6))
        plt.title('Feature Importances')
        sns.barplot(x=ftr_importances, y=ftr_importances.index)
        plt.show()
        
        
    def shap(self, model):
        test_X = self.test_X
        
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(test_X)
        
        shap.summary_plot(shap_values, test_X, plot_type = "bar")
        plt.show()
        
        
class XGB():
# {'n_estimators': list(np.linspace(100, 5000, 50)),
#                          'learning_rate': [0.2, 0.1, 0.05, 0.01],
#                          'max_depth': list(range(3,11)),
#                          'gamma': list(range(5)),
#                          'colsample_bytree':[0.5, 0.75, 1]}
    
    def __init__(self, train_X, train_Y, val_X, val_Y, test_X, test_Y, random_state):
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.params = {'n_estimators': [100, 200],
                         'learning_rate': [0.2, 0.1],
                         'max_depth': [3,4],
                         'gamma': [0,1],
                         'colsample_bytree':[0.5, 0.75]}
        self.random_state = random_state
        
    def grid_search(self):
        train_X = self.train_X
        train_Y = self.train_Y
        val_X = self.val_X
        val_Y = self.val_Y
        params = self.params
        
        xgb = XGBClassifier()
        xgb_grid = RandomizedSearchCV(xgb, param_distributions= params, n_iter=5, scoring="f1", cv=3, refit=True, random_state=self.random_state)
        xgb_grid.fit(train_X, train_Y, early_stopping_rounds=50, eval_metric='auc', eval_set=[(val_X, val_Y)])
        
        print('best parameters : ', xgb_grid.best_params_)
        print('best val score : ', round(xgb_grid.best_score_, 4))
        
        best_model = xgb_grid.best_estimator_
        
        return best_model
    
    def test_score(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        pred = model.predict(test_X)
        F1_score = f1_score(test_Y, pred)
        print("f1_score: ", F1_score)
        print(classification_report(test_Y, pred))
        
    def confusion_matrix(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        label=['0', '1']
        plot = plot_confusion_matrix(model,
                                    test_X, test_Y, 
                                    display_labels=label,
                                    # cmap=plt.cm.Blue,  (plt.cm.Reds, plt.cm.rainbow)
                                    normalize=None) # 'true', 'pred', 'all', default=None
        plot.ax_.set_title('Confusion Matrix')
        plt.show()
        
    def feature_importance(self, model):
        from xgboost import plot_importance
        fig, ax = plt.subplots(figsize=(9,11))
        plot_importance(model, ax=ax)
        plt.show()
        
        
    def shap(self, model):
        test_X = self.test_X
        
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(test_X) 
        
        plt.subplot(121)
        shap.initjs() 
        shap.summary_plot(shap_values, test_X)
        plt.show()
        
        plt.subplot(122)
        shap.summary_plot(shap_values, test_X, plot_type = "bar")
        plt.show()
        
        
class LGBM():
    # {'n_estimators': list(np.linspace(100, 5000, 50)),
    #                          'learning_rate': [0.2, 0.1, 0.05, 0.01],
    #                          'max_depth': list(range(3,11)),
    #                          'subsample': [0,7, 0.8, 0.9, 1],
    #                          'colsample_bytree':[0.5, 0.75, 1]}
    
    def __init__(self, train_X, train_Y, val_X, val_Y, test_X, test_Y, random_state):
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.params = {'n_estimators': [100, 200, 300],
                         'learning_rate': [0.2, 0.1, 0.05],
                         'max_depth': [3,4]}
        self.random_state = random_state
        
        
    def grid_search(self):
        train_X = self.train_X
        train_Y = self.train_Y
        val_X = self.val_X
        val_Y = self.val_Y
        params = self.params
        
        lgbm = LGBMClassifier()
        lgbm_grid = RandomizedSearchCV(lgbm, param_distributions= params, n_iter=5, scoring="f1", cv=3, refit=True, random_state=self.random_state)
        lgbm_grid.fit(train_X, train_Y, eval_metric='auc', eval_set=[(val_X, val_Y)])
        
        print('best parameters : ', lgbm_grid.best_params_)
        print('best val score : ', round(lgbm_grid.best_score_, 4))
        
        best_model = lgbm_grid.best_estimator_
        
        return best_model
    
    def test_score(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        pred = model.predict(test_X)
        F1_score = f1_score(test_Y, pred)
        print("f1_score: ", F1_score)
        print(classification_report(test_Y, pred))
        
    def confusion_matrix(self, model):
        test_X = self.test_X
        test_Y = self.test_Y
        
        label=['0', '1']
        plot = plot_confusion_matrix(model,
                                    test_X, test_Y, 
                                    display_labels=label,
                                    # cmap=plt.cm.Blue,  (plt.cm.Reds, plt.cm.rainbow)
                                    normalize=None) # 'true', 'pred', 'all', default=None
        plot.ax_.set_title('Confusion Matrix')
        plt.show()
        
    def feature_importance(self, model):
        from lightgbm import plot_importance
        fig, ax = plt.subplots(figsize=(9,11))
        plot_importance(model, ax=ax)
        plt.show()
        
        
    def shap(self, model):
        test_X = self.test_X
        
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(test_X)
        
        plt.subplot(121)
        shap.initjs() 
        shap.summary_plot(shap_values, test_X)
        plt.show()
        
        plt.subplot(122)
        shap.summary_plot(shap_values, test_X, plot_type = "bar")
        plt.show()