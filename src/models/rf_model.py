from tkinter import Grid
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance


class RF():
    
    def __init__(self, train_X, train_Y, val_X, val_Y, random_state):
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.params = {'n_estimators': list(np.linspace(100, 5000, 50)),
                         'max_depth': list(range(3,11)),
                         'min_samples_leaf': [5,10,15,20]}
        self.random_state = random_state
        
        
    def grid_search(self):
        train_X = self.train_X
        train_Y = self.train_Y
        params = self.params
        
        rf_model = RandomForestClassifier()
        rf_grid = RandomizedSearchCV(rf_model, param_distributions= params, n_iter=5, scoring="f1", cv=5, refit=True, random_state=self.random_state)
        rf_grid.fit(train_X, train_Y)
        
        print('best parameters : ', rf_grid.best_params_)
        print('best val score : ', round(rf_grid.best_score_, 4))
        
        best_model = rf_grid.best_estimator_
        
        return best_model
    
    def val_score(self, model):
        val_X = self.val_X
        val_Y = self.val_Y
        
        pred = model.predict(val_X)
        F1_score = f1_score(val_Y, pred)
        print("f1_score: ", F1_score)
        print(classification_report(val_Y, pred))
        
    def confusion_matrix(self, model):
        val_X = self.val_X
        val_Y = self.val_Y
        
        label=['0', '1']
        plot = plot_confusion_matrix(model,
                                    val_X, val_Y, 
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
        val_X = self.val_X
        
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(val_X)
        
        shap.summary_plot(shap_values, val_X, plot_type = "bar")
        plt.show()
    
    def permutation_importance(self, model):
        val_X = self.val_X
        val_Y = self.val_Y
        perm = PermutationImportance(model, scoring = "f1", random_state = self.random_state).fit(val_X, val_Y)
        eli5.show_weights(perm, top = 20, feature_names = val_X.columns.tolist())
        
    def pred_testset(self, test_X, model):
        pred = model.predict(test_X)
        test_X['is_applied'] = pred
        return test_X