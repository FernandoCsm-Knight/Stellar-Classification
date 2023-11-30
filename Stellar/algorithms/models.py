from typing import Sequence
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from scipy import stats

class ModelManager:

    img_path: str = 'img/'
    csv_path: str = 'tests/'
    log_path: str = 'log/response.log'

    def __init__(self, cv = 5, random_state: int = 42):
        with open(self.log_path, 'w', encoding='utf-8') as file:
            file.write('')
    
        self.cv = cv
        self.grid_search = None
        self.random_search = None
        self.random_state = random_state

    def __grid_search(self, estimator, params, X, y):
        self.grid_search = GridSearchCV(estimator, params, cv=self.cv, n_jobs=-1)
        self.grid_search.fit(X, y)
        print(f'Melhores parâmetros: {self.grid_search.best_params_}')
        print(f'Acurácia do Grid Search: {self.grid_search.best_score_}')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(f'Melhores parâmetros: {self.grid_search.best_params_}\n')
            file.write(f'Acurácia do Grid Search: {self.grid_search.best_score_}\n')
        return self.grid_search.best_estimator_
    
    def __random_search(self, estimator, params, X, y):
        self.random_search = RandomizedSearchCV(estimator, params, cv=self.cv, n_jobs=-1)
        self.random_search.fit(X, y)
        print(f'Melhores parâmetros: {self.random_search.best_params_}')
        print(f'Acurácia do Random Search: {self.random_search.best_score_}')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(f'Melhores parâmetros: {self.random_search.best_params_}\n')
            file.write(f'Acurácia do Random Search: {self.random_search.best_score_}\n')
        return self.random_search.best_estimator_

    def __determine_estimator(self, estimator, X, y, *, params, grid_search):
        m = estimator
        
        if params != None:
            if(grid_search):
                m = self.__grid_search(estimator=estimator,
                                       params=params, 
                                       X=X, y=y) 
                
            else: 
                m = self.__random_search(estimator=estimator,
                                         params=params, 
                                         X=X, y=y)
        else:
            m.fit(X, y)
        
        return m
        
    def __get_pipeline(self, model, steps: Sequence[tuple] = None) -> Pipeline:
        if(steps == None):
            return Pipeline(steps=[('classifier', model)])
        else:
            return Pipeline(steps=[*steps, ('classifier', model)])
        
    def train_MLP(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando MLP...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando MLP...\n')
        
        pipe = self.__get_pipeline(model=MLPClassifier(random_state=self.random_state), steps=steps)
        mlp = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('MLP treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('MLP treinado!\n')
        return mlp 
    
    def train_KNN(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando KNN...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando KNN...\n')
        
        pipe = self.__get_pipeline(model=KNeighborsClassifier(), steps=steps)
        knn = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('KNN treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('KNN treinado!\n')
        return knn
    
    def train_XGB(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando XGB...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando XGB...\n')
        
        pipe = self.__get_pipeline(model=XGBClassifier(random_state=self.random_state), steps=steps)
        xgb = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('XGB treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('XGB treinado!\n')
        return xgb
    
    def train_RF(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando RF...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando RF...\n')
        
        pipe = self.__get_pipeline(model=RandomForestClassifier(random_state=self.random_state), steps=steps)
        rf = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('RF treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('RF treinado!\n')
        return rf
    
    def train_SVC(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando SVC...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando SVC...\n')
        
        pipe = self.__get_pipeline(model=svm.SVC(random_state=self.random_state), steps=steps)
        svc = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('SVC treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('SVC treinado!\n')
        return svc
    
    def train_BAG(self, X, y, *, grid_search: bool = False, steps: Sequence[tuple] = None, params = None) -> Pipeline:
        print('Treinando BAG...')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('Treinando BAG...\n')
        
        pipe = self.__get_pipeline(model=BaggingClassifier(random_state=self.random_state), steps=steps)
        bag = self.__determine_estimator(estimator=pipe,
                                         X=X, y=y, 
                                         params=params, 
                                         grid_search=grid_search)
        
        print('BAG treinado!')
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write('BAG treinado!\n')
        return bag

    def test(self, title: str, model, X_train, y_train, X_test, y_test, steps: Sequence[tuple] = None):
        print(f'Validando {title}...')
        validation = cross_validate(self.__get_pipeline(model=model, steps=steps), X_train, y_train, cv=self.cv, n_jobs=-1)
        
        print(f"Cross Validation scores for {title}: {validation['test_score']}")
        print(f"Mean score for {title}: {validation['test_score'].mean()}")
        print(f"Std score for {title}: {validation['test_score'].std()}")
        print(f'Validação {title} concluída!')
        
        print(f'Testando {title}...')
        score = model.score(X_test, y_test)
        print(f'Acurácia: {score}')
        
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(f'Validando {title}...\n')
            file.write(f'Cross Validation scores for {title}: {validation["test_score"]}\n')
            file.write(f'Mean score for {title}: {validation["test_score"].mean()}\n')
            file.write(f'Std score for {title}: {validation["test_score"].std()}\n')
            file.write(f'Validação {title} concluída!\n')
            file.write(f'Testando {title}...\n')
            file.write(f'Acurácia: {score}\n')

        prediction = model.predict(X_test.copy())
        cm = confusion_matrix(y_test.copy(), prediction)
        ConfusionMatrixDisplay(cm).plot()

        plt.savefig(f'{self.img_path}{title}.png')

        class_report = classification_report(y_test.copy(), prediction, output_dict=True)
        class_report = pd.DataFrame(class_report).transpose()
        class_report.insert(0, 'X', ['Dwarf', 'Giant', 'accuracy', 'macro avg', 'weighted avg'])
        class_report.to_csv(f'{self.csv_path}{title}.csv', index=False)
        print(f'Teste {title} concluído!')
        return validation;
    
    def validate_alternative_hypothesis(self, all_scores, model_names):
        num_models = len(all_scores)
        alpha = 0.05
        t_test_results = []

        for i in range(num_models):
            for j in range(i+1, num_models):
                stat, p_value = stats.ttest_ind(all_scores[i], all_scores[j])
                mean_i = np.mean(all_scores[i])
                mean_j = np.mean(all_scores[j])
                mean_diff = mean_i - mean_j
                df = len(all_scores[i]) + len(all_scores[j]) - 2  # graus de liberdade
                t_critical = stats.t.ppf(1 - alpha/2, df)  # valor crítico bilateral

                sigma_i = np.std(all_scores[i], ddof=1) / np.sqrt(len(all_scores[i]))  # erro padrão da média conjunto i
                sigma_j = np.std(all_scores[j], ddof=1) / np.sqrt(len(all_scores[j]))  # erro padrão da média conjunto j

                ci_i = stats.t.interval(0.95, len(all_scores[i]) - 1, loc=mean_i, scale=sigma_i)  # intervalo de confiança conjunto i
                ci_j = stats.t.interval(0.95, len(all_scores[j]) - 1, loc=mean_j, scale=sigma_j)  # intervalo de confiança conjunto j

                significant_difference = abs(stat) > t_critical
                result = {
                    'Comparison': f'{model_names[i]} vs {model_names[j]}',
                    f'Mean {model_names[i]}': mean_i,
                    f'Mean {model_names[j]}': mean_j,
                    'Mean Difference': abs(mean_diff),
                    'T-Statistic': stat,
                    'P-Value': p_value,
                    'T-Critical': t_critical,
                    f'95% CI {model_names[i]}': ci_i,
                    f'95% CI {model_names[j]}': ci_j,
                    'Significant': significant_difference
                }
                t_test_results.append(result)

        return t_test_results
    

    def save_hypothesis_results(self, results, filename_prefix='hypothesis_test'):
        csv_file = f'{self.csv_path}{filename_prefix}.csv'
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False, encoding='utf-8')

        txt_file = f'{self.csv_path}{filename_prefix}.txt'
        with open(txt_file, 'w', encoding='utf-8') as file:
            for result in results:
                for key, value in result.items():
                    file.write(f'{key}: {value}\n')
                file.write('\n')
    
    def plot_roc_curves(self, models, X_test, y_test):
        plt.figure(figsize=(10, 8))

        for name, model in models.items():
            if hasattr(model, "predict"):
                probas = model.predict(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            else:
                print(f"Modelo {name} não suporta predict_proba e será ignorado.")

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title('Curvas ROC para Diversos Modelos')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.img_path}roc_curves.png') 
        
    def plot_boxplot_scores(self, data):        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(data))
        plt.title("Comparação de Desempenho dos Modelos")
        plt.ylabel("Score (e.g., Acurácia)")
        plt.xlabel("Modelos")
        plt.xticks(rotation=45)
        plt.savefig(f'{self.img_path}model_performance_boxplot.png')

    def plot_kfold_result(self, X, y):
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm

        def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
            for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
                indices = np.array([np.nan] * len(X))
                indices[tt] = 1
                indices[tr] = 0

                ax.scatter(
                    range(len(indices)),
                    [ii + 0.5] * len(indices),
                    c=indices,
                    marker="_",
                    lw=lw,
                    cmap=cmap_cv,
                    vmin=-0.2,
                    vmax=1.2,
                )

            ax.scatter(
                range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
            )

            yticklabels = list(range(n_splits)) + ["class"]
            ax.set(
                yticks=np.arange(n_splits + 1) + 0.5,
                yticklabels=yticklabels,
                xlabel="Sample index",
                ylabel="CV iteration",
                ylim=[n_splits + 1.1, -0.1],
                xlim=[0, 100],
            )
            ax.set_title("{}".format(type(cv).__name__), fontsize=15)
            return ax

        fig, ax = plt.subplots(figsize=(19, 8))
        plot_cv_indices(self.cv, X=X, y=y, ax=ax, n_splits=(self.cv if type(self.cv) == int else self.cv.get_n_splits()))

        ax.legend(
            [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
            ["Testing set", "Training set"],
            loc=(1.02, 0.8),
        )

        plt.tight_layout()
        fig.subplots_adjust(right=0.7)

        plt.savefig(f'{self.img_path}Kfold_cv.png')

