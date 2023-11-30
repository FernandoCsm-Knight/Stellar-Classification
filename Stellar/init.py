import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.impute import KNNImputer
from sklearn import tree
from sklearn.metrics import silhouette_score

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from algorithms.star import StarClassification
from algorithms.models import ModelManager

# Leitura dos dados

print("Lendo arquivo...")

star = StarClassification(
  db = pd.read_csv('/home/fernandocsdm/Área de trabalho/Stellar/Star_raw_nan_fixed.csv'), 
  inputer = KNNImputer(n_neighbors=10),
  splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2),
  under_sampler = NearMiss(version=1),
  over_sampler = SMOTE(random_state=42),
  binary_encoder = LabelEncoder()
)

# Remoção de registros nulos para o atributo *'SpType'* e criação do atributo alvo da predição.

print("Removendo registros nulos...")

def determine_category(sp_type):
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    sp_type = sp_type.upper()

    for numeral in roman_numerals:
        if numeral in sp_type:
            if numeral in ['I', 'II', 'III']:
                return 'Giant'
            elif numeral in ['IV', 'V', 'VI', 'VII']:
                return 'Dwarfs'

    return np.nan

star.db.dropna(subset=['SpType'], inplace=True)
star.db['Target'] = star.db['SpType'].apply(determine_category)
star.db.dropna(subset=['Target'], inplace=True)

# Remoção de atributos inadequados para o treinamento e teste do modelo

print("Removendo atributos inadequados...")

star.db.drop(['ID', 'SpType'], axis=1, inplace=True)

# Remoção de *Outliers* com o uso do IQR

print("Removendo outliers...")

outliers_to_test = pd.DataFrame(columns=star.db.columns)

outliers_sum = 0
for col in star.db.columns:
  if col != 'Target':
    outliers = star.outliers_remove(col)
    
    if outliers_to_test.empty:
        outliers_to_test = outliers
    else:
        outliers_to_test = pd.concat([outliers_to_test, outliers])

    outliers_sum += outliers.shape[0]

outliers = star.outliers_remove('Plx')
outliers_to_test = pd.concat([outliers_to_test, outliers])
outliers_sum += outliers.shape[0]

# Tratamento de valores faltantes

print("Tratando valores faltantes...")

for col in star.db.columns:
  if col != 'Target':
    star.input_values(col)
    
# Criação do atributo 'Ameg' ou maginitude absoluta.    

print("Criando atributo 'Ameg'...")

num = star.db[star.db['Plx'] == 0.0].shape[0]
star.db = star.db[star.db['Plx'] != 0.0].copy()

mv = np.array(star.db['Vmag'], dtype=float)
Plxi = np.array(star.db['Plx'], dtype=float)

Plxi = np.maximum(Plxi, 1e-10) # Garante que Plxi é positivo antes de calcular di  

di = 1000 / Plxi
Mv = mv - 5 * np.log10(di) + 5
star.db['Amag'] = Mv.copy()

outliers = star.outliers_remove('Amag')
outliers_to_test = pd.concat([outliers_to_test, outliers])
outliers_sum += outliers.shape[0]

# Codificação do atributo "Target"

print("Codificando atributo 'Target'...")

star.encode_binary('Target')

# Separação dos dados em treino e teste.

print("Separando dados em treino e teste...")

db_train, db_test = star.split(['Target'])

# Aplicação dos modelos

model_manager = ModelManager(cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

entries = db_train.drop(['Target'], axis=1).shape[1]
outputs = db_train['Target'].nunique()

mlp = model_manager.train_MLP(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('undersampler', NearMiss(version=1))],
  params={
    'classifier__hidden_layer_sizes': [(entries*2 + 1,), (entries*2 + 1, entries + 1), (int((entries + outputs) / 2),), (int(np.sqrt(entries*outputs)),)],
    'classifier__activation': ['relu', 'tanh', 'logistic'],
    'classifier__alpha': [0.0000001, 0.0001, 0.001, 0.01],
    'classifier__max_iter': [1000],
  },
)

knn = model_manager.train_KNN(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('undersampler', NearMiss(version=1))],
  grid_search=True,
  params={
    'classifier__n_neighbors': [1, 3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
  },
)

xgb = model_manager.train_XGB(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('undersampler', NearMiss(version=1))],
  params={
    'classifier__n_estimators': [100, 150, 200, 250],
    'classifier__max_depth': [3, 5, 7, 9, 10],
    'classifier__learning_rate': [0.1, 0.2, 0.3],
    'classifier__gamma': [0, 0.1, 0.2, 0.3]
  },
)

rf = model_manager.train_RF(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('undersampler', NearMiss(version=1))],
  grid_search=True,
  params={
    'classifier__n_estimators': [100, 150, 200, 250],
    'classifier__max_depth': [3, 5, 7, 9, 10],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion': ['gini', 'entropy']
  },
)

svc = model_manager.train_SVC(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('scaler', MinMaxScaler()), ('undersampler', NearMiss(version=1))],
  params={
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__C': [0.1, 0.5, 1, 1.5, 2],
    'classifier__gamma': [0.1, 0.5, 1, 1.5, 2],
  },
)

bag = model_manager.train_BAG(
  X=db_train.drop(['Target'], axis=1).copy(),
  y=db_train['Target'].copy(),
  steps=[('undersampler', NearMiss(version=1))],
  params={
    'classifier__n_estimators': [100, 150, 200, 250],
    'classifier__max_samples': [0.1, 0.5, 1.0],
    'classifier__max_features': [0.1, 0.5, 1.0],
  },
)

# Teste dos modelos

validation_mlp = model_manager.test(
  title='MLP',
  model=mlp, 
  steps=[('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

validation_knn = model_manager.test(
  title='KNN',
  model=knn, 
  steps=[('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

validation_xgb = model_manager.test(
  title='XGB',
  model=xgb, 
  steps=[('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

validation_rf = model_manager.test(
  title='RF',
  model=rf, 
  steps=[('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

validation_svc = model_manager.test(
  title='SVC',
  model=svc, 
  steps=[('scaler', MinMaxScaler()), ('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

validation_bag = model_manager.test(
  title='BAG',
  model=bag, 
  steps=[('undersampler', NearMiss(version=1))],
  X_train=db_train.drop(['Target'], axis=1).copy(),
  y_train=db_train['Target'].copy(),
  X_test=db_test.drop(['Target'], axis=1).copy(),
  y_test=db_test['Target'].copy(),
)

# Validation Results

print("Resultados MLP:")
print(validation_mlp, end='\n\n')
print("Resultados KNN:")
print(validation_knn, end='\n\n')
print("Resultados XGB:")
print(validation_xgb, end='\n\n')
print("Resultados RF:")
print(validation_rf, end='\n\n')
print("Resultados SVC:")
print(validation_svc, end='\n\n')
print("Resultados BAG:")
print(validation_bag, end='\n\n')

with open(model_manager.log_path, 'a') as f:
  f.write("Resultados MLP:\n")
  f.write(str(validation_mlp))
  f.write("\n\n")
  f.write("Resultados KNN:\n")
  f.write(str(validation_knn))
  f.write("\n\n")
  f.write("Resultados XGB:\n")
  f.write(str(validation_xgb))
  f.write("\n\n")
  f.write("Resultados RF:\n")
  f.write(str(validation_rf))
  f.write("\n\n")
  f.write("Resultados SVC:\n")
  f.write(str(validation_svc))
  f.write("\n\n")
  f.write("Resultados BAG:\n")
  f.write(str(validation_bag))
  f.write("\n\n")

scores_mlp = validation_mlp['test_score']
scores_knn = validation_knn['test_score']
scores_xgb = validation_xgb['test_score']
scores_rf = validation_rf['test_score']
scores_svc = validation_svc['test_score']
scores_bag = validation_bag['test_score']

# Showing Validation Results

all_scores = [scores_mlp, scores_knn, scores_rf, scores_svc, scores_bag]
model_names = ['MLP', 'KNN', 'RF', 'SVC', 'BAG']

models = {
    'MLP': mlp,
    'KNN': knn,
    'RF': rf,
    'SVC': svc,
    'BAG': bag
}

data = {
    'MLP': scores_mlp,
    'KNN': scores_knn,
    'RF': scores_rf,
    'SVC': scores_svc,
    'BAG': scores_bag
}

# Show ROC Curves

model_manager.plot_roc_curves(models, db_test.drop(['Target'], axis=1).copy(), db_test['Target'].copy())

# Show Boxplot Cross Validatio Scores

model_manager.plot_boxplot_scores(data)

# Visualize the T Test P-Values

hypothesis_results = model_manager.validate_alternative_hypothesis(all_scores, model_names)
model_manager.save_hypothesis_results(hypothesis_results)

# Visualize K-Fold Results

model_manager.plot_kfold_result(db_train.drop(['Target'], axis=1).copy(), db_train['Target'].copy())
