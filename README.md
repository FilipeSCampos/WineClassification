# Documentação do Modelo de Machine Learning para Classificação de Vinhos

O código fornecido implementa um modelo de machine learning para classificar vinhos com base em um dataset famoso do Kaggle. Ele segue as seguintes etapas:

https://www.kaggle.com/datasets/rajyellow46/wine-quality?resource=download

## 1. Importação das bibliotecas
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

```
Foram essas bibliotecas que utilizei para esse estudo, incluindo o NumPy, Pandas, Matplotlib, Seaborn, entre outras, que são utilizadas para manipulação de dados, visualização de gráficos, avaliação de métricas e algoritmos de classificação.

## 2. Carregamento dos dados
```
df = pd.read_csv('winequalityN.csv')
display(df)

```
O dataset de vinhos é carregado a partir de um arquivo CSV chamado 'winequalityN.csv' e armazenado em um DataFrame chamado `df`. Em seguida, o DataFrame é exibido.

## 3. Análise exploratória dos dados
```
df.info()
df.describe().T
df.isnull().sum()

```
O código fornece informações básicas sobre o DataFrame, como o número de entradas, o número de colunas e os tipos de dados de cada coluna. Também são exibidas estatísticas descritivas para as colunas numéricas e a contagem de valores nulos em cada coluna.

## 4. Tratamento de valores nulos
```
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
df.isnull().sum()

```
Os valores nulos do DataFrame são tratados substituindo-os pela média dos valores não nulos da respectiva coluna.

## 5. Visualização dos dados
```
df.hist(bins=20, color='purple', figsize=(10,10))
plt.show()

plt.bar(df['quality'], df['alcohol'], color='purple')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

plt.figure(figsize=(12,10))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

```
São exibidos histogramas para cada coluna do DataFrame, mostrando a distribuição dos dados. Também é exibido um gráfico de barras relacionando a qualidade do vinho com o teor alcoólico. Além disso, é mostrada uma matriz de correlação entre as colunas do DataFrame.

![image](https://github.com/FilipeSCampos/WineClassification/assets/113521439/5b9d7a2a-0e32-4694-80ec-7ac305422f13)


![image](https://github.com/FilipeSCampos/WineClassification/assets/113521439/dab12d96-943e-44c5-a836-71a58adf972e)

![image](https://github.com/FilipeSCampos/WineClassification/assets/113521439/3110e388-0b71-4dc0-97a2-68129bdd0d07)


## 6. Preparação dos dados
```
df = df.drop('total sulfur dioxide', axis=1)

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

df.replace({'white': 1, 'red': 0}, inplace=True)

display(df)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

```
O DataFrame passa por algumas transformações, incluindo a remoção de uma coluna, a criação de uma nova coluna com base em uma condição, nesse caso eu tomei a decisão de que qualquer vinho com nota maior que 6 seria classificado como bom e abaixo de 6 como ruim, e a substituição de valores categóricos por valores numéricos. Em seguida, os dados são divididos em características (features) e alvo (target) e são separados em conjuntos de treinamento e teste. As características são normalizadas usando a escala mín-máx (MinMaxScaler).

![Capturar](https://github.com/FilipeSCampos/WineClassification/assets/113521439/41a8f3ad-9bfd-472e-9ca8-3997cdf319c3)

## 7. Treinamento e avaliação do modelo

```
modelo = DecisionTreeClassifier(criterion='gini', max_depth=7)

models = [LogisticRegression(), modelo, SVC(kernel='rbf'), XGBClassifier()]

for i in range(4):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()

```

Nessa parte eu fiz a escolha dos modelos de classificação, escolhi uma árvore de decisão que é um excelente modelo, performa bem em praticamente todos os casos, usei um modelo de Regressão logística para testar a performance, foi minha primeira vez utilizando esse modelo, também testei o Support Vector Classifier e por fim o XGBoost(Extreme Gradient Boosting) que é um modelo super poderoso que em todos os meus testes alcançou a melhor performance melhor resposta e a maior acurácia.

## 8. Importância das características
```
importancias = modelo.feature_importances_
nomes_features = features.columns

# Imprimir importância das características
for feature, importancia in zip(nomes_features, importancias):
    print(feature, importancia)
print("Acurácia Treino: ", modelo.score(xtrain, ytrain))
print("Acurácia Teste: ", modelo.score(xtest, ytest))

```

As métricas de acurácia de treinamento e teste são exibidas.
De início eu achei que o XGboost poderia estar um pouco overfitado, mas é o que tem a maior acurácia de teste também.

![Capturar2](https://github.com/FilipeSCampos/WineClassification/assets/113521439/5462aecb-4b2e-47e5-b390-335d8bbc03cc)



## 9. Matriz de Confusão e Relatório de Classificação

```
from sklearn.metrics import confusion_matrix
ypred = model.predict(X_test)

cm = confusion_matrix(ytest, models[i].predict(xtest))
sb.heatmap(cm, annot=True, fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Previsão")
plt.ylabel("Verdadeiro")
plt.show()

print('Logistic Regression \n ',metrics.classification_report(ytest,models[0].predict(xtest)))
print('Decision Tree \n ',metrics.classification_report(ytest,models[1].predict(xtest)))
print('SVC \n ',metrics.classification_report(ytest,models[2].predict(xtest)))
print('XGBOOST \n ',metrics.classification_report(ytest,models[3].predict(xtest)))

```

A matriz de confusão é calculada para o último modelo da lista e é exibida como um mapa de calor. Em seguida, são impressos os relatórios de classificação para cada modelo, que fornecem métricas detalhadas como precisão, recall e f1-score para cada classe.

![image](https://github.com/FilipeSCampos/WineClassification/assets/113521439/9c5c1904-3342-4516-acbe-6b5a387d5d49)

![Capturar4](https://github.com/FilipeSCampos/WineClassification/assets/113521439/34b6b588-5b7f-4ab8-b54f-29f6be9dc190)



