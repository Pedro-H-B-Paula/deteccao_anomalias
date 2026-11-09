import pandas as pd 

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'

df = pd.read_csv(url) #como o arquivo é csv o read_csv é usado ao invés do data frame

print(df.head())#esse comando faz mostrar as 5 primeiras linhas, podendo alterar pra mais e pra menos

df['Class'].value_counts(normalize=True) #mostra as quantidades se False, se for True exibe em forma de porcentagem

import numpy as np

df['Amount_log'] = np.log1p(df['Amount'])#quando transformado em log, todos os valores que são 0 deixam de existir, pois log de 0 n vai existir

from sklearn.preprocessing import StandardScaler
#o SKLEARN transforma os valores e faz com que eles tenham a média igual a 0 e o desvio padrão igual a 1, isso é feito para que valores muito grandes ou muito pequenos fiquem na mesma escala, sendo comparáveis
scaler = StandardScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
#o fit vai calcular a media e o desvio padrão da coluna Amount e o transform vai fazer a transformação anteriormente explicada, tendo o resultado armazenado na coluna Amount_scaled

#muitos modelos são sensiveis a grandes diferenças de valores, então transferindo todos os valores para uma mesma escala o modelo é treinado mais facilmente

from sklearn.model_selection import train_test_split

x = df.drop('Class', axis=1)#o x representa as features, ou seja, todas as variáveis de entrada, pois isso usamos drop class com axis=1(axis 0 é linha, axis 1 é coluna), removendo a coluna class, se usassemos o class junto a isso, o modelo saberia a resposta ele tem que tentar adivinhar a class com base nas informações dessas características
y =  df['Class'] #o y representa o target, ou seja, o que estamos querendo ver (a class), por isso removemos da anterior(o X são as características)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42)
#essa parte é muito importante, pois é aqui que separamos nosso modelo em conjuntos de treino e teste, pois o conjunto de treino serve pra treinar o modelo, e o conjunto de teste serve para avaliar a situação do modelo.

#utilizamos teste_size 0.3 para informar que 30% dos dados eles vão para teste o resto(70%) vão pro modelo ser treinado

#o state 42 serve para garantir que a divisão seja sempre a mesma

#o stratify garante que a proporção de fraudes e não-fraudes seja mantida no treino e no teste
from sklearn.linear_model import LogisticRegression #o LogisticRegression é um modelo de classificação, sua função é prever categorias, nesse caso, prever se é uma fraude(classe 1) ou não(classe 0) com base nas categorias.
model = LogisticRegression(max_iter=1000)
model.fit (x_train, y_train) #o model.fit faz com que o modelo aprenda padrões a partir dos dados de treino, tentando encontrar relações entre as variáveis, features e etc. conseguindo distinguir fraude ou não fraude
y_pred = model.predict(x_test)
#aqui é onde o modelo faz previsões nos dados de teste, ou seja, nos dados que ele nunca viu e tentar classificar como fraude ou não fraude

from sklearn.metrics import classification_report #com isso conseguimos ver as métricas mais importantes
print(classification_report (y_test, y_pred)) 
#com o recall vemos que o modelo acertou 67% das fraudes reais, com uma margem de erro de 32%
#o precision diz quantas vezes ao afirmar que é fraude, essa afirmação está correta

# Undersampling diminui a quantidade de transações normais, a afim de deixar a proporção de fraudes maior
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(10000, random_state=42)

df_under = pd.concat([fraudes, normais])

# Separando X e y
X = df_under.drop("Class", axis=1)
y = df_under["Class"]

# Oversampling aumenta quantidade de ocorrência fraudes adaptando de uma forma melhor o modelo
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_res, y_res = smote.fit_resample(X, y)

from sklearn.ensemble import RandomForestClassifier  #o random forest é um modelo baseado em árvores de decisão, cada árvore vai aprender um padrão diferente dos dados, e no final da operação ele vai mesclar essas decisões
rf = RandomForestClassifier(
    n_estimators=50, #a quantidade de árvores definidas
    max_depth=10, #define a profundidade que uma árvore pode chegar, o nível está em 10 pois evita overfitting, um comportamento do modelo de criar regras ultra-especificas que só funcionam para modelos já vistos
    class_weight="balanced", #balanceia a importancia das classes no treino, então quando aparecer uma fraude(que é mais rara, o modelo ignora menos)
    n_jobs=-1,
    random_state=42
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test, y_pred_rf)) #após o uso do forest, o modelo teve mais informações e dados a serem analisados e com isso, ele foi capaz que aumentar o recall(ou seja, a % de fraudes detectadas aumentou)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()), #padronizando escalas para agilizar o processo para o modelo
    ("model", LogisticRegression(max_iter=1000)) #treinamento do modelo, usado para identificar os dados
])

pipeline.fit(x_train, y_train) #o pipeline serve para organizar o fluxo de processamento, mas esse do scikit-learn serve para organizar e automatizar uma sequência de etapas do processamento e treinamento de um modelo de machine learning

y_pred = pipeline.predict(x_test)


threshold = 0.3 #por padrão o modelo vem com 0.5 d threshold(ou seja, se a probabilidade de fraude for maior qu 0.5 o modelo classifica como fraude. Fazer isso deixa o modelo mais rigoroso, mas também aumenta a chance de falsos positivos)

y_probs = pipeline.predict_proba(x_test)[:, 1]

y_pred_custom = (y_probs > threshold).astype(int)

print(classification_report(y_test, y_pred_custom))


from xgboost import XGBClassifier #o xgboost é quase igual o forest, mas ele é sequencial (ou seja, primeira uma 'arvore' treina e depois vem outra, corrigindo os erros dessa anterior)

xgb = XGBClassifier(
    scale_pos_weight=10,  # ajuda com desbalanceamento
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

print(classification_report(y_test, y_pred_xgb))

from sklearn.model_selection import GridSearchCV #aqui se utiliza uma ferramenta chamada GridSearchCV, que testa automaticamente diferentes configurações do XGBoost para encontrar uma combinação que apresente o melhor resultado na métrica escolhida

param_grid = { #configurações que serão testadas.
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
} #aqui ele vai testar 4 combinações com os valores pedidos (3, 50/ 3, 100, / 5, 50 / 5, 100 / e vai dizer qual das 4 é a mais eficiente )

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall", #utiliza o Recall para avaliar os resultados.
    cv=3 #divide os dados de treinamento em 3 partes para fazer validação cruzada(mais explicado em explicações.txt).
)

grid.fit(x_train, y_train)

print("Melhor modelo:", grid.best_params_)

import shap

explainer = shap.Explainer(xgb) #esse explicador vai analisar como cada variável influencia as previsões do modelo, pense nele como uma ferramenta que tenta descobrir por que o XGBoost tomou determinada decisão
shap_values = explainer(x_test[:100]) #seleciona apenas as primeiras 100 linhas do conjunto de testes e o shap_values guarda os resultados dessa analise.

shap.plots.bar(shap_values) #cria um gráfico de barras mostrando a importância das variáveis para as previsões do modelo, esse gráfico mostra a importância média das variáveis nas previsões analisadas, não se uma variável aumenta ou diminui a chance de fraude..

#por que apenas 100 linhas? porque analisar o conjunto inteiro pode ser mais demorado. Essas 100 linhas servem como uma amostra para visualizar as explicações.