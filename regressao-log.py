import pandas as pd
import numpy as np
import statsmodels.api as sm

# Dados fictícios
data = {
    'idade': [25, 45, 35, 50, 23, 40, 30, 60, 50, 35, 55, 62, 43, 31, 37],
    'colesterol': [200, 250, 240, 270, 180, 220, 210, 280, 260, 230, 300, 310, 220, 210, 225],
    'doenca_cardiaca': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)

# Adicionando uma constante
df['constante'] = 1

# Variáveis dependentes e independentes
y = df['doenca_cardiaca']
X = df[['constante', 'idade', 'colesterol']]

# Ajuste do modelo de regressão logística
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Resumo dos resultados
print(result.summary())

# Previsões de probabilidade
y_pred = result.predict(X)
df['probabilidade'] = y_pred
print(df[['idade', 'colesterol', 'doenca_cardiaca', 'probabilidade']])
