import autokeras as ak
import pandas as pd
from sklearn.model_selection import train_test_split


def teste_auto_keras():
    df = pd.read_csv("Churn_treino.csv", sep=";")
    X = df.iloc[:, 0:10]
    y = df.iloc[:, 10]
    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
    modelo = ak.StructuredDataClassifier(max_trials=25)
    modelo.fit(x=X_train, y=y_train, epochs=500)
    print(f"Nome do modelo: {modelo.name}")
    print(modelo)
    avaliacao = modelo.evaluate(X_test, y=y_test)
    print("***********   Avaliação   ***********")
    print(avaliacao)

    df_prever = pd.read_csv("Churn_prever.csv", sep=";")
    previsao = modelo.predict(df_prever)
    print("Previsão:")
    print(previsao)


if __name__ == "__main__":
    teste_auto_keras()