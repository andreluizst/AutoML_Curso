from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


def teste_mlbox():
    caminho = ["Churn_treino.csv", "Churn_teste.csv"]
    reader = Reader(sep=";")
    daods = reader.train_test_split(caminho, target_name="Exited")

    #remover o drift (realiza um balanceameto da base)
    rdrift = Drift_thresholder()
    dados = rdrift.fit_transform(daods)

    otimizador = Optimiser()
    espaco_busca = {
        'fs__strategy': {"search": "choice", "space": ["variance", "rf_feature_importance"]},
        'est__colsample_bytree': {"search": "uniform", "space": [0.3, 0.7]}
    }

    modelo = otimizador.optimise(espaco_busca, dados, max_evals=26)

    previsor = Predictor()

    previsor.fit_predict(modelo, dados)


if __name__ == "__main__":
    teste_mlbox()