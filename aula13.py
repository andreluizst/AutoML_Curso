import h2o
import pandas as pd
from h2o.automl import H2OAutoML


def teste1():
    imp = pd.read_csv("Churn_treino.csv", sep=";")
    print(imp.head())
    imp = h2o.H2OFrame(imp)
    treino, teste = imp.split_frame(ratios=[0.7])

    print("teste normal")
    print(f"imp is type of {type(imp)}")
    print(teste)

    treino["Exited"] = treino["Exited"].asfactor()
    teste["Exited"] = teste["Exited"].asfactor()

    print("teste após asfactor()")
    print(teste)

    modelo = H2OAutoML(max_runtime_secs=60)
    modelo.train(y="Exited", training_frame=treino)

    ranking = modelo.leaderboard
    ranking = ranking.as_data_frame()

    ranking.to_csv("ranking_od_modelos_2_60s.csv", index=False)

    print("Finished!")


if __name__ == "__main__":
    h2o.init()
    teste1()