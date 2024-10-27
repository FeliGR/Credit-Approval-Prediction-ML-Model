import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.base import TransformerMixin
import os
# Definir la clase personalizada corr_selection
class corr_selection(TransformerMixin):

    def __init__(self, umbral=0.95, verbose=False):
        self.umbral = umbral
        self.verbose = verbose

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        correlaciones = X.corr().abs()
        upper = correlaciones.where(np.triu(np.ones(correlaciones.shape), k=1).astype('bool'))
        self.indices_variables_a_eliminar = [i for i, column in enumerate(upper.columns) if any(upper[column] > self.umbral)]
        if self.verbose:
            print('Se han eliminado {} variables, que son: '.format(len(self.indices_variables_a_eliminar)))
            print(list(X.columns[self.indices_variables_a_eliminar]))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_uncorr = X.copy()
        X_uncorr = X_uncorr.drop(columns=X.columns[self.indices_variables_a_eliminar], axis=1)
        return X_uncorr

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"umbral": self.umbral}

# Función para cargar los modelos
def load_models(classifier_path, regressor_path):
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    return classifier, regressor

# Función para realizar las predicciones
def make_predictions(input_file, classifier, regressor):
    df = pd.read_csv(input_file)
    pred_clf = classifier.predict(df)  # El pipeline manejará el preprocesamiento
    pred_reg = regressor.predict(df)

    results = pd.DataFrame({
        'Id': df.index + 1,
        'CreditoAprobado': pred_clf,
        'ScoreRiesgo': pred_reg
    })
    
    return results

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <input_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Obtener la ruta actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Definir las rutas de los modelos relativas al directorio actual del script
    classifier_path = os.path.join(script_dir, 'best_classifier.pkl')
    regressor_path = os.path.join(script_dir, 'best_regressor.pkl')

    classifier, regressor = load_models(classifier_path, regressor_path)
    # Realizar las predicciones
    results = make_predictions(input_file, classifier, regressor)

    # Imprimir los resultados
    print("Id,CreditoAprobado,ScoreRiesgo")
    for _, row in results.iterrows():
        print(f"{int(row['Id'])},{int(row['CreditoAprobado'])},{int(row['ScoreRiesgo'])}")

if __name__ == "__main__":
    main()
