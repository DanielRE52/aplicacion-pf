import streamlit as st
import pandas as pd
import numpy as np
import sksurv
import sklearn
import warnings
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning


#nombre de la pagina y config general
st.set_page_config(
    page_title="Analizador de Variables Influyentes - Reprobación de estudiantes",
    layout="wide"
)

#titulo y solicitarle al usuario que cargue el archivo en el formato correcto
st.title("Evaluación de Variables Influyentes en el Rendimiento Académico")
st.info("Proyecto final del departamento de Ingenieria Industrial de la Universidad del Norte. Realizado por Daniel Ramirez, Santiago Reyes y Edwin Yunis.")
st.write("***Carga un archivo CSV o Excel, selecciona las columnas de tiempo y evento, y construye la base para el análisis.***")

st.header("Análisis grupal de las variables de influencia: ")
#carga de archivo
st.subheader("Selección de base de datos y de variables de supervivencia.")
archivo = st.file_uploader("Selecciona la base de datos la cual quieres evaluar.", type=["csv", "xlsx"])

#lectura del archivo
if archivo is not None:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        st.subheader("Vista previa de la base de datos: ")
        st.dataframe(df.head())  #un head para que vea que monto

        st.subheader("Información general de la base de datos: ")  #observaciones y variables de la base de datos
        st.write(f"Observaciones: {df.shape[0]}")
        st.write(f"Variables: {df.shape[1]}")

        st.write("Selecciona las variables de supervivencia:")
        st.info("En un modelo de regresión de Cox, las variables de supervivencia son fundamentales porque **permiten definir cuándo ocurre el evento de interés y si dicho evento ocurrió o no**. En este caso, se deben seleccionar dos variables: una de tiempo y una de evento")
        st.info("La **variable tiempo** representa el periodo transcurrido hasta que se observa el resultado de interés. En el contexto de este proyecto, se interpreta como la cantidad de cortes académicos de la asignatura, ya que se asume que el resultado final del estudiante (aprobar o reprobar) se determina al finalizar el semestre.")
        st.info("Por su parte, la **variable evento** indica si el suceso de interés ocurrió. Para este análisis, el evento corresponde al resultado académico del estudiante en la asignatura, es decir, si aprobó o reprobó.")
        
        columnas = df.columns.tolist()
        if "TIEMPO" in columnas and "EVENTO" in columnas:
            ytime = "TIEMPO"
            yevent = "EVENTO"
            st.success("Se reconocieron automáticamente las columnas 'TIEMPO' y 'EVENTO'.")
        else:
            st.warning("No se encontraron automáticamente las columnas 'TIEMPO' y 'EVENTO'.")
            st.write("Escriba manualmente el nombre de las columnas correspondientes.")
            col_tiempo = st.text_input("Nombre de la columna de tiempo")
            col_evento = st.text_input("Nombre de la columna de evento")
            
            if col_tiempo and col_evento:
                if col_tiempo == col_evento:
                    st.error("La columna de tiempo y la columna de evento deben ser diferentes.")
                    ytime = None
                    yevent = None
                elif col_tiempo not in columnas or col_evento not in columnas:
                    st.error("Uno o ambos nombres ingresados no existen en la base de datos.")
                    ytime = None
                    yevent = None
                else:
                    df = df.rename(columns={
                col_tiempo: "TIEMPO",
                col_evento: "EVENTO"
                    })
                    ytime = "TIEMPO"
                    yevent = "EVENTO"
                    st.success("Las columnas fueron renombradas correctamente a 'TIEMPO' y 'EVENTO'.")

        if ytime and yevent:
            if ytime == yevent:  #validacion de errores
                st.error("La variable tiempo y la variable evento deben ser diferentes.")
            else:
                y = Surv.from_dataframe(event=yevent, time=ytime, data=df)  #creacion de y con las variables time y event
                X = df.drop(columns=[ytime, yevent])  #creacion de x con el resto de variables de la base

                st.success("Variables de tiempo y evento cargadas correctamente.")
                
                #dummies para variables categoricas
                X = pd.get_dummies(X, drop_first=True)

                #quitar columnas constantes
                X = X.loc[:, X.nunique() > 1]

                #split de training y testing
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
                st.write("**Presiona el botón 'Ajustar modelo Cox' si la información es correcta.**")
                st.info("***El modelo de Cox se ajustará inicialmente utilizando el conjunto de entrenamiento, y luego se evaluará su desempeño en el conjunto de prueba. Esto permitirá obtener una estimación más realista de la capacidad predictiva del modelo.***")
                st.info("**Separar los datos en conjuntos de training y testing es fundamental para la construcción de modelos predictivos. De acuerdo con los autores Emmert-Streib y Dehmer (2019), el conjunto de entrenamiento se utiliza para estimar o conocer los parámetros de los modelos, también conocido como el ajuste del modelo. Por otro lado, el testing se utiliza para evaluar el rendimiento del modelo entrenado, utilizando este conjunto de datos para la estimación de la generalización del error del modelo final.**")
                st.info("**Puesto de otra manera, entrenar el modelo permite que este identifique correctamente patrones y comportamientos, mientras que el testing evalúa el desempeño y la precisión del modelo después del training con datos no vistos, que también son datos reales. En este caso, se utiliza el 70% de los datos para entrenar y el 30% restante se emplea para evaluar.**")

                if st.button("Ajustar modelo Cox"):
                    estimator = CoxPHSurvivalAnalysis(alpha=0.1)
                    estimator.fit(X_train, y_train)
                    st.success("Modelo ajustado")
                    st.subheader("Log-hazard Ratios: Coeficientes de la Regresión de Cox")

                    #coeficientes
                    coef = pd.Series(estimator.coef_, index=X_train.columns)
                    st.bar_chart(coef.sort_values(ascending=False),x_label="Variables", y_label="Valor de los coeficientes", color="#CC00CC")

                    #hazard ratios
                    st.subheader("Hazard Ratios: Magnitud de influencia de las variables")
                    hazard_ratios = np.exp(estimator.coef_)
                    hazard_ratios_series = pd.Series(hazard_ratios, index=X_train.columns).sort_values(ascending=False)
                    st.write(hazard_ratios_series)
                    st.info("**Como interpretar los valores**: Un Hazard Ratio superior a 1 indica que la variable aumenta la probabilidad de reprobación. De misma manera, un Hazard Ratio menor que 1 indica que la variable disminuye la probabilidad de reprobación.")
                    st.info("**Ejemplo 1**: Si el Hazard Ratio de la variable EDAD_INGRESO_16_17 es 1.5, esto implica que si un estudiante entro a la Universidad con 16 o 17 años su probabilidad de reprobación aumenta en un 50%.")
                    st.info("**Ejemplo 2**: Si el Hazard Ratio de la variable ESTADO_ACADEMICO=ESTUDIANTE_DISTINGUIDO es 0.6. Esto implica que si el ESTADO_ACADEMICO del estudiante es Distinguido entonces su probabilidad de reprobación disminuye en un 40%.")
                    st.subheader("Variables más influyentes: ")
                    max5 = (hazard_ratios_series.nlargest(8)-1)*100
                    st.write("Las 8 variables que más aumentan la probabilidad de reprobación son: ")
                    st.bar_chart(max5, y_label = "Porcentaje de aumento", x_label ="Variables", color ="#000068")
                    min5= (1-hazard_ratios_series.nsmallest(8))*100
                    st.write("Las 8 variables que más disminuyen la probabilidad de reprobación son: ")
                    st.bar_chart(min5, y_label = "Porcentaje de disminución", x_label ="Variables", color ="#89cff0")

                    st.subheader("Evaluación del Modelo: ")
                    st.info("En la regresión de Cox se emplea el Harrell’s Concordance Index, o C-index, para validar que tan bueno es el ajuste del modelo. Un C-index de 0 indica que el modelo es perfectamente incorrecto, 0.5 significa que es aleatorio, mientras que si es 1 si el modelo es totalmente correcto.")
                    #evaluacion en training
                    train_predict = estimator.predict(X_train)
                    c_index_train = concordance_index_censored(y_train["EVENTO"], y_train["TIEMPO"], train_predict)[0]
                    st.write("El resultado del Harrell's Concordance Index en training es: ", f"{c_index_train:.5f}")
                    
                    #evaluacion en testing
                    test_predict = estimator.predict(X_test)
                    c_index_test = concordance_index_censored(y_test["EVENTO"], y_test["TIEMPO"], test_predict)[0]
                    st.write("El resultado del Harrell's Concordance Index en testing es: ", f"{c_index_test:.5f}")

                    c_index_test= estimator.score(X_test,y_test)
                    if c_index_test > 0.75:
                        st.write("***En vista de que el C-index para el modelo de testing es superior a 0.75, se puede decir que el modelo representa correctamente la realidad.***")
                    elif c_index_test >=0.5:
                        st.write("***En vista de que el C-index para el modelo de testing es superior a 0.5 pero inferior a 0.75, se puede decir que el modelo representa parcialmente la realidad.***")
                    else:
                        st.write("***En vista de que el C-index para el modelo de testing es inferior a 0.5, se puede decir que el modelo es aleatorio.***")
                    #prueba de que variable por si misma tiene mayor predictivo
                    #se le ajusta un modelo de cox a cada variable y se guarda el c_index
                    st.subheader("Feature Selection: Construcción de un modelo que elimine las variables que no contribuyen.")
                    st.write("De acuerdo con Li et al. (2017), el objetivo de feature selection es la construcción de modelos mas simples y comprensibles, que utilizan exclusivamente las variables representativas. De esta manera se consiguen modelos mas eficientes y efectivos, mediante la reducción de la dimensionalidad.")
                    st.write("Según Anurdha y Venkatesh (2019), se busca reducir la cantidad de dimensiones para disminuir la tasa de error causada por los datos redundantes, de esta manera dejando un modelo que utiliza las variables relevantes para predecir con menor error.")
                    st.write("En primera instancia, se le ajusta un modelo de Cox a cada variable individual, y se guarda su c-index para ver cuales tienen mayor poder predictivo.")
                    def fit_and_score_features(X_train, y_train):
                        if hasattr(X_train, "values"):
                            X_train = X_train.values
                        n_features = X_train.shape[1]
                        scores = np.empty(n_features)  #alpha=0.1 agrega regularizacion para evitar matrices singulares
                        for j in range(n_features):
                            Xj = X_train[:, j : j + 1]
                            m = CoxPHSurvivalAnalysis(alpha=0.1)
                            try:
                                m.fit(Xj, y_train)
                                scores[j] = m.score(Xj, y_train)
                            except Exception:
                                scores[j] = 0.5
                        return scores
                    #resultados de que variable individual tiene mayor poder predictivo
                    scores = fit_and_score_features(X_train.values, y_train)
                    feature_scores = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
                    st.write("**Poder predictivo de cada variable individual:** ",feature_scores)
                    #pipeline para indicar los procesos a realizar, primero se debe normalizar despues seleccionar el mejor k para un modelo de cox con regularizacion
                    pipe = Pipeline(
                        [
                            ("scaler", StandardScaler()), #normalizar para evitar los errores de overflow
                            ("select", SelectKBest(fit_and_score_features, k=3)),
                            ("model", CoxPHSurvivalAnalysis(alpha=0.1)), #regularizacion
                        ]
                    )
                    #limitar k para evitar sobreajuste con pocos datos
                    k_max = min(X_train.shape[1], X_train.shape[0] // 10)  #maximo 1 variable por cada 10 observaciones
                    k_max = max(k_max, 1)

                    #definicion de parametros para el grid search
                    param_grid = {"select__k": np.arange(1, k_max + 1)}
                    cv = KFold(n_splits=3, random_state=1, shuffle=True)

                    #gridsearchcv prueba diferentes k para determinar que modelo es el mejor
                    gcv = GridSearchCV(
                        pipe,
                        param_grid,
                        return_train_score=True,
                        cv=cv,
                        error_score=0.0 #si falla el fold vale 0
                        )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning) #eliminar los runtime warning errors y el resto de errores que solo aparecen y no frenan
                        warnings.simplefilter("ignore", RuntimeWarning)
                        gcv.fit(X_train, y_train)
                        #resultados del gridsearch
                        results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)

                    #resultado final
                    pipe.set_params(**gcv.best_params_)
                    pipe.fit(X_train, y_train)

                    #informacion del modelo final
                    st.subheader("Información general del modelo final de Cox: ")
                    transformer, final_estimator = pipe.named_steps["select"], pipe.named_steps["model"]

                    #variables con las que se queda el modelo final reducido
                    selected_features = X_train.columns[transformer.get_support()]

                    #coeficientes del modelo
                    modreduc_coefs = pd.Series(final_estimator.coef_, index=selected_features)
                    st.write(f"**Numero óptimo de variables**: {gcv.best_params_['select__k']}")

                    #C-index del modelo final reducido
                    st.subheader("Evaluación del modelo final: ")
                    st.write(f"El Harrell's Concordance Index del modelo final es: {gcv.best_score_:.4f}")
                    c_index_final = gcv.best_score_
                    if c_index_final > 0.75:
                        st.write("***En vista de que el C-index para el modelo final es superior a 0.75, se puede decir que el modelo representa correctamente la realidad.***")
                    elif c_index_final >=0.5:
                        st.write("***En vista de que el C-index para el modelo de testing es superior a 0.5 pero inferior a 0.75, se puede decir que el modelo representa parcialmente la realidad.***")
                    else:
                        st.write("***En vista de que el C-index para el modelo de testing es inferior a 0.5, se puede decir que el modelo es aleatorio.***")
                    st.subheader("Variables seleccionadas y sus coeficientes:")
                    st.write(modreduc_coefs.sort_values(ascending=False))
                    #hazard ratios del modelo final reducido
                    modreduc_hr = np.exp(modreduc_coefs)
                    st.subheader("Hazard Ratios del modelo final: ")
                    st.write(pd.Series(modreduc_hr,index=selected_features).sort_values(ascending=False))
                    modreduc_hr_series = pd.Series(modreduc_hr, index=selected_features).sort_values(ascending=False)
                    modreduc_hrpositivos = modreduc_hr_series[modreduc_hr_series > 1]
                    modreduc_hrnegativos = modreduc_hr_series[modreduc_hr_series < 1]
                    st.write("Las variables que aumentan la probabilidad de reprobación son (***Valores presentados en porcentaje***): ")
                    st.bar_chart((modreduc_hrpositivos-1)*100, y_label = "Porcentaje de aumento", x_label ="Variables", color ="#6f32a8")
                    st.write("Las variables que disminuyen la probabilidad de reprobación son: ")   
                    st.bar_chart((1-modreduc_hrnegativos)*100, y_label = "Porcentaje de disminución", x_label ="Variables", color ="#a83271")
      
        else:
            st.warning("La base de datos no tiene columnas numéricas para analizar.")

    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {e}")
else:
    st.info("Por favor, carga un archivo para comenzar.")
