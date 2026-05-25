import streamlit as st
import pandas as pd
import numpy as np
import sksurv
import sklearn
import warnings
import shap
import xgboost as xgb
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report

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
st.divider()
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
        st.dataframe(df.head())
 
        st.subheader("Información general de la base de datos: ")
        st.write(f"Observaciones: {df.shape[0]}")
        st.write(f"Variables: {df.shape[1]}")
 
        st.write("Selecciona las variables de supervivencia:")
        with st.expander("Expande para conocer más sobre que es el análisis de supervivencia y la regresión de Cox"):
            st.subheader("¿Qué es el análisis de supervivencia y la regresión de Cox?")
            st.write('Salinas (2008) define el análisis de la supervivencia de la siguiente manera: "Cuando hablamos de análisis de supervivencia nos referimos al análisis del tiempo de seguimiento (T) de cada unidad de observación hasta que ocurre un fenómeno predefinido (muerte, por ejemplo)". Este método predictivo se usa predominantemente en la medicina para evaluar la evolución de la probabilidad de supervivencia de un paciente, como su nombre lo indica, desde el momento en que el paciente es diagnosticado.')
            st.write('No obstante, el analisis de supervivencia hoy en día es utilizado en una multitud de campos, no solamente en la medicina. Para demostrar la utilidad del modelo, el autor da los siguientes ejemplos: "Las aplicaciones de estos métodos van más allá de solamente evaluar si los individuos viven o no. Existen otros casos donde el fenómeno de interés puede ser analizado con estos métodos, por ejemplo: el tiempo que demora un trabajador expuesto en desarrollar una enfermedad profesional; el tiempo que demora un tratamiento en ser efectivo; el tiempo que demora culminar un trabajo modificando ciertas variables que influyen la productividad; la duración de la lactancia materna, etc". (Salinas, 2008.)')
            st.write('Dentro del análisis de supervivencia uno de los modelos mas utilizados es la regresión de Cox. También conocido como el modelo de riesgos proporcionales de Cox, Taucher (1999) explica que, "este modelo no tiene curva de supervivencia predefinida, pero sí permite ver la influencia de predictores en la respuesta", y Salinas (2008) menciona como el modelo se encarga de, "generar un modelo aproximado y cuantificar la influencia de la(s) variable(s) predictora(s).". Es decir, el análisis de supervivencia en su totalidad tiene como propósito conocer como varia la probabilidad de supervivencia de una entidad desde que sucede un determinado evento. Por otro lado, la regresión de Cox busca cuantificar el impacto de diferentes variables en esta probabilidad.')
            st.write("En el caso del rendimiento estudiantil, la regresión de Cox resulta útil para analizar a nivel grupal cuales son las variables que influyen en la probabilidad de reprobación de los estudiantes, y la magnitud de su influencia.")
            st.divider()
            st.subheader("¿Qué son las variables de supervivencia?")
            st.write("En un modelo de regresión de Cox, estas variables son fundamentales porque **permiten definir cuándo ocurre el evento de interés y si dicho evento ocurrió o no**. En este caso, se deben seleccionar dos variables: una de tiempo y una de evento")
            st.write("La **variable tiempo** representa el periodo transcurrido hasta que se observa el resultado de interés. En el contexto de este proyecto, se interpreta como la cantidad de cortes académicos de la asignatura, ya que se asume que el resultado final del estudiante (aprobar o reprobar) se determina al finalizar el semestre.")
            st.write("Por su parte, la **variable evento** indica si el suceso de interés ocurrió. Para este análisis, el evento corresponde al resultado académico del estudiante en la asignatura, es decir, si aprobó o reprobó.")
 
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
            if ytime == yevent:
                st.error("La variable tiempo y la variable evento deben ser diferentes.")
            else:
                y = Surv.from_dataframe(event=yevent, time=ytime, data=df)
                X = df.drop(columns=[ytime, yevent])
 
                st.success("Variables de tiempo y evento cargadas correctamente.")
 
                #dummies para variables categoricas
                X = pd.get_dummies(X, drop_first=True)
 
                #quitar columnas constantes
                X = X.loc[:, X.nunique() > 1]
 
                #split de training y testing
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
                st.write("**Presiona el botón 'Ajustar modelo Cox' si la información es correcta.**")
 
                if st.button("Ajustar modelo Cox"):
                    estimator = CoxPHSurvivalAnalysis(alpha=0.1)
                    estimator.fit(X_train, y_train)
                    st.success("Modelo ajustado")
                    st.info("***La aplicación presenta directamente el mejor modelo, siendo este el que incluye todas las variables. Al final de la sección podrás conocer acerca de la otra alternativa evaluada.***")
 
                    st.subheader("Ajuste del modelo de Cox incluyendo todas las variables:")
                    st.write("El ajuste del modelo se hace utilizando el conjunto de entrenamiento, y luego se evaluará su desempeño en el conjunto de prueba. Esto permitirá obtener una estimación más realista de la capacidad predictiva del modelo. Para esto se emplean todas las variables de la base.")
                    with st.expander("Presiona aqui si deseas conocer más sobre la importancia de separar los datos en entrenamiento y prueba."):
                        st.subheader("**¿Qué es el training y testing?**")
                        st.write("Separar los datos en conjuntos de training y testing es fundamental para la construcción de modelos predictivos. De acuerdo con los autores Emmert-Streib y Dehmer (2019), el conjunto de entrenamiento se utiliza para estimar o conocer los parámetros de los modelos, también conocido como el ajuste del modelo. Por otro lado, el testing se utiliza para evaluar el rendimiento del modelo entrenado, utilizando este conjunto de datos para la estimación de la generalización del error del modelo final.")
                        st.write("Puesto de otra manera, entrenar el modelo permite que este identifique correctamente patrones y comportamientos, mientras que el testing evalúa el desempeño y la precisión del modelo después del training con datos no vistos, que también son datos reales. En este caso, se utiliza el 70% de los datos para entrenar y el 30% restante se emplea para evaluar.")

                    #coeficientes y hazard ratios
                    coef = pd.Series(estimator.coef_, index=X_train.columns)
                    st.divider()
                    st.subheader("Hazard Ratios: Magnitud de influencia de las variables")
                    hazard_ratios = np.exp(estimator.coef_)
                    hazard_ratios_series = pd.Series(hazard_ratios, index=X_train.columns).sort_values(ascending=False)
                    st.write(hazard_ratios_series)
                    st.info("**Como interpretar los valores**: Un Hazard Ratio superior a 1 indica que la variable aumenta la probabilidad de reprobación. De misma manera, un Hazard Ratio menor que 1 indica que la variable disminuye la probabilidad de reprobación.")
                    st.info("**Ejemplo 1**: Si el Hazard Ratio de la variable EDAD_INGRESO_16_17 es 1.5, esto implica que si un estudiante entro a la Universidad con 16 o 17 años su probabilidad de reprobación aumenta en un 50%.")
                    st.info("**Ejemplo 2**: Si el Hazard Ratio de la variable ESTADO_ACADEMICO=ESTUDIANTE_DISTINGUIDO es 0.6. Esto implica que si el ESTADO_ACADEMICO del estudiante es Distinguido entonces su probabilidad de reprobación disminuye en un 40%.")
                    st.divider()
                    st.subheader("Variables más influyentes: ")
                    max10 = hazard_ratios_series.nlargest(10)
                    max10 = (hazard_ratios_series.nlargest(10)-1)*100
                    st.write("Las 10 variables que más aumentan la probabilidad de reprobación son (***Valores presentados en porcentaje***): ")
                    st.bar_chart(max10, y_label="Porcentaje de aumento", x_label="Variables", color="#000068")
                    min10 = (1-hazard_ratios_series.nsmallest(10))*100
                    st.write("Las 10 variables que más disminuyen la probabilidad de reprobación son (***Valores presentados en porcentaje***): ")
                    st.bar_chart(min10, y_label="Porcentaje de disminución", x_label="Variables", color="#89cff0")
                    st.divider()
                    st.subheader("Evaluación del Modelo: ")
                    st.info("En la regresión de Cox se emplea el Harrell's Concordance Index, o C-index, para validar que tan bueno es el ajuste del modelo. Un C-index de 0 indica que el modelo es perfectamente incorrecto, 0.5 significa que es aleatorio, mientras que si es 1 si el modelo es totalmente correcto.")
                    train_predict = estimator.predict(X_train)
                    c_index_train = concordance_index_censored(y_train["EVENTO"], y_train["TIEMPO"], train_predict)[0]
                    st.write("El resultado del Harrell's Concordance Index en training es: ", f"{c_index_train:.5f}")
                    test_predict = estimator.predict(X_test)
                    c_index_test = concordance_index_censored(y_test["EVENTO"], y_test["TIEMPO"], test_predict)[0]
                    st.write("El resultado del Harrell's Concordance Index en testing es: ", f"{c_index_test:.5f}")
                    c_index_test = estimator.score(X_test, y_test)
                    if c_index_test > 0.75:
                        st.write("***En vista de que el C-index para el modelo de testing es superior a 0.75, se puede decir que el modelo representa correctamente la realidad.***")
                    elif c_index_test >= 0.5:
                        st.write("***En vista de que el C-index para el modelo de testing es superior a 0.5 pero inferior a 0.75, se puede decir que el modelo representa parcialmente la realidad.***")
                    else:
                        st.write("***En vista de que el C-index para el modelo de testing es inferior a 0.5, se puede decir que el modelo es aleatorio.***")


                    st.subheader("Determinación de perfiles de riesgo: ")
                    st.write("Con base a los resultados del modelo de Cox, se puede evidenciar que las 10 variables que más aumentan la probabilidad de reprobación caen en una de las siguientes 4 categorías:")
                    st.write("1. Bajo rendimiento en la asignatura: Estudiantes que pierden al menos 2 de los 3 exámenes.")
                    st.write("2. Bajo rendimiento en el semestre: Estudiantes con el promedio semestral entre 3.5 y 4, o con el promedio semestral relativo menor a 3.5.") 
                    st.write("3. Bajo rendimiento históricamente: Estudiantes con estado académico de periodo de prueba.")
                    st.write("4. Condiciones sociodemográficas: Estudiantes quienes ingresaron con menos de 16 años, o estudiantes quienes financian su educación mediante becas institucionales o empresariales.")
                    st.write("")
                    st.write("De estas 4 categorías, salen los siguientes 6 perfiles de riesgo:")
                    st.write("**Perfil de riesgo 1**: Estudiantes con bajo rendimiento en la asignatura e históricamente.")
                    st.write("**Perfil de riesgo 2**: Estudiantes con bajo rendimiento en la asignatura y en el semestre.")
                    st.write("**Perfil de riesgo 3**: Estudiantes con bajo rendimiento en la asignatura y condiciones sociodemográficas vulnerables de acuerdo con el análisis de Cox.")
                    st.write("**Perfil de riesgo 4**: Estudiantes con bajo rendimiento histórico y condiciones sociodemográficas vulnerables de acuerdo con el análisis de Cox.")
                    st.write("**Perfil de riesgo 5**: Estudiantes con bajo rendimiento histórico y en el semestre.")
                    st.write("**Perfil de riesgo 6**: Estudiantes con bajo rendimiento en el semestre y condiciones sociodemográficas vulnerables de acuerdo con el análisis de Cox.")
                    st.write("Cabe aclarar que un estudiante puede caer en más de un perfil de riesgo, sin embargo, estos son los 6 principales.")
                    st.info("El perfil de riesgo en el que cae cada estudiante individualmente se presenta en la siguiente sección, junto con los resultados de la probabilidad de reprobación estimada y las variables que influyen al estudiante en particular.")
                    st.divider()
                    st.divider()
                    st.header("Análisis individual de las variables de influencia y las probabilidades de reprobación: ")
                    st.write("Para realizar el análisis individual se emplea el modelo de XGBoost, y para mejorar su interpretabilidad se utilizan los SHAP values.")
                    with st.expander("Presiona aqui si deseas conocer más sobre el modelo de XGBoost y los SHAP values."):
                        st.subheader("¿Qué es el modelo de XGBoost?")
                        st.write("XGBoost es un modelo de aprendizaje automático basado en árboles de decisión y técnicas de gradient boosting, diseñado para identificar patrones complejos dentro de los datos y generar predicciones con alta precisión (Chen & Guestrin, 2016). En el contexto del proyecto, este modelo se utiliza para estimar la probabilidad de aprobación o reprobación de un estudiante a partir de las variables académicas, sociodemográficas y de contexto incluidas en la base de datos.")
                        st.write("El resultado principal entregado por XGBoost corresponde a una probabilidad asociada al evento definido en el modelo. Valores cercanos a 1 indican una mayor probabilidad de ocurrencia del evento, mientras que valores cercanos a 0 representan una menor probabilidad. Por ejemplo, una probabilidad de 0.85 indica una alta posibilidad de que ocurra el evento analizado, mientras que una probabilidad de 0.20 representa una baja posibilidad.")
                        st.write("Una de las principales ventajas de este modelo es su capacidad para analizar simultáneamente múltiples variables y capturar relaciones no lineales entre ellas, permitiendo generar predicciones individuales para cada estudiante.")
                        st.divider()
                        st.subheader("¿Qué son los SHAP values?")
                        st.write("SHAP (SHapley Additive exPlanations) es una técnica de interpretación basada en teoría de juegos cooperativos, utilizada para explicar cómo cada variable influye en las predicciones generadas por modelos complejos de aprendizaje automático (Lundberg & Lee, 2017). En este proyecto, SHAP complementa el modelo XGBoost permitiendo comprender el impacto individual de cada variable sobre la probabilidad estimada para cada estudiante.")
                        st.write("Mientras XGBoost entrega la probabilidad final del evento, SHAP permite descomponer dicha predicción en contribuciones individuales de las variables. De esta manera, es posible identificar cuáles factores aumentan o disminuyen la probabilidad de aprobación o reprobación.")
 
                    #limpieza de datos para xgboots 
                    y_xgb = df["EVENTO"].astype(int)
                    X_xgb = df.drop(columns=["EVENTO", "TIEMPO"])
                    X_xgb = pd.get_dummies(X_xgb, drop_first=True)
                    X_xgb = X_xgb.apply(pd.to_numeric, errors="coerce")
                    X_xgb = X_xgb.fillna(X_xgb.median(numeric_only=True))
                    X_xgb = X_xgb.loc[:, X_xgb.nunique() > 1]
 
                    #training testing xgboost
                    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                        X_xgb,
                        y_xgb,
                        test_size=0.25,
                        random_state=42,
                        stratify=y_xgb
                    )
 
                    # xgBoost
                    xgb_model = xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=42
                    )
                    xgb_model.fit(X_train_xgb, y_train_xgb)
 
                    #predicciones
                    y_pred = xgb_model.predict(X_test_xgb)
                    y_prob = xgb_model.predict_proba(X_test_xgb)[:, 1]

                    #guardar para que no se resetee cada vez que uno quiera probar un estudiante diferente
                    st.session_state["xgb_model"] = xgb_model
                    st.session_state["X_test_xgb"] = X_test_xgb
                    st.session_state["y_test_xgb"] = y_test_xgb
                    st.session_state["y_pred"] = y_pred
                    st.session_state["y_prob"] = y_prob


                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X_test_xgb)
                    st.session_state["shap_values"] = shap_values
                    st.session_state["X_xgb_cols"] = X_xgb.columns
 
                    st.subheader("Evaluación del modelo XGBoost:")
                    with st.expander("Presiona aquí si deseas conocer como se hace la evaluación de un modelo de XGBoost."):
                        st.subheader("¿Qué es la métrica AUC-ROC?")
                        st.write("La métrica AUC-ROC (Area Under the Curve – Receiver Operating Characteristic) permite evaluar la capacidad del modelo para diferenciar correctamente entre las dos clases del problema de clasificación binaria. La curva ROC representa la relación entre la tasa de verdaderos positivos (True Positive Rate) y la tasa de falsos positivos (False Positive Rate).")
                        st.write("Por su parte, el valor AUC corresponde al área bajo dicha curva. Su interpretación general es la siguiente:")
                        st.write("AUC cercano a 0.5 = desempeño similar al azar")
                        st.write("AUC entre 0.7 y 0.8 = desempeño aceptable")
                        st.write("AUC entre 0.8 y 0.9 = buen desempeño")
                        st.write("AUC superior a 0.9 = desempeño excelente")
                        
                    st.write(f"**Accuracy:** {accuracy_score(y_test_xgb, y_pred):.4f}")
                    st.write(f"**AUC-ROC:** {roc_auc_score(y_test_xgb, y_prob):.4f}")
                    if roc_auc_score(y_test_xgb, y_prob) >= 0.9:
                        st.write("En vista de que el valor de la curva AUC-ROC es mayor o igual que 0.9, el desempeño del modelo es excelente.")
                    elif roc_auc_score(y_test_xgb, y_prob) >= 0.8:
                        st.write("En vista de que el valor de la curva AUC-ROC es mayor o igual que 0.8, el desempeño del modelo es bueno.")
                    elif roc_auc_score(y_test_xgb, y_prob) >= 0.7:
                        st.write("En vista de que el valor de la curva AUC-ROC es mayor o igual que 0.8, el desempeño del modelo es aceptable.")
                    else:
                        st.write("En vista de que el valor de la curva AUC-ROC es mayor o igual que 0.8, el desempeño del modelo es aleatorio o deficiente.")
                        
                    with st.expander("Presiona aqui si deseas ver la matriz de confusión y el reporte de clasificación."):
                        st.write("**Matriz de confusión:**")
                        st.dataframe(pd.DataFrame(
                            confusion_matrix(y_test_xgb, y_pred),
                            index=["Real: No evento", "Real: Evento"],
                            columns=["Pred: No evento", "Pred: Evento"]
                        ))
                        st.text(classification_report(y_test_xgb, y_pred))
 
                    # SHAP
                    st.write("Análisis con SHAP values: Contribución de cada variable a la probabilidad de una observacion individual:")
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X_test_xgb)
                    shap_importance = pd.DataFrame({
                        "Variable": X_test_xgb.columns,
                        "Importancia_SHAP_promedio": np.abs(shap_values).mean(axis=0)
                    }).sort_values("Importancia_SHAP_promedio", ascending=False)
                    with st.expander("Presiona aqui si deseas conocer las variables mas influyentes segun SHAP."):
                        st.write("**Importancia promedio SHAP (top 15):**")
                        st.dataframe(shap_importance.head(15))

                if "xgb_model" in st.session_state:
                    xgb_model = st.session_state["xgb_model"]
                    X_test_xgb = st.session_state["X_test_xgb"]
                    shap_values = st.session_state["shap_values"]
                        
                    #explicacion individual
                    st.subheader("Explicación individual de estudiantes:")
                    idx = st.number_input("Selecciona el índice del estudiante en el conjunto de prueba:", min_value=0, max_value=len(X_test_xgb)-1, value=0, step=1)
                    idx = int(idx)
                    probabilidad_evento = xgb_model.predict_proba(X_test_xgb.iloc[[idx]])[0, 1]
                    st.write(f"**Probabilidad estimada de reprobación:** {probabilidad_evento:.2%}")
                    explicacion_individual = pd.DataFrame({
                        "Variable": X_test_xgb.columns,
                        "Valor_estudiante": X_test_xgb.iloc[idx].values,
                        "Contribucion_SHAP": shap_values[idx]
                    })
                    explicacion_individual["Magnitud_abs"] = explicacion_individual["Contribucion_SHAP"].abs()
                    explicacion_individual = explicacion_individual.sort_values("Magnitud_abs", ascending=False)
                    st.write("**Top 10 variables más influyentes para este estudiante:**")
                    st.dataframe(explicacion_individual.head(10))
                    st.info("**Como interpretar los valores**: Valor_estudiante representa el valor de la variable para el estudiante seleccionado, con 1 siendo verdadero y 0 siendo falso.")
                    st.info("Los valores SHAP positivos indican que la variable aumenta la probabilidad del evento, y por su contrario los negativos indican que la variable disminuye la probabilidad del evento. La magnitud del valor SHAP representa qué tan fuerte es la influencia de la variable sobre la predicción.")
                    st.info("**Ejemplo 1**: Si la variable P2_MENOR_2 presenta un valor SHAP de +1.5, esto significa que haber obtenido una nota menor a 2 en el segundo corte aumenta significativamente la probabilidad de reprobación del estudiante.")
                    st.info("**Ejemplo 2**: Si la variable PGA_4_5 presenta un valor SHAP de -0.8, esto implica que tener un promedio acumulado entre 4 y 5 reduce la probabilidad de reprobación y favorece la aprobación del estudiante.")

                    #creacion de dataframe para los perfiles de riesgo  
                    df2 = pd.DataFrame(X_test_xgb.values, columns=X_test_xgb.columns)

                    #creacion de variables de perfil de riesgo
                    df2["PERFIL_DE_RIESGO_1"] = (
                        (
                            ((df2["FINAL_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_MENOR_2"] == 1))|
                            ((df2["P2_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["P2_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) 
                        ) &
                        (df2["ESTADO_ACADEMICO_PERIODO_DE_PRUEBA"] == 1)
                    ).astype(int)

                    df2["PERFIL_DE_RIESGO_2"] = (
                        (
                            ((df2["FINAL_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_MENOR_2"] == 1))|
                            ((df2["P2_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["P2_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) 
                        ) &
                        (df2["PROMEDIO_SEM_3.5_4"] == 1) |
                        (df2["PROMEDIO_SEM_REL_MENOR_3.5"] == 1)
                    ).astype(int)

                    df2["PERFIL_DE_RIESGO_3"] = (
                        (
                            ((df2["FINAL_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_2_3"] == 1) & (df2["P2_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_2_3"] == 1)) |
                            ((df2["FINAL_MENOR_2"] == 1) & (df2["P2_MENOR_2"] == 1))|
                            ((df2["P2_2_3"] == 1) & (df2["P1_MENOR_2"] == 1)) |
                            ((df2["P2_MENOR_2"] == 1) & (df2["P1_MENOR_2"] == 1)) 
                        ) &
                        ((df2["EDAD_INGRESO_MENOR_16"] == 1) | (df2["FINANCIAMIENTO_SEMESTRE_BECA_EMP_INST"]==1))
                    ).astype(int)
                    
                    df2["PERFIL_DE_RIESGO_4"] = (
                        (df2["ESTADO_ACADEMICO_PERIODO_DE_PRUEBA"] == 1) &
                        ((df2["EDAD_INGRESO_MENOR_16"] == 1) | (df2["FINANCIAMIENTO_SEMESTRE_BECA_EMP_INST"]==1))
                    ).astype(int)

                    df2["PERFIL_DE_RIESGO_5"] = (
                        (df2["ESTADO_ACADEMICO_PERIODO_DE_PRUEBA"] == 1) &
                        (df2["PROMEDIO_SEM_3.5_4"] == 1) |
                        (df2["PROMEDIO_SEM_REL_MENOR_3.5"]  == 1)
                    ).astype(int)

                    df2["PERFIL_DE_RIESGO_6"] = (
                        ((df2["EDAD_INGRESO_MENOR_16"] ==   1) | (df2["FINANCIAMIENTO_SEMESTRE_BECA_EMP_INST"]==1)) &
                        (df2["PROMEDIO_SEM_3.5_4"] == 1) |
                        (df2["PROMEDIO_SEM_REL_MENOR_3.5"]  == 1)
                    ).astype(int)

                    #ciclos para ver en que perfil de riesgo cae el estudiante y mostrar la información correspondiente
                    st.subheader("Perfil de riesgo del estudiante: ")
                    if df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 1 y 2, es decir, ha tenido un desempeño académico bajo en la asignatura, un bajo rendimiento histórico, y un bajo rendimiento en el semestre.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 1 y 3, es decir, ha tenido un desempeño académico bajo en la asignatura, un bajo rendimiento histórico, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 1 y 4, es decir, ha tenido un desempeño académico bajo en la asignatura, un bajo rendimiento histórico, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 1 y 5, es decir, ha tenido un desempeño académico bajo en la asignatura, un bajo rendimiento histórico, y ha tenido un bajo rendimiento en el semestre.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 1 y 6, es decir, ha tenido un desempeño académico bajo en la asignatura, en el semestre e histórico, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 2 y 3, es decir, ha tenido un desempeño académico bajo en la asignatura y en el semestre, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 2 y 4, es decir, ha tenido un desempeño académico bajo en la asignatura, en el semestre e históricamente, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 2 y 5, es decir, ha tenido un desempeño académico bajo en la asignatura, en el semestre e históricamente.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 2 y 6, es decir, ha tenido un desempeño académico bajo en la asignatura, en el semestre, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 3 y 4, es decir, ha tenido un desempeño académico bajo en la asignatura e históricamente y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 3 y 5, es decir, ha tenido un desempeño académico bajo en la asignatura, en el semestre e históricamente, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 3 y 6, es decir, ha tenido un desempeño académico bajo en la asignatura y en el semestre, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 4 y 5, es decir, ha tenido un desempeño académico bajo en el semestre e históricamente, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece a los Perfiles de Riesgo 4 y 6, es decir, ha tenido un desempeño académico bajo en el semestre e históricamente, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1 & df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 5 y 6, es decir, ha tenido un desempeño académico bajo en el semestre e históricamente, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_1"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 1, es decir, ha tenido un desempeño académico bajo en la asignatura y un bajo rendimiento histórico, en vista de que su estado academico es de periodo de prueba.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_2"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 2, es decir, ha tenido un desempeño académico bajo en la asignatura y en el semestre.")
                        st.warning("Para de este perfil de riesgo se sugiere: ")
                        st.warning("-Realizar seguimiento académico general y proponer tutorías para fortalecer hábitos de estudio y desempeño acumulado.")
                        st.warning("-Analizar la evolución del estudiante durante el semestre y reforzar los momentos donde se evidencia mayor caída.")
                        st.warning("-Revisar los temas evaluados en los cortes con bajo desempeño y asignar actividades de refuerzo específicas.")
                        st.warning("-Planeación de monitorias adicionales focalizadas en las falencias grupales que presenten los estudiantes pertenecientes al perfil de riesgo.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_3"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 3, es decir, ha tenido un desempeño académico bajo en la asignatura y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_4"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 4, es decir, ha tenido un desempeño académico bajo historicamente, al estar en periodo de prueba, y tiene condiciones sociodemograficas vulnerables.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_5"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 5, es decir, ha tenido un desempeño académico bajo historicamente, al estar en periodo de prueba, y ha tenido un bajo rendimiento en el semestre.")
                    elif df2.loc[idx, "PERFIL_DE_RIESGO_6"] == 1:
                        st.warning("Este estudiante pertenece al Perfil de Riesgo 6, es decir, ha tenido un desempeño academico bajo en el semestre, y tiene condiciones sociodemograficas vulnerables.")
                    else:
                        st.success("Este estudiante no pertenece a ninguno de los perfiles de riesgo identificados, lo que implica que no ha tenido un desempeño académico bajo ni tiene condiciones socioeconómicas vulnerables según las variables consideradas.")

                    st.divider()
                    st.divider()
                    st.subheader("Alternativa explorada para la construcción del modelo de Cox: Feature Selection")
                    with st.expander("***Si deseas conocer el detalle de la otra alternativa explorada con la regresión de Cox, puedes expandir esta sección***"):
                        st.subheader("Feature Selection: Construcción de un modelo que reduce la dimensionalidad mediante la eliminación de variables.")
                        with st.expander("Presiona aqui si deseas conocer más sobre el proceso de feature selection."):
                            st.subheader("¿Qué es el proceso de feature selection?")
                            st.write("De acuerdo con Li et al. (2017), el objetivo de feature selection es la construcción de modelos mas simples y comprensibles, que utilizan exclusivamente las variables representativas. De esta manera se consiguen modelos mas eficientes y efectivos, mediante la reducción de la dimensionalidad.")
                            st.write("Según Anurdha y Venkatesh (2019), se busca reducir la cantidad de dimensiones para disminuir la tasa de error causada por los datos redundantes, de esta manera dejando un modelo que utiliza las variables relevantes para predecir con menor error.")
                        st.write("En primera instancia, se le ajusta un modelo de Cox a cada variable individual, y se guarda su c-index para ver cuales tienen mayor poder predictivo.")
                        def fit_and_score_features(X, y):
                            if hasattr(X, "values"):
                                X = X.values
                            n_features = X.shape[1]
                            scores = np.empty(n_features)
                            for j in range(n_features):
                                Xj = X[:, j : j + 1]
                                m = CoxPHSurvivalAnalysis(alpha=0.1)
                                try:
                                    m.fit(Xj, y)
                                    scores[j] = m.score(Xj, y)
                                except Exception:
                                    scores[j] = 0.5
                            return scores
                        scores = fit_and_score_features(X_train.values, y_train)
                        feature_scores = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
                        st.write("**Poder predictivo de cada variable individual (C-index):** ", feature_scores)
                        st.info("En la regresión de Cox se emplea el Harrell's Concordance Index, o C-index, para validar que tan bueno es el ajuste del modelo. Un C-index de 0 indica que el modelo es perfectamente incorrecto, 0.5 significa que es aleatorio, mientras que si es 1 si el modelo es totalmente correcto. En este caso el c-index evalua que tan correcto es el modelo si solo se emplea una variable.")
                        with st.spinner("Realizando proceso de selección de variables relevantes..."):
                            pipe = Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("select", SelectKBest(fit_and_score_features, k=3)),
                                    ("model", CoxPHSurvivalAnalysis(alpha=0.1)),
                                ]
                            )
                            k_max = min(X_train.shape[1], X_train.shape[0] // 10)
                            k_max = max(k_max, 1)
                            param_grid = {"select__k": np.arange(1, k_max + 1)}
                            cv = KFold(n_splits=3, random_state=1, shuffle=True)
                            gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv, error_score=0.0)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", ConvergenceWarning)
                                warnings.simplefilter("ignore", RuntimeWarning)
                                gcv.fit(X_train, y_train)
                                results = pd.DataFrame(gcv.cv_results_).sort_values(by="mean_test_score", ascending=False)
                            pipe.set_params(**gcv.best_params_)
                            pipe.fit(X_train, y_train)
                            st.divider()
                            st.subheader("Información general del modelo reducido de Cox: ")
                            transformer, final_estimator = pipe.named_steps["select"], pipe.named_steps["model"]
                            selected_features = X_train.columns[transformer.get_support()]
                            modreduc_coefs = pd.Series(final_estimator.coef_, index=selected_features)
                            st.write(f"**Numero óptimo de variables**: {gcv.best_params_['select__k']}")
                            st.divider()
                            st.subheader("Evaluación del modelo reducido: ")
                            train_predict_final = pipe.predict(X_train)
                            c_index_train_final = concordance_index_censored(y_train["EVENTO"], y_train["TIEMPO"], train_predict_final)[0]
                            st.write("El resultado del Harrell's Concordance Index en training del modelo reducido es: ", f"{c_index_train_final:.5f}")
                            test_predict_final = pipe.predict(X_test)
                            c_index_test_final = concordance_index_censored(y_test["EVENTO"], y_test["TIEMPO"], test_predict_final)[0]
                            st.write("El resultado del Harrell's Concordance Index en testing del modelo reducido es: ", f"{c_index_test_final:.5f}")
                            if c_index_test_final > 0.75:
                                st.write("***En vista de que el C-index para el modelo reducido en testing es superior a 0.75, se puede decir que el modelo representa correctamente la realidad.***")
                            elif c_index_test_final >= 0.5:
                                st.write("***En vista de que el C-index para el modelo reducido en testing es superior a 0.5 pero inferior a 0.75, se puede decir que el modelo representa parcialmente la realidad.***")
                            else:
                                st.write("***En vista de que el C-index para el modelo reducido en testing es inferior a 0.5, se puede decir que el modelo es aleatorio.***")
                            with st.expander("Presiona aqui si deseas conocer más sobre la evaluación del modelo reducido."):
                                st.subheader("Evaluación del modelo: ")
                                st.write("El c-index es la misma medida de evaluacion que se menciono anteriormente en la seccion del poder predictivo de cada variable. Tal como se menciono, un c-index de 1 indica que el modelo es perfectamente correcto, de 0.5 indica que es aleatorio y de 0 que el modelo es perfectamente incorrecto.")
                            st.divider()
                            st.subheader("Variables seleccionadas y sus Hazard Ratios:")
                            st.info("**Como interpretar los valores**: Un Hazard Ratio superior a 1 indica que la variable aumenta la probabilidad de reprobación. De misma manera, un Hazard Ratio menor que 1 indica que la variable disminuye la probabilidad de reprobación.")
                            modreduc_hr = np.exp(modreduc_coefs)
                            modreduc_hr_series = pd.Series(modreduc_hr, index=selected_features).sort_values(ascending=False)
                            modreduc_hrpositivos = modreduc_hr_series[modreduc_hr_series > 1]
                            modreduc_hrnegativos = modreduc_hr_series[modreduc_hr_series < 1]
                            st.write("Las variables que aumentan la probabilidad de reprobación son (***Valores presentados en porcentaje***): ")
                            st.bar_chart((modreduc_hrpositivos-1)*100, y_label="Porcentaje de aumento", x_label="Variables", color="#6f32a8")
                            st.info("**Ejemplo de interpretación**: Si el Hazard Ratio de la variable EDAD_INGRESO_16_17 es 1.5, esto implica que si un estudiante entro a la Universidad con 16 o 17 años su probabilidad de reprobación aumenta en un 50%.")
                            st.write("Las variables que disminuyen la probabilidad de reprobación son (***Valores presentados en porcentaje***): ")
                            st.bar_chart((1-modreduc_hrnegativos)*100, y_label="Porcentaje de disminución", x_label="Variables", color="#a83271")
                            st.info("**Ejemplo de interpretación**: Si el Hazard Ratio de la variable ESTADO_ACADEMICO=ESTUDIANTE_DISTINGUIDO es 0.6. Esto implica que si el ESTADO_ACADEMICO del estudiante es Distinguido entonces su probabilidad de reprobación disminuye en un 40%.")
                            st.write("Todas las variables del modelo reducido y sus hazard ratios: ")
                            st.write(modreduc_hr_series)
 
    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {e}")
else:
    st.info("Por favor, carga un archivo para comenzar.")
