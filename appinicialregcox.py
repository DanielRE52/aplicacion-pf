import streamlit as st
import pandas as pd
import numpy as np
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


#nombre de la pagina y config general
st.set_page_config(
    page_title="Analizador de Variables Influyentes - Reprobación de estudiantes",
    layout="wide"
)

#titulo y solicitarle al usuario que cargue el archivo en el formato correcto
st.title("Evaluación de Variables Influyentes en el Rendimiento Académico")
st.info("Proyecto final del departamento de Ingenieria Industrial de la Universidad del Norte. Realizado por Daniel Ramirez, Santiago Reyes y Edwin Yunis.")
st.write("***Carga un archivo CSV o Excel, selecciona las columnas de tiempo y evento, y construye la base para el análisis.***")

#carga de archivo
archivo = st.file_uploader("Selecciona tu base de datos", type=["csv", "xlsx"])

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

        columnas = df.columns.tolist()  #nombres de las variables en la base de datos

        ytime = st.selectbox("Seleccione la variable tiempo", columnas)  #ingreso de variable time
        yevent = st.selectbox("Seleccione la variable evento", columnas)  #ingreso de variable event

        if ytime and yevent:
            if ytime == yevent:  #validacion de errores
                st.error("La variable tiempo y la variable evento deben ser diferentes.")
            else:
                y = Surv.from_dataframe(event=yevent, time=ytime, data=df)  #creacion de y con las variables time y event
                X = df.drop(columns=[ytime, yevent])  #creacion de x con el resto de variables de la base

                st.success("Variables de tiempo y evento cargadas correctamente.")
                st.subheader("Variables predictoras (X)")  #muestra de las variables a las que se les hara analisis
                st.dataframe(X.head())

                st.subheader("Variable de supervivencia (y)")  #muestra de variable de supervivencia
                st.write(y)

                #dummies para variables categoricas
                X = pd.get_dummies(X, drop_first=True)

                #quitar columnas constantes
                X = X.loc[:, X.nunique() > 1]
                st.write("**Presiona el botón 'Ajustar modelo Cox' si la información es correcta.**")

                if st.button("Ajustar modelo Cox"):
                    estimator = CoxPHSurvivalAnalysis(alpha=0.01)
                    estimator.fit(X, y)
                    st.success("Modelo ajustado")
                    st.subheader("Log-hazard Ratios: Coeficientes de la Regresión de Cox")

                    #coeficientes
                    coef = pd.Series(estimator.coef_, index=X.columns)
                    st.bar_chart(coef.sort_values(ascending=False),x_label="Variables", y_label="Valor de los coeficientes", color="#CC00CC")

                    #hazard ratios
                    st.subheader("Hazard Ratios: Magnitud de influencia de las variables")
                    hazard_ratios = np.exp(estimator.coef_)
                    hazard_ratios_series = pd.Series(hazard_ratios, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(hazard_ratios_series,x_label="Variables", y_label="Magnitud de influencia de las variables", color="#192841")
                    st.subheader("Variables más influyentes: ")
                    max5 = hazard_ratios_series.nlargest(8)
                    st.write("Las 8 variables que más aumentan la probabilidad de reprobación son: ", max5)
                    min5= hazard_ratios_series.nsmallest(8)
                    st.write("Las 8 variables que más disminuyen la probabilidad de reprobación son: ", min5)
                    st.info("**Como interpetar los valores**: Un Hazard Ratio superior a 1 indica que la variable aumenta la probabilidad de reprobacion. De misma manera, un Hazard Ratio menor que 1 indica que la variable disminuye la probabilidad de reprobación")
                    st.info("Ej: El Hazard Ratio de la variable ESTADO_ACADEMICO=ESTUDIANTE_DISTINGUIDO es 0.6. Esto implica que si el ESTADO_ACADEMICO del estudiante es Distinguido entonces su probabilidad de reprobación disminuye en un 40%.")
                    #calculo del harrells concordance index para evaluar el modelo
                    st.subheader("Evaluación del Modelo: ")
                    prediction = estimator.predict(X)
                    result = concordance_index_censored(y[yevent], y[ytime],prediction)
                    st.write("**El Resultado del Harrell's Concordance Index es: ", f"{result[0]:.5f}**")
                    c_index = estimator.score(X,y)
                    #ciclo que promptea una explicacion del HCI
                    if c_index > 0.75:
                        st.write("***En vista de que el HCI es superior a 0.75, se puede decir que el modelo representa correctamente la realidad.***")
                    elif c_index >=0.5:
                        st.write("***En vista de que el HCI es superior a 0.5 pero inferior a 0.75, se puede decir que el modelo representa parcialmente la realidad.***")
                    else:
                        st.write("***En vista de que el HCI es inferior a 0.5, se puede decir que el modelo es aleatorio.***")


        else:
            st.warning("La base de datos no tiene columnas numéricas para analizar.")

    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {e}")
else:
    st.info("Por favor, carga un archivo para comenzar.")
