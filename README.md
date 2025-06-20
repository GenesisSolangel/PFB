Proyecto de Fin de Bootcamp: Red Eléctrica Española (REE)

Este proyecto es una aplicación interactiva desarrollada en Streamlit que permite visualizar, analizar y predecir la demanda eléctrica registrada por la Red Eléctrica de España (REE), enfocándose en indicadores clave como demanda, generación, balance eléctrico e intercambios.

## Objetivos de la aplicación
  - Explorar la evolución de estos aspectos en diferentes periodos de tiempo.
  - Comparar métricas de demanda entre años específicos y detectar posibles años atípicos (outliers).
  - Realizar predicciones de los valores de demanda mediante diferentes modelos de deep learning.

## ¿Cómo funciona?
  - **Supabase:** Se usa como base de datos para almacenar y consultar la información histórica de forma eficiente.
  - **Kaggle:** Se emplea como entorno de entrenamiento para los modelos de predicción (RNN y Prophet).
  - **Streamlit:** Permite el desarrollo de esta app en la nube, ofreciendo interacción por parte del usuario.""")
        
## Secciones de navegación:
  - **Descripción:** Página de inicio con la descripción general del proyecto.
  - **Filtros de consulta de datos:** Barra lateral para filtrar los datos según el interés del usuario.
  - **Visualización:** Análisis gráfico de los aspectos históricos comentados anteriormente.
  - **Comparador:** Comparación de la demanda entre dos años seleccionables y detección de outliers.
  - **Predicciones RNN:** Predicciones generadas mediante Redes Neuronales Recurrentes.
  - **Predicciones Prophet:** Predicciones usando el modelo Prophet de Facebook.
  - **Extras:** Análisis complementarios de interés.
  - **Quiénes somos:** Información sobre el equipo.

## Instalación y ejecución
  - Clona este repositorio e instala las librerías dentro de requirements.txt.
  - Crea una cuenta en Supabase y configura tus credenciales para conectarte a esta.
  - Crea una cuenta de streamlit y conéctala con tu repositorio clonado.
  - Ejecuta el archivo Streamlit_REE_auto.py. Al ejecutar el script se inicia la descarga de los datos y el resultado se almacena en un DataFrame de Pandas.
