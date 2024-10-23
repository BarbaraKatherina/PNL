## Desafios PNL
# Desafío 1: Vectorización de Documentos y Medición de Similaridad

## Vectorización de Documentos
Se realizó la vectorización de los textos, seleccionando 5 documentos al azar para medir su similaridad con el resto del corpus. La representación de los documentos se basó en técnicas como TF-IDF, permitiendo calcular la distancia entre los vectores documentales. Posteriormente, se analizaron los 5 documentos más similares a cada uno de los seleccionados.

### Resultados:
- Los documentos más similares a aquellos pertenecientes a categorías como **soc.religion.christian** o **sci.space** mostraron una fuerte coherencia temática. Los documentos similares también pertenecían en su mayoría a la misma clase, lo que sugiere que la similaridad capturaba correctamente el contenido.
- En otros casos, como el documento de la clase **rec.motorcycles**, los documentos más similares pertenecían a distintas clases (**talk.politics.misc**, **rec.sport.baseball**), lo que indica que la similaridad estaba influenciada por otras características textuales, como el estilo o el tono, más que por el tema explícito.
- En algunas clases, la similaridad capturada fue menos precisa, mostrando diversidad de temas en los documentos similares.

## Entrenamiento de Modelos Naïve Bayes
Se entrenaron modelos de clasificación **Naïve Bayes** (*MultinomialNB* y *ComplementNB*) para maximizar el desempeño en términos del *f1-score macro*. Se realizaron varios experimentos ajustando parámetros como `max_df`, `min_df` y `ngram_range` en el vectorizador, y se comparó el desempeño de los modelos.

### Conclusiones:
- **MultinomialNB** sin ajuste de parámetros logró un *f1-score macro* de `0.5854`.
- Los ajustes en el vectorizador (Prueba 2) no mejoraron los resultados, con un *f1-score macro* de `0.5703`.
- **ComplementNB** (Prueba 3) superó significativamente los otros modelos, logrando un *f1-score macro* de `0.6930`, sugiriendo que es más adecuado para este conjunto de datos.

## Análisis de Similaridad de Términos
Se transpusieron las matrices documento-término para obtener una matriz término-documento, y se analizaron las similaridades entre palabras seleccionadas manualmente, estudiando sus 5 términos más similares. Este análisis permitió explorar cómo el modelo capturaba la relación entre las palabras en función de su coocurrencia en los documentos.

---

# Desafío 2 - Custom Embeddings con Gensim

El objetivo de este desafío fue crear **embeddings** de palabras utilizando un corpus de libros. Los vectores se generaron en función del uso de las palabras dentro de los textos seleccionados, de manera que el espacio de embeddings capturara las relaciones semánticas entre los términos.

## Pasos realizados para obtener los libros en formato .txt:
1. **Descarga de libros:** Se descargaron libros desde textos.info.
2. **Conversión a formato .txt:** Los libros, originalmente en formato PDF, fueron convertidos a archivos .txt.
3. **Organización de archivos:** Todos los libros se colocaron en una carpeta, se comprimieron en un archivo ZIP y se subieron al repositorio.

Durante el desafío, se realizaron las siguientes actividades:
- Se generaron vectores de palabras con Gensim, basados en los contenidos vistos en clase, utilizando un nuevo dataset (los libros seleccionados).
- Se probaron términos de interés y se analizaron las similitudes en el espacio de embeddings.
- Se plantearon y probaron *tests* de analogías para evaluar la coherencia semántica de los embeddings.
- Se graficaron los embeddings resultantes para visualizar la distribución de las palabras en el espacio vectorial.
- Se extrajeron conclusiones a partir de los resultados obtenidos.

### Resultados:
- Se observó que palabras como **"manos"** y **"cabeza"** se encontraban en la misma zona del gráfico, lo que permitió inferir que el modelo capturó correctamente las relaciones semánticas entre términos relacionados con partes del cuerpo. Esto sugiere que el modelo agrupó palabras con significados similares o que frecuentemente aparecen en contextos cercanos dentro de los textos, como descripciones físicas o acciones relacionadas con el cuerpo.
- En resumen, el modelo detectó suficientes co-ocurrencias de estos términos en los textos, permitiendo agruparlos de manera coherente en el espacio de embeddings.

---

# Desafío 3

En el Desafío 3, se realizaron numerosas pruebas, inicialmente trabajando con el libro **Alejandro Dumas - El Conde de Montecristo**. Sin embargo, debido al gran tamaño del archivo, la memoria se agotaba constantemente. Por ello, se decidió probar con libros más pequeños, de aproximadamente 14 mil páginas, obteniendo mejores resultados. También se experimentó con distintos optimizadores, pero se notó que **adam** ofrecía un rendimiento inferior en comparación con **rmsprop**, por lo que se optó por este último. A pesar de probar diferentes configuraciones, siempre se regresaba a las opciones predeterminadas, ya que estas lograban una mejor reducción de la perplejidad. Finalmente, se observó que el modelo funcionaba mejor con un número menor de épocas, siendo más efectivo con 20 o menos.

## Parte 1: Modelo de lenguaje con tokenización por palabras

### Pasos realizados:
1. **Selección del corpus:** Se eligió un corpus de texto para entrenar el modelo.
2. **Preprocesamiento:** Se realizó la tokenización del corpus, y las secuencias de entrenamiento se dividieron entre un conjunto de entrenamiento (80%) y uno de validación (20%).
3. **Propuesta de arquitecturas:** Se diseñaron modelos de redes neuronales basadas en unidades recurrentes para implementar el modelo de lenguaje.
4. **Estrategias de generación de secuencias:** Se probó la generación de secuencias a partir de secuencias de contexto con las estrategias de *greedy search* y *beam search* (determinístico y estocástico), observando el efecto de la temperatura en la generación estocástica.

### Recomendaciones seguidas:
- **Métrica de perplejidad:** Se utilizó la métrica de perplejidad para guiar el entrenamiento y evaluar la calidad de la generación de secuencias.
- **Exploración de hiperparámetros:** Se ajustaron hiperparámetros como el tamaño de contexto máximo, y se utilizaron optimizadores recomendados como **rmsprop**.

## Parte 2: Modelo de lenguaje con tokenización por caracteres

### Pasos realizados:
1. **Selección del corpus:** Se seleccionó un corpus de texto para entrenar el modelo de lenguaje. Dado que el modelo trabaja a nivel de caracteres, todo el corpus puede ser tratado como un solo documento sin necesidad de fragmentarlo.
2. **Pre-procesamiento y tokenización:** Se realizó un pre-procesamiento adecuado del corpus, donde el texto fue tokenizado a nivel de caracteres. A continuación, se estructuró el dataset y se dividió entre datos de entrenamiento y validación.
3. **Arquitectura del modelo:** Se propuso una arquitectura de red neuronal recurrente. En este caso, se utilizaron capas como `CategoryEncoding` para convertir los caracteres en vectores One-Hot Encoded (OHE), y `TimeDistributed` para aplicar la codificación a lo largo de la secuencia de caracteres en el tiempo.
4. **Entrenamiento:** Durante el entrenamiento, se utilizó un *callback* personalizado para calcular la perplejidad en cada epoch.
5. **Optimización:** El optimizador recomendado fue **rmsprop** para asegurar una buena convergencia, aunque se permitió explorar otras opciones.

---

# Desafío 4: Traducción de Texto con Encoder/Decoder

Este desafío consistió en replicar la arquitectura y el esquema básico visto en la clase 7, enfocado en la traducción de texto utilizando un modelo **encoder-decoder**.

## Objetivo:
El objetivo fue utilizar los datos del **Conversational Intelligence Challenge 2 (ConvAI2)** para construir un chatbot (BOT) capaz de responder preguntas del usuario en inglés.

## Pasos realizados:

### Preprocesamiento:
- Se llevó a cabo el preprocesamiento necesario para generar:
  - Índices y longitud máxima de las palabras de entrada.
  - Índices y longitud máxima de las palabras de salida, así como el número total de palabras en la salida.
  - Secuencias de entrada y salida preparadas para el encoder y decoder.

### Preparar los embeddings:
- Se utilizaron los embeddings preentrenados de **GloVe** para transformar los tokens de entrada en vectores que representen sus significados en el espacio de embeddings.

### Entrenamiento del modelo:
- El modelo basado en el esquema **encoder-decoder** fue entrenado utilizando los datos generados en los pasos previos, siguiendo como referencia los ejemplos vistos en clase.

### Inferencia:
- Se experimentó con la inferencia del modelo, evaluando su capacidad para generar respuestas a partir de nuevas entradas.

Este desafío permitió la creación de un sistema básico de preguntas y respuestas utilizando un modelo de traducción de texto, aplicando técnicas avanzadas de procesamiento del lenguaje natural.

# Desafío 5: Clasificación de Sentimientos con BERT

## Descripción del Proyecto

En este desafío, utilizamos **BERT (Bidirectional Encoder Representations from Transformers)** para realizar una tarea de **clasificación de sentimientos**. El objetivo es predecir el sentimiento (negativo, neutral o positivo) de reseñas basadas en texto, utilizando un modelo preentrenado de BERT, y luego ajustar capas adicionales para mejorar el rendimiento en la tarea específica.

## Estructura del Código

1. **Carga y Preprocesamiento de Datos**
    - Se descargan dos archivos CSV: `apps.csv` y `reviews.csv`.
    - La columna `content` contiene las reseñas, mientras que la columna `score` es el puntaje dado a la reseña.
    - Se asignan tres clases de sentimiento basadas en el puntaje:
      - **0**: Sentimiento negativo (puntuación ≤ 2)
      - **1**: Sentimiento neutral (puntuación = 3)
      - **2**: Sentimiento positivo (puntuación ≥ 4)

2. **Tokenización con BERT**
    - Usamos el **tokenizador de BERT** para transformar el texto en secuencias de tokens.
    - Se ajusta el parámetro `max_length`, que define la longitud máxima de texto que BERT puede procesar en una sola instancia. Esto es importante porque BERT tiene una capacidad limitada (512 tokens).

3. **Modelo BERT**
    - Cargamos el modelo base de **BERT (bert-base-uncased)**.
    - Congelamos las capas de BERT para no reentrenarlas, y añadimos capas densas y de dropout adicionales para clasificar el sentimiento.
    - Se utiliza una capa final con **softmax** para obtener probabilidades para cada clase de sentimiento.

4. **Entrenamiento**
    - El modelo se entrena utilizando el optimizador **Adam** y la pérdida **categorical_crossentropy**.
    - Se mide el rendimiento en términos de exactitud, **accuracy**, y **F1-score** para evaluar mejor el desempeño del modelo.

5. **Pruebas Realizadas**
    - Se realizaron varias pruebas variando el parámetro `max_length` y agregando capas ocultas al modelo:
      - **Prueba 1**: `max_length = 140`  
        - **Loss**: 0.94, **Accuracy**: 0.55, **F1-score**: 0.45
      - **Prueba 2**: `max_length = 280`  
        - **Loss**: 0.98, **Accuracy**: 0.50, **F1-score**: 0.49
      - **Prueba 3**: `max_length = 100`  
        - **Loss**: 0.92, **Accuracy**: 0.56, **F1-score**: 0.48
      - **Prueba 4**: `max_length = 200`, agregando dos capas ocultas adicionales  
        - **Loss**: 0.92, **Accuracy**: 0.54, **F1-score**: 0.47

## Resultados

- La **Prueba 2** (max_length = 280) mostró un mejor **F1-score**, lo cual es importante en tareas de clasificación.
- Sin embargo, la **Prueba 4** (max_length = 200 con dos capas ocultas) tuvo un **F1-score** similar, pero con mejor exactitud y menor pérdida, lo que indica que el modelo está aprendiendo mejor las relaciones en los datos.
- A pesar de la pequeña caída en el F1-score, la **Prueba 4** parece ser una mejor opción en general debido a su rendimiento balanceado entre la pérdida y la precisión.

## Conclusiones

A partir de los resultados obtenidos, el enfoque con `max_length = 200` y la adición de capas ocultas parece ofrecer un mejor balance entre exactitud y pérdida. Sin embargo, el **F1-score** aún puede mejorarse ajustando otros parámetros del modelo, como el número de neuronas en las capas densas o la tasa de dropout. La tarea de clasificación de sentimientos en texto es compleja, y seguir ajustando el modelo podría mejorar aún más los resultados.



