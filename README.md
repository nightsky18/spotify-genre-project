# Spotify Genre Classifier

Sistema de clasificación de géneros musicales desarrollado con scikit-learn, FastAPI y Docker.
El modelo predice el género de una canción a partir de sus características acústicas extraídas del dataset de Spotify.

> **Actividad 03 - Aplicación de modelos de ML en Python y prototipado usando Orange (15%)**
> Materia: ING01216 - Aprendizaje de Máquina y Computación Evolutiva
> Docente: Jeison Alejandro Zapata Pulgarín

---

## Integrantes

- María José Arcila Cano
- Sebastián Lopez Osorio
- Mariana Montoya Sepúlveda

---

## Despliegue público

| Servicio | URL |
|----------|-----|
| Dashboard (Streamlit) | http://72.60.68.22:8014 |
| API REST (FastAPI) | http://72.60.68.22:8013 |
| Documentación interactiva (Swagger) | http://72.60.68.22:8013/docs |

> El servicio está desplegado en un servidor público mediante Docker y Docker Compose, accesible desde cualquier ubicación en internet.

---

## Estructura del proyecto

```
spotify-genre-project/
├── app/
│   ├── __init__.py
│   ├── main.py               # API con FastAPI
│   └── streamlit_app.py      # Dashboard con Streamlit
├── models/
│   └── gb_pipeline.pkl       # Modelo entrenado (Pipeline)
├── notebooks/
│   └── 01_modelo.ipynb       # Notebook principal de análisis y modelado
├── spotify_songs.csv         # Dataset original
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Configuración del entorno local

### Requisitos

- Python 3.11 (recomendado para compatibilidad con todas las librerías)
- pip actualizado

### 1. Crear el entorno virtual

```bash
python -m venv .venv_clean
```

### 2. Activar el entorno

En Windows:
```bash
.venv_clean\Scripts\activate
```

En Linux/Mac:
```bash
source .venv_clean/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** PyCaret solo soporta Python 3.9, 3.10 y 3.11. Si usas Python 3.12, la instalación fallará con un `RuntimeError`. Ver sección [PyCaret](#punto-e--pycaret).

---

## Ejecutar la API localmente

Desde la raíz del proyecto:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre en el navegador:

```
http://127.0.0.1:8000/docs
```

---

## Ejecutar con Docker

### Requisitos

- Docker instalado
- Docker Compose instalado

### Levantar todos los servicios

Desde la carpeta del proyecto:

```bash
docker compose up --build
```

| Servicio | Puerto local | Descripción |
|----------|-------------|-------------|
| `spotify-genre-api` | `8013` | API REST con FastAPI |
| `spotify-genre-dashboard` | `8014` | Dashboard visual con Streamlit |

### Acceder a los servicios

```
http://localhost:8013       -> API REST
http://localhost:8013/docs  -> Swagger UI (documentación interactiva)
http://localhost:8014       -> Dashboard Streamlit
```

### Detener los servicios

```bash
docker compose down
```

---

---

# Informe - Actividad 03

## Punto B - Aplicación de modelos de ML en Python

---

### 1. Data Cleaning

Se identificaron y aplicaron los siguientes aspectos de limpieza sobre el dataset `spotify_songs.csv`:

- **Eliminación de columnas no numéricas:** Se removieron columnas de texto e identificadores (`track_id`, `track_name`, `track_album_id`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_id`, `playlist_subgenre`, `track_artist`) que no aportan información cuantitativa al modelo.
- **Revisión de valores nulos:** Se identificaron y eliminaron filas con valores nulos en cualquiera de las variables restantes.
- **Verificación de tipos de datos:** Se confirmó que todas las variables independientes son numéricas (`float64` o `int64`).
- **Revisión de estadísticas descriptivas:** Se utilizó `.describe()` para detectar rangos anómalos o valores extremos en variables como `loudness`, `tempo` y `durationms`.

---

### 2. Descripción del dataset y análisis de correlación

El dataset contiene registros de canciones de Spotify con características acústicas extraídas automáticamente por la plataforma.

**Variable objetivo (dependiente):** `playlist_genre`

| Género | Descripción |
|--------|-------------|
| edm    | Electronic Dance Music |
| latin  | Música latina |
| pop    | Pop |
| r&b    | Rhythm and Blues |
| rap    | Rap / Hip-Hop |
| rock   | Rock |

**Variables independientes seleccionadas:**

| Variable | Descripción |
|----------|-------------|
| `trackpopularity` | Popularidad de la canción (0-100) |
| `danceability` | Qué tan bailable es (0.0-1.0) |
| `energy` | Intensidad percibida (0.0-1.0) |
| `key` | Tonalidad musical (0-11) |
| `loudness` | Volumen promedio en dB |
| `mode` | Modalidad mayor/menor (0 o 1) |
| `speechiness` | Presencia de palabras habladas |
| `acousticness` | Probabilidad de ser acústica |
| `instrumentalness` | Ausencia de voz |
| `liveness` | Probabilidad de grabación en vivo |
| `valence` | Positividad musical (0.0-1.0) |
| `tempo` | Tempo en BPM |
| `durationms` | Duración en milisegundos |

**Análisis de correlación:**
El mapa de calor generado con `seaborn` mostró que ninguna variable tiene una correlación lineal fuerte con el género musical, lo cual es esperado dado que la variable objetivo es categórica. Variables como `energy`, `acousticness` y `danceability` presentaron mayor varianza entre géneros y fueron más relevantes para el modelo. Variables como `key` y `mode` presentaron correlaciones muy bajas entre sí y con el target, pero se retuvieron por su potencial aporte en modelos no lineales como GradientBoosting. Se identificó correlación negativa entre `energy` y `acousticness` (-0.73), lo que indica que las canciones más enérgicas tienden a ser menos acústicas.

**Conjuntos de entrenamiento y prueba:**

- Entrenamiento: 80% de los datos
- Prueba: 20% de los datos
- División realizada con `train_test_split(random_state=42, stratify=y)` para mantener la proporción de clases.

**Conclusiones sobre el dataset:**

El dataset de Spotify es rico en variables acústicas continuas, pero presenta una limitación estructural: los géneros musicales no tienen fronteras acústicas rígidas. Géneros como `pop`, `r&b` y `latin` comparten rangos similares en variables como `danceability` y `energy`, lo que hace que cualquier modelo de clasificación enfrente solapamiento real entre clases. La distribución de clases está relativamente equilibrada (aproximadamente 5.000 canciones por género), por lo que no se requirió balanceo artificial. El dataset tiene 13 variables numéricas útiles tras la limpieza, lo que representa una dimensionalidad manejable sin necesidad de reducción.

---

### 3. Ingeniería de características

**No fue necesario aplicar ingeniería de características compleja.**

- Las variables disponibles en el dataset ya son representaciones acústicas directamente interpretables y procesadas por la API de Spotify.
- Se descartó la creación de variables derivadas por la naturaleza ya normalizada del dataset.
- El único proceso aplicado fue el escalado de variables con `StandardScaler`, incluido dentro del `Pipeline` para garantizar consistencia entre entrenamiento y predicción.
- Se evaluó PCA, pero dado que no hay alta dimensionalidad (13 variables) y los modelos de árbol no requieren escalado para funcionar, no se justificó su aplicación.

---

### 4. Modelo con mejor desempeño

El modelo seleccionado fue **GradientBoostingClassifier** de scikit-learn, implementado dentro de un Pipeline con los siguientes hiperparámetros:

```python
gb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        random_state=42
    ))
])
```

También se evaluó RandomForestClassifier con el siguiente pipeline:

```python
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])
```

GradientBoosting fue seleccionado tras comparar múltiples algoritmos (Logistic Regression, Decision Tree, Random Forest, GradientBoosting) evaluando accuracy y F1-score sobre el conjunto de prueba.

---

### 5. Matriz de confusión

La matriz de confusión se generó sobre el conjunto de prueba con las 6 clases de género:

```
Clases: ['edm', 'latin', 'pop', 'r&b', 'rap', 'rock']
```

La diagonal de la matriz concentra los valores más altos, indicando que el modelo clasifica correctamente la mayoría de las instancias. Los géneros con mayor confusión entre sí fueron `pop` y `r&b`, lo cual es coherente con su similitud acústica (alta danceability y energía moderada). El género `rap` fue el más fácilmente separable, posiblemente por su alta `speechiness`.

> La imagen de la matriz de confusión se encuentra en el notebook `notebooks/01_modelo.ipynb`.

---

### 6. Métricas de efectividad

| Métrica | GradientBoosting | RandomForest |
|---------|-----------------|--------------|
| Accuracy | ~0.59 | ~0.56 |
| Precision (weighted) | ~0.58 | ~0.56 |
| Recall (weighted) | ~0.59 | ~0.56 |
| F1-score (weighted) | ~0.58 | ~0.56 |

Las métricas fueron calculadas con `classification_report` de scikit-learn sobre el conjunto de prueba (20% de los datos).

---

### 7. Conclusiones del modelado

- GradientBoosting fue el modelo con mejor desempeño general, con un accuracy de ~0.59 sobre datos de prueba, superando a RandomForest (~0.56) y a los demás algoritmos evaluados.
- El problema presenta alta dificultad inherente: los géneros musicales no tienen fronteras acústicas perfectamente definidas y existe solapamiento real entre géneros como `pop`, `r&b` y `latin`.
- La variable `acousticness` resultó ser la más discriminante para separar `rock` y `edm` del resto, mientras que `speechiness` fue clave para identificar `rap`.
- `key` y `mode` aportaron poco en modelos lineales, pero GradientBoosting pudo extraer interacciones no lineales útiles de estas variables gracias a su construcción secuencial de árboles.
- El uso de `n_estimators=200` y `max_depth=4` en GradientBoosting permitió capturar patrones más complejos sin sobreajuste excesivo, en comparación con configuraciones más simples evaluadas previamente.
- **Método seleccionado para implementación:** GradientBoostingClassifier dentro de un Pipeline con StandardScaler, por ser el de mayor F1-score ponderado y mayor estabilidad frente a los demás modelos evaluados.

---

## Punto C - FastAPI

Se implementó una API REST con FastAPI que expone el modelo entrenado para realizar predicciones en tiempo real desde cualquier ubicación en internet.

### Qué es FastAPI

FastAPI es un framework moderno de Python para construir APIs REST de alto rendimiento, basado en anotaciones de tipo y compatible con el estándar OpenAPI. Genera documentación interactiva automática con Swagger UI, lo que facilita el consumo y la prueba del servicio sin necesidad de clientes externos (Ramírez, 2018).

### Implementación

**POST** `/predict`

**Body de entrada (JSON):**
```json
{
  "trackpopularity": 65.0,
  "danceability": 0.72,
  "energy": 0.85,
  "key": 5.0,
  "loudness": -5.3,
  "mode": 1.0,
  "speechiness": 0.08,
  "acousticness": 0.12,
  "instrumentalness": 0.0,
  "liveness": 0.11,
  "valence": 0.65,
  "tempo": 128.0,
  "durationms": 210000.0
}
```

**Respuesta:**
```json
{
  "playlistgenre": "pop"
}
```

### Acceso público

El servicio quedó disponible públicamente mediante despliegue en servidor con Docker:

| Recurso | URL |
|---------|-----|
| API base | http://72.60.68.22:8013 |
| Documentación Swagger | http://72.60.68.22:8013/docs |
| Dashboard visual | http://72.60.68.22:8014 |

---

## Punto D - Pipeline (scikit-learn)

### Para qué sirve Pipeline

`Pipeline` de scikit-learn permite encadenar múltiples pasos de procesamiento y modelado en un único objeto. Cada paso recibe la salida del anterior, garantizando que la misma transformación aplicada en entrenamiento se aplique exactamente igual en predicción (Buitinck et al., 2013).

**Ventajas principales:**
- Evita *data leakage*: el escalado se ajusta solo con datos de entrenamiento y se aplica a los de prueba sin filtración de información.
- Simplifica el despliegue: se serializa un solo objeto `.pkl` que incluye preprocesamiento y modelo.
- Compatible con `cross_val_score`, `GridSearchCV` y `joblib`.

### Implementación

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

gb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        random_state=42
    ))
])

gb_pipeline.fit(X_train, y_train)
joblib.dump(gb_pipeline, "models/gb_pipeline.pkl")
```

El pipeline serializado con `joblib` es el archivo cargado directamente por la API de FastAPI para realizar predicciones.

---

## Punto E - PyCaret

### Qué es PyCaret

PyCaret es una librería de AutoML de bajo código para Python que automatiza la comparación, ajuste y evaluación de múltiples modelos de machine learning con pocas líneas de código (Ali, 2020).

### Resultado en el proyecto

**No fue posible usar PyCaret** porque la librería soporta oficialmente Python 3.9, 3.10 y 3.11. El entorno de desarrollo utilizó Python 3.12, lo que generó el siguiente error al intentar importarla:

```
RuntimeError: Pycaret only supports python 3.9, 3.10, 3.11.
Please DOWNGRADE your Python version.
```

Por esta razón, la comparación de modelos se realizó manualmente con scikit-learn, que ofrece compatibilidad completa con Python 3.12 y mayor control sobre el proceso de evaluación.

**Qué hubiera aportado PyCaret:**
- Comparación automática de más de 20 modelos con `compare_models()`.
- Generación automática de métricas, gráficas y matrices de confusión.
- Ajuste de hiperparámetros con `tune_model()`.
- Exportación directa del mejor modelo con `save_model()`.

---

## Punto F - Ensamble

### Implementación

Se implementó un `VotingClassifier` con tres estimadores base:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

ensemble = VotingClassifier(estimators=[
    ("lr", lr_pipeline),
    ("rf", RandomForestClassifier(random_state=42)),
    ("dt", DecisionTreeClassifier(random_state=42))
], voting="hard")

ensemble.fit(X_train, y_train)
```

### Comparación de resultados

| Modelo | Accuracy | F1-score (weighted) |
|--------|----------|---------------------|
| GradientBoostingClassifier (Pipeline) | ~0.59 | ~0.58 |
| RandomForestClassifier (Pipeline) | ~0.56 | ~0.56 |
| VotingClassifier (ensamble) | ~0.44 | ~0.43 |

### Conclusiones del ensamble

- El `VotingClassifier` no superó al mejor modelo individual. Su accuracy fue aproximadamente 15 puntos porcentuales inferior al de GradientBoosting.
- La regresión logística, uno de los tres estimadores del ensamble, presentó advertencias de convergencia incluso con `max_iter=1000`, lo que indica que el problema no es linealmente separable y ese componente aportó predicciones de baja calidad al conjunto.
- El `DecisionTreeClassifier` sin restricciones de profundidad tiende al sobreajuste, lo que generó predicciones inestables que afectaron la votación mayoritaria.
- El ensamble por votación dura (`hard voting`) no es el más adecuado cuando los estimadores tienen calidades muy diferentes: un estimador débil tiene el mismo peso que uno fuerte, lo que diluye el aporte del mejor modelo.
- Un ensamble con votación suave (`soft voting`) o un `StackingClassifier` podrían haber dado mejores resultados, pero requieren que todos los estimadores entreguen probabilidades calibradas.
- **Método seleccionado para implementación:** GradientBoostingClassifier dentro de un Pipeline, por su mayor accuracy, F1-score y estabilidad comprobada frente al ensamble evaluado.

---

## Conclusiones generales del trabajo

**Sobre el dataset:**
El dataset de Spotify representa un caso real de clasificación multiclase con solapamiento natural entre categorías. La principal dificultad no está en la cantidad de datos ni en la calidad del preprocesamiento, sino en la naturaleza del problema: los géneros musicales son convenciones culturales, no categorías acústicamente puras. Esto establece un techo de desempeño inherente que ningún modelo puede superar con estas variables sin incorporar información adicional como letra, artista o contexto cultural.

**Sobre el proceso de modelado:**
GradientBoosting demostró ser el algoritmo más adecuado para este problema por su capacidad de capturar interacciones no lineales entre variables y su resistencia al ruido en los datos. La construcción secuencial de árboles, donde cada árbol corrige los errores del anterior, le permite manejar mejor los patrones complejos que definen géneros como `rap` (alta speechiness) o `edm` (alta energy y baja acousticness). El uso de Pipeline garantizó reproducibilidad y consistencia en el despliegue.

**Sobre el ensamble:**
Los métodos de ensamble no garantizan mejoras automáticas. Su efectividad depende de que los estimadores base sean diversos y de calidad comparable. En este caso, incluir regresión logística (modelo lineal en un problema no lineal) y un árbol de decisión sin restricciones generó un ensamble menos efectivo que el mejor modelo individual. Para futuras iteraciones, un ensamble de GradientBoosting con RandomForest y XGBoost podría ser más competitivo.

**Sobre el despliegue:**
La implementación de FastAPI dentro de un contenedor Docker permitió exponer el modelo como un servicio REST accesible desde cualquier ubicación. El uso de Pipeline como unidad de serialización garantiza que el preprocesamiento y el modelo se apliquen de forma consistente en producción, eliminando riesgos de inconsistencia entre el entrenamiento y la inferencia.

---

## Estado del proyecto

| Componente | Estado |
|------------|--------|
| Data Cleaning | Completado |
| Análisis exploratorio | Completado |
| Análisis de correlación | Completado |
| Modelado con scikit-learn | Completado |
| Matriz de confusión | Completado |
| Métricas de efectividad | Completado |
| Pipeline | Completado |
| Guardado del modelo (joblib) | Completado |
| FastAPI (local) | Completado |
| FastAPI (público en servidor) | Completado - http://72.60.68.22:8013 |
| Dashboard Streamlit (público) | Completado - http://72.60.68.22:8014 |
| Despliegue con Docker | Completado |
| Ensamble (VotingClassifier) | Completado |
| PyCaret | Incompatible con Python 3.12 |
| Ingeniería de características avanzada | No aplicada (no fue necesaria) |

---

## Pendientes / Limitaciones

- **PyCaret:** Requiere Python 3.10 o 3.11. No compatible con el entorno utilizado (Python 3.12).
- **Ajuste de hiperparámetros sistemático:** No se aplicó `GridSearchCV` ni `RandomizedSearchCV` sobre el modelo final. Queda como mejora futura.
- **Balanceo de clases:** Se puede explorar SMOTE si el desbalance afecta el rendimiento en producción.
- **Ensamble mejorado:** Un `StackingClassifier` o un ensamble con estimadores de calidad comparable podría superar el resultado de GradientBoosting individual.

---

## Referencias

- Ali, M. (2020). *PyCaret: An open source, low-code machine learning library in Python*. https://www.pycaret.org
- Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. *ECML PKDD Workshop: Languages for Data Mining and Machine Learning*, 108-122.
- FastAPI. (s. f.). *Tutorial - User Guide*. https://fastapi.tiangolo.com/tutorial/
- PyCaret. (2024). *Installation*. https://pycaret.readthedocs.io/en/stable/installation.html
- PyCaret. (2024). *PyPI*. https://pypi.org/project/pycaret/
- Ramírez, S. (2018). *FastAPI*. https://fastapi.tiangolo.com
- scikit-learn. (s. f.). *Pipeline*. https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- scikit-learn. (s. f.). *VotingClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
- Docker. (s. f.). *Docker Compose*. https://docs.docker.com/compose/
