# Guía de Defensa del Proyecto — CO2 y Cambio de Temperatura Global

> Este documento explica, celda por celda, qué hace cada parte del notebook y cómo interpretar cada resultado y gráfica. El objetivo es que cualquiera del equipo pueda responder preguntas del profesor sin haber escrito el código.

---

## 0. La idea general del proyecto (resúmenla así si preguntan "¿qué hicieron?")

> "Usamos una base de datos pública (Our World in Data) con información de CO2, población y PIB de 254 países desde 1750 hasta 2024. Limpiamos los datos, y aplicamos 3 técnicas de minería de datos: correlación, regresión lineal y clustering, para estudiar la relación entre las emisiones de CO2 de un país y el cambio de temperatura global que se le atribuye."

**La pregunta que respondimos:** ¿Existe una relación medible entre las emisiones de CO2 de un país (totales y per cápita) y el cambio de temperatura que se le atribuye?

**Variable dependiente:** `temperature_change_from_co2` (cambio de temperatura atribuible al CO2, en °C)
**Variables independientes:** `co2` (toneladas totales), `co2_per_capita`, `population`

---

## 1. Carga de datos

```python
import pandas as pd
df = pd.read_csv("../data/owid-co2-data.csv")
df.head()
```

**Qué hace:** Carga el archivo CSV descargado de Our World in Data y lo convierte en una tabla de datos de Python (un "DataFrame"). `.head()` solo muestra las primeras 5 filas para verificar que se cargó bien.

**Si preguntan "¿por qué este dataset?":** Es una fuente pública, confiable, mantenida por una organización reconocida (Our World in Data, en colaboración con el Global Carbon Project), actualizada regularmente y usada en investigación académica real.

---

## 2. Exploración inicial

```python
print(df.shape)
print(df.columns.tolist())
```

**Qué hace:** `.shape` regresa (filas, columnas) — en nuestro caso **(50,411 filas, 79 columnas)**. `.columns.tolist()` imprime el nombre de todas las columnas disponibles.

**Por qué importa:** Antes de limpiar cualquier dataset hay que saber qué tan grande es y qué información contiene.

```python
df.info()
```

**Qué hace:** Muestra, para cada columna, cuántos valores **no nulos** tiene y de qué tipo de dato es (texto, número entero, número decimal). Es la forma más rápida de ver dónde hay huecos en los datos.

**Hallazgo clave:** columnas como `gdp` (PIB) solo tienen 15,251 valores no nulos de 50,411 posibles — es decir, **70% de los datos de PIB faltan**. Esto es esperado: muchos países no llevaban registro de PIB antes de mediados del siglo XX.

```python
df.describe()
```

**Qué hace:** Da estadísticas básicas (promedio, mínimo, máximo, percentiles) de todas las columnas numéricas. Sirve para detectar valores raros, por ejemplo, valores negativos donde no deberían existir, o máximos absurdamente altos.

```python
print(df['country'].nunique())
print(df['year'].min(), '-', df['year'].max())
```

**Qué hace:** Cuenta cuántos países distintos hay (`nunique()` = "number of unique") y muestra el rango de años disponible.

**Resultado:** 254 países/entidades, años de **1750 a 2024**.

---

## 3. Selección de columnas relevantes

```python
cols = ['country', 'year', 'co2', 'co2_per_capita', 'gdp', 'population',
        'temperature_change_from_co2', 'temperature_change_from_ghg']
df_sub = df[cols]
```

**Qué hace:** De las 79 columnas originales, nos quedamos solo con las 8 que necesitamos para nuestra problemática específica. No tiene sentido cargar/limpiar columnas que no vamos a usar (como emisiones de cemento, de gas, etc.).

```python
df_sub.isnull().sum()
```

**Qué hace:** Cuenta cuántos valores nulos (vacíos) tiene cada una de esas 8 columnas.

**Si preguntan "¿por qué tantos nulos?":** Porque el dataset cubre desde 1750, y la mayoría de países no tenían sistemas de medición de CO2, PIB o temperatura en esa época. Los datos modernos (post-1950) están mucho más completos.

---

## 4. Limpieza de datos — el paso más importante para defender

### 4.1 Quitar agregados regionales

```python
regiones = ['World', 'Asia', 'Africa', 'Europe', 'North America', 'South America',
            'Oceania', 'European Union (27)', 'European Union (28)',
            'High-income countries', 'Low-income countries',
            'Lower-middle-income countries', 'Upper-middle-income countries']

df_sub = df_sub[~df_sub['country'].isin(regiones)]
```

**Qué hace:** El dataset original no solo tiene países — también tiene filas para "World" (el mundo entero), "Asia" (todo el continente), grupos de ingreso, etc. Si las dejamos, estaríamos comparando un país (ej. México) contra un continente entero (Asia), lo cual no tiene sentido para nuestro análisis.

**`~` y `.isin()`:** `.isin(regiones)` regresa `True` para las filas cuyo país está en esa lista. El símbolo `~` al inicio significa "lo contrario" (NOT), entonces `df_sub[~...]` se queda con las filas que **NO** son esas regiones — es decir, los países reales.

**Resultado:** de 50,411 filas bajamos a **46,836** (se quitaron 3,575 filas de agregados regionales, multiplicados por todos los años disponibles).

### 4.2 Filtrar por año

```python
df_clean = df_sub[df_sub['year'] >= 1950]
```

**Qué hace:** Nos quedamos solo con años desde 1950 en adelante.

**Por qué 1950 específicamente:** Quien revisó los datos antes de 1950 vio que la cobertura era muy escasa (poquísimos países con PIB o población registrada). 1950 es un punto de corte razonable donde la mayoría de países ya tenían registros más consistentes, sin sacrificar demasiados datos históricos relevantes.

**Resultado:** de 46,836 bajamos a **18,009** filas.

### 4.3 Quitar nulos en las variables clave

```python
df_clean = df_clean.dropna(subset=['co2', 'co2_per_capita', 'temperature_change_from_co2', 'population'])
```

**Qué hace:** `dropna(subset=[...])` elimina cualquier fila que tenga un valor nulo (vacío) en **alguna** de las columnas listadas. Si una fila no tiene dato de CO2, o no tiene cambio de temperatura, no nos sirve para el análisis, así que se elimina.

**Importante — pregunta típica del profesor: "¿por qué no quitaron también los nulos de `gdp`?"**
Respuesta: porque `gdp` tiene muchísimos más nulos que las demás variables (cerca del 70% faltante). Si hubiéramos exigido que `gdp` también esté completo, habríamos perdido miles de filas útiles para los análisis que no necesitan PIB (como correlación entre CO2 y temperatura). Dejamos `gdp` aparte a propósito, como variable "opcional" que se usaría solo si se necesita específicamente.

**Resultado final:** **15,218 filas limpias** — este es el dataset (`df_clean`) que usamos para todo el análisis.

| Etapa | Filas |
|---|---|
| Original | 50,411 |
| Sin regiones agregadas | 46,836 |
| Año ≥ 1950 | 18,009 |
| Sin nulos en variables clave | **15,218** |

---

## 5. Exploración rápida: Top 10 países por CO2 en 2022

```python
top10 = df_clean[df_clean['year']==2022].sort_values('co2', ascending=False).head(10)
top10[['country','co2','co2_per_capita','temperature_change_from_co2']]
```

**Qué hace:** Filtra solo el año 2022, ordena de mayor a menor por emisiones totales de CO2, y muestra los 10 países que más emiten.

**Resultado y por qué importa:** China y Estados Unidos dominan en emisión total, pero si te fijas en `co2_per_capita`, Arabia Saudita tiene el valor más alto del top 10 (20.7 toneladas por persona) a pesar de no ser de los que más emiten en total — esto ya anticipa el hallazgo de que **emisión total** y **emisión per cápita** cuentan historias distintas.

---

## 6. Gráfica de dispersión: CO2 vs Cambio de Temperatura

```python
plt.scatter(df_clean['co2'], df_clean['temperature_change_from_co2'], alpha=0.3)
```

**Qué hace:** Dibuja un punto por cada fila del dataset (15,218 puntos), con el CO2 en el eje X y el cambio de temperatura en el eje Y. `alpha=0.3` hace los puntos semitransparentes para poder ver dónde se acumulan más.

**Cómo interpretar la forma de la gráfica (pregunta muy probable):**
Notarás que la nube de puntos no es uniforme — se ven varias "colas" o trazos curvos que suben. **Cada una de esas colas es un solo país avanzando en el tiempo** (1950 → 2024). Como `temperature_change_from_co2` se calcula a partir de las emisiones **acumuladas** de cada país, mientras más años pasan y más emite un país, más sube tanto su CO2 como su "cambio de temperatura atribuido" — por eso cada país traza su propia línea ascendente en vez de aparecer como ruido disperso.

**Pregunta trampa que puede hacer el profesor:** "¿Por qué la correlación es tan alta, casi perfecta?" — Respuesta honesta: porque la variable de temperatura se construye matemáticamente a partir del CO2 acumulado, entonces la relación fuerte es esperada por cómo se define el dato, no es un "descubrimiento" inesperado. 

---

## 7. Análisis de Correlación

```python
corr_cols = ['co2', 'co2_per_capita', 'population', 'temperature_change_from_co2']
corr_matrix = df_clean[corr_cols].corr()
```

**Qué hace:** `.corr()` calcula el coeficiente de correlación de Pearson entre cada par de columnas numéricas. El resultado es una tabla (matriz) donde cada celda muestra qué tan relacionadas están dos variables.

**Cómo se interpreta el coeficiente (r):**
- Va de -1 a 1.
- Cerca de 1 = correlación positiva fuerte (cuando una sube, la otra también).
- Cerca de -1 = correlación negativa fuerte (cuando una sube, la otra baja).
- Cerca de 0 = no hay relación lineal entre las variables.

```python
import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
```

**Qué hace:** Convierte la tabla de correlación en un mapa de calor — colores más rojos/oscuros indican correlación más fuerte y positiva, azules indican correlación negativa. `annot=True` pone el número exacto dentro de cada celda.

**Resultados obtenidos:**

| Relación | r | Qué significa |
|---|---|---|
| CO2 ↔ Cambio de temperatura | **0.86** | Correlación fuerte positiva |
| CO2 ↔ Población | **0.66** | Correlación moderada-fuerte |
| Población ↔ Cambio de temperatura | **0.48** | Correlación moderada |
| CO2 per cápita ↔ todo lo demás | **≈0.05** | Prácticamente sin relación |

**El hallazgo más importante de esta sección — apréndanselo bien:**
El CO2 **per cápita** casi no correlaciona con nada, mientras que el CO2 **total** sí correlaciona fuerte con población y temperatura. Esto significa que **el tamaño del país (población) pesa más que la "eficiencia" de cada persona** al determinar el impacto climático total de un país. Un país con muchísima gente que emite poco por persona (como India) puede tener un impacto total mayor que un país pequeño con alto consumo per cápita.

```python
from scipy.stats import pearsonr
r, p = pearsonr(df_clean['co2'], df_clean['temperature_change_from_co2'])
```

**Qué hace:** Calcula el mismo coeficiente `r` pero además regresa el **p-value**, que indica si la correlación es estadísticamente significativa (no producto del azar).

**Resultado: r = 0.8595, p ≈ 0.000000**

**Cómo explicar el p-value si preguntan:** Un p-value menor a 0.05 significa que la relación encontrada es estadísticamente significativa — es decir, prácticamente imposible que sea casualidad. Con 15,218 observaciones y un p-value tan cercano a cero, podemos afirmar con mucha confianza que la relación es real dentro de estos datos.

---

## 8. Regresión Lineal

```python
from sklearn.model_selection import train_test_split

X = df_clean[['co2']]
y = df_clean['temperature_change_from_co2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Qué hace:** Prepara los datos para entrenar un modelo predictivo.
- `X` es la variable que usamos para predecir (CO2) — se llama "variable independiente".
- `y` es lo que queremos predecir (cambio de temperatura) — "variable dependiente".
- `train_test_split` divide los datos en dos partes: **80% para entrenar** el modelo (`X_train`, `y_train`) y **20% para probarlo** (`X_test`, `y_test`) con datos que el modelo nunca vio.

**Por qué dividir en train/test (pregunta común):** Si entrenamos y evaluamos con los mismos datos, el modelo podría simplemente "memorizar" en vez de aprender un patrón generalizable. Al evaluar con datos que no vio durante el entrenamiento, comprobamos si realmente aprendió algo útil.

**`random_state=42`:** Es solo una "semilla" para que la división aleatoria sea siempre la misma cada vez que se corre el código (reproducibilidad). El número 42 no tiene significado especial, es solo convención común en programación.

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)
```

**Qué hace:** Crea un modelo de regresión lineal y lo "entrena" (`fit`) con los datos de entrenamiento. El modelo encuentra la línea recta que mejor se ajusta a los puntos.

**Resultado:**
- Pendiente (coeficiente): **0.0000232**
- Intercepto: **0.00088**

**Cómo se lee la ecuación:** `cambio_de_temperatura = 0.00088 + 0.0000232 × CO2`

Esto significa que por cada tonelada adicional de CO2 emitida, el cambio de temperatura atribuible sube en promedio 0.0000232°C. Dicho de otra forma: hacen falta unas 43,000 Mt de CO2 adicionales para sumar 1°C completo.

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
```

**Qué hace:** Usa el modelo ya entrenado para predecir el cambio de temperatura de los datos de prueba (`X_test`), y compara esas predicciones contra los valores reales (`y_test`) usando dos métricas:

- **R² (R cuadrada):** Indica qué porcentaje de la variación en `y` es explicado por el modelo. Va de 0 a 1 (a veces se expresa en %). Nuestro resultado: **0.7245**, es decir, el modelo explica el **72.45%** de la variación del cambio de temperatura usando solo el CO2.

  *Dato interesante para defender:* 0.86² = 0.7396, muy cerca de nuestro R² (0.7245). Esto **no es coincidencia** — matemáticamente, en una regresión lineal simple, el R² es aproximadamente igual al coeficiente de correlación al cuadrado. Si te preguntan por qué coinciden estos números, esa es la razón, y demuestra que los cálculos están bien hechos.

- **RMSE (Root Mean Squared Error / Raíz del Error Cuadrático Medio):** Es el error promedio de predicción, en las mismas unidades que la variable original (°C en este caso). Nuestro resultado: **0.0073°C** — un error pequeño en términos absolutos.

```python
plt.scatter(X_test, y_test, alpha=0.3, label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')
```

**Qué hace:** Grafica los puntos reales de prueba (azul) junto con la línea de predicción del modelo (roja).

**Cómo interpretarla si preguntan:** La línea roja sigue bien la tendencia general, pero notarán que en valores muy altos de CO2 (la zona derecha de la gráfica, donde están China, EE.UU. e India) los puntos reales se alejan más de la línea. Esto sugiere que **la relación real podría no ser perfectamente lineal** en los extremos — una limitación que reconocemos honestamente en el reporte.

---

## 9. Clustering (K-Means)

```python
from sklearn.preprocessing import StandardScaler

cluster_cols = ['co2', 'co2_per_capita', 'population']
X_cluster = df_clean[cluster_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
```

**Qué hace y por qué es necesario "escalar":** `co2` se mide en miles de Mt, `population` en cientos de millones, y `co2_per_capita` en unidades de 1 a 20 aproximadamente. Si no escalamos, el algoritmo de clustering le daría toda la importancia a `population` y `co2` (porque sus números son enormes) e ignoraría casi por completo `co2_per_capita`. `StandardScaler` transforma cada columna para que tenga promedio 0 y desviación estándar 1, poniendo a las 3 variables en pie de igualdad.

**Pregunta típica: "¿por qué no se escala antes, desde la limpieza?"** Porque escalar es un paso específico para el clustering — los otros análisis (correlación, regresión) no lo necesitan, ya que no comparan magnitudes entre variables de la misma forma.

```python
from sklearn.cluster import KMeans

inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1,10), inertias, marker='o')
```

**Qué hace — el "Método del Codo":** Prueba el algoritmo K-Means con diferentes números de grupos (de 1 a 9) y mide qué tan "compactos" quedan los grupos en cada caso (`inertia_` = qué tan dispersos están los puntos respecto al centro de su grupo; menos inercia = grupos más compactos).

**Cómo se elige el número de clusters (k):** Se busca el punto en la gráfica donde la curva deja de bajar tan rápido — el "codo". Antes de ese punto, agregar más grupos mejora mucho los resultados; después, casi no ayuda.

**Nuestro resultado:** El codo aparece claramente en **k = 4** — después de ese punto la curva se aplana notoriamente.

```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)
```

**Qué hace:** Entrena el modelo final con 4 clusters y le asigna a cada fila del dataset un número de cluster (0, 1, 2 o 3) según qué grupo le quedó más cercano.

```python
df_clean.groupby('cluster')[cluster_cols + ['temperature_change_from_co2']].mean()
```

**Qué hace:** Agrupa el dataset por cluster y calcula el promedio de cada variable dentro de cada grupo — esto es lo que nos permite "ponerle nombre" a cada cluster.

**Resultados (memorízense esta tabla, es la más preguntada):**

| Cluster | N (filas) | CO2 prom. (Mt) | CO2/cápita prom. | Población prom. | Δ Temp. prom. (°C) | Perfil |
|---|---|---|---|---|---|---|
| 0 | 15,006 | 61.9 | 4.74 | 15.6 M | 0.0023 | Mayoría de países, bajo impacto |
| 1 | 81 | 6,173.1 | 15.83 | 556.9 M | 0.1575 | Grandes emisores históricos (China, EE.UU.) |
| 2 | 20 | 25.7 | **418.83** | 70.5 mil | 0.00005 | Casos atípicos (ver siguiente sección) |
| 3 | 111 | 1,336.6 | 1.22 | 955.1 M | 0.0263 | Gigantes poblacionales, bajo per cápita (India) |

```python
plt.scatter(df_clean['co2'], df_clean['temperature_change_from_co2'], c=df_clean['cluster'], cmap='viridis')
```

**Qué hace:** El mismo scatter plot de antes, pero ahora coloreado según el cluster al que pertenece cada punto (`c=df_clean['cluster']` le dice a matplotlib que use el número de cluster para elegir el color).

---

## 10. El hallazgo más interesante: investigando el Cluster 2

Este es el momento donde demostramos pensamiento crítico, no solo ejecución de código. **Esta sección es probablemente la que más preguntas genere — domínenla bien.**

```python
df_clean[df_clean['cluster'] == 2][['country', 'year', 'co2_per_capita', 'population']].sort_values('co2_per_capita', ascending=False).head(10)
```

**Qué hace:** Filtra solo las filas del Cluster 2 y las ordena por CO2 per cápita, de mayor a menor, para ver quiénes son exactamente estos casos "raros".

**Primer resultado:** Casi todas las filas resultaron ser **Sint Maarten (parte holandesa)** en los años 1950-1970, con poblaciones de apenas 1,500 a 6,400 personas.

**Por qué pasa esto — explicación que deben poder dar:** Cuando la población de un territorio es extremadamente pequeña, cualquier actividad económica o industrial se divide entre muy poca gente, y el promedio "per cápita" se infla artificialmente. **No es que cada persona en Sint Maarten realmente consumiera/emitiera 700+ toneladas** — es un artefacto del cálculo matemático (dividir entre un número muy chico).

```python
df_clean[df_clean['cluster'] == 2]['population'].describe()
```

**Qué hace:** Da estadísticas (mínimo, máximo, promedio) de la población dentro del Cluster 2 específicamente.

**Resultado sorprendente:** aunque la mayoría son territorios chiquitos (mediana ≈ 2,641 habitantes), el **máximo de población es 1,351,403** — mucho más grande que Sint Maarten. Esto nos hizo investigar más.

```python
df_clean[df_clean['cluster'] == 2][['country', 'year', 'co2_per_capita', 'population']].sort_values('population', ascending=False).head(10)
```

**Qué hace:** Ahora ordenamos por población (de mayor a menor) para encontrar ese caso atípico dentro de los atípicos.

**Resultado: Kuwait, 1991**, con co2_per_capita = 364.79 y población de 1,351,403.

**La explicación histórica (esto es lo que más impresiona si lo explican bien):**
1991 fue el año de la **Guerra del Golfo**. Tras la invasión de Irak a Kuwait, las tropas iraquíes en retirada incendiaron deliberadamente cientos de pozos petroleros kuwaitíes (los conocidos "Kuwaiti oil fires"). Estos incendios ardieron sin control durante meses, generando una cantidad descomunal de emisiones de CO2 para un país con relativamente poca población — disparando su CO2 per cápita a niveles atípicos **solo para ese año**.

**La conclusión que deben poder explicar:**
El Cluster 2 agrupa dos fenómenos completamente distintos bajo la misma etiqueta estadística:
1. **Territorios microscópicos** (Sint Maarten) → artefacto matemático del denominador poblacional pequeño.
2. **Un evento histórico extraordinario** (Kuwait 1991) → un dato real pero excepcional, producto de una guerra, no de un patrón de consumo energético normal.

Esto demuestra que **el clustering agrupa por similitud numérica, sin saber el "por qué" detrás de los números** — la interpretación con contexto histórico/real es responsabilidad nuestra, no del algoritmo.

---

## 11. Preguntas que probablemente haga el profesor (y cómo responderlas)

**P: ¿Por qué eligieron este dataset y no otro?**
R: Es una fuente pública oficial y confiable (Our World in Data + Global Carbon Project), bien documentada, con suficientes registros (50,411 filas originales) para hacer un análisis estadísticamente sólido.

**P: ¿Por qué filtraron desde 1950?**
R: Porque antes de esa fecha la cobertura de datos —sobre todo PIB y población— era muy escasa para la mayoría de países, lo que distorsionaría el análisis.

**P: ¿Por qué la correlación es tan alta (0.86)?**
R: Porque la variable de cambio de temperatura se calcula matemáticamente a partir de las emisiones acumuladas de CO2 de cada país — la relación fuerte es, en parte, consecuencia de cómo se construye el indicador, no un hallazgo inesperado.

**P: ¿Cómo decidieron usar 4 clusters?**
R: Con el método del codo — graficamos la inercia para distintos valores de k, y el "codo" (donde dejó de bajar tan rápido) apareció en k=4.

**P: ¿Qué significa el R² de la regresión?**
R: Que el 72.45% de la variación en el cambio de temperatura se explica solo con el CO2 emitido — un buen ajuste para un modelo de una sola variable.

**P: ¿Cuál fue el hallazgo más interesante?**
R: El Cluster 2, que mezclaba territorios pequeños (artefacto estadístico, ej. Sint Maarten) con un evento histórico real (Kuwait 1991, incendios petroleros de la Guerra del Golfo).

**P: ¿Qué limitaciones tiene su análisis?**
R: (1) La regresión lineal pierde precisión en valores extremos de CO2 (China, EE.UU., India). (2) El clustering no distingue causas reales de artefactos estadísticos — requiere interpretación humana. (3) Dejamos fuera el PIB del análisis principal por tener muchos datos faltantes (70%).

**P: ¿Por qué no usaron PIB en el modelo?**
R: Porque tiene demasiados valores faltantes (solo 15,251 de 50,411 filas originales tienen ese dato) — incluirlo nos habría obligado a descartar miles de filas útiles para los demás análisis.

---

## 12. Glosario rápido de términos usados

| Término | Significado simple |
|---|---|
| DataFrame | Una tabla de datos en Python (como una hoja de Excel) |
| `NaN` / nulo | Un valor vacío o faltante |
| Correlación (r) | Qué tan relacionadas están dos variables, de -1 a 1 |
| p-value | Probabilidad de que un resultado sea casualidad; si es menor a 0.05, se considera significativo |
| Regresión lineal | Un modelo que ajusta una línea recta para predecir una variable a partir de otra |
| R² | Qué porcentaje de la variación es explicado por el modelo (0 a 1) |
| RMSE | Error promedio de predicción, en las mismas unidades que el dato original |
| Train/test split | Dividir los datos en una parte para entrenar el modelo y otra para evaluarlo |
| Clustering / K-Means | Agrupar datos similares entre sí en "k" grupos |
| StandardScaler | Pone todas las variables en la misma escala antes de comparar |
| Método del codo | Técnica gráfica para elegir cuántos clusters (k) usar |
| Outlier / atípico | Un dato que se sale mucho del patrón general |
