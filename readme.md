# Análisis de Crímenes en Los Ángeles

Esta aplicación interactiva permite explorar y analizar datos de crímenes en Los Ángeles mediante visualizaciones dinámicas y filtros personalizables. Está diseñada para facilitar la comprensión de patrones de criminalidad en diferentes zonas de la ciudad.

## Funcionalidades Principales

1. **Mapa de Calor**: 
   - Visualiza la concentración de crímenes en diferentes ubicaciones de la ciudad.
   - Incluye un filtro interactivo para ajustar la escala de colores según los datos disponibles.
   - Utiliza `plotly` para proporcionar un mapa interactivo basado en coordenadas geográficas.

2. **Crímenes por Mes y Raza**:
   - Analiza el número de crímenes cometidos cada mes según las principales razas de las víctimas.
   - Gráfico de líneas que permite identificar tendencias estacionales.

3. **Distribución de Tipo de Crímenes**:
   - Examina los principales tipos de crímenes cometidos según el mes seleccionado.
   - Presenta los datos en un gráfico polar para resaltar las diferencias entre categorías.

4. **Matriz de Correlación**:
   - Explora relaciones entre variables como edad, género y raza.
   - Utiliza un heatmap generado con `seaborn` para visualizar las correlaciones.

5. **Distribución por Sexo**:
   - Presenta un análisis visual de los crímenes según el género de las víctimas.
   - Incluye un gráfico de torta interactivo.

6. **Análisis por Hora, Raza y Tipo de Arma**:
   - Examina distribuciones de edad, género y armas utilizadas en los crímenes.
   - Ofrece varias opciones de visualización: histogramas, pirámides de población y gráficos de barras.

7. **Crímenes por Día**:
   - Relaciona la cantidad de crímenes con el día del mes.
   - Incluye filtros para categorías específicas, como "Estado del caso".

## Requisitos del Sistema

- **Python**: Versión 3.9 o superior.
- **Bibliotecas necesarias**: Todas las dependencias están listadas en `requirements.txt`.

## Instalación

Sigue estos pasos para configurar y ejecutar la aplicación en tu entorno local:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/usuario/mi-repositorio.git
   cd mi-repositorio
   ```

2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de que el archivo `crimen23.csv` esté en la misma carpeta que `nuevas_vis.py`.
   - Este archivo contiene los datos utilizados para las visualizaciones.
   - Debe incluir columnas como `LAT`, `LON`, `Vict Descent`, `Vict Sex`, y otras relevantes.

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando:

```bash
streamlit run nuevas_vis.py
```

Esto abrirá una pestaña en tu navegador predeterminado con la interfaz interactiva de la aplicación.

## Interacción

1. **Navegación**:
   - Usa la barra lateral para seleccionar entre diferentes visualizaciones.
   - Cada visualización incluye opciones para filtrar y personalizar los datos mostrados.

2. **Filtros**:
   - Puedes ajustar filtros como:
     - Meses del año.
     - Razas de las víctimas.
     - Tipos de crímenes.
   - Los filtros son interactivos y afectan los datos en tiempo real.

3. **Exploración de gráficos**:
   - Los gráficos son interactivos y permiten acercar, alejar y obtener información detallada al pasar el cursor.

## Problemas Comunes

1. **Error al cargar datos**:
   - Verifica que `crimen23.csv` esté en la ruta correcta.
   - Asegúrate de que el archivo tiene las columnas requeridas y está en formato CSV válido.

2. **Dependencias faltantes**:
   - Si falta alguna biblioteca, vuelve a instalar las dependencias ejecutando:
     ```bash
     pip install -r requirements.txt
     ```

3. **Problemas con caché**:
   - La aplicación utiliza `@st.cache_data` para mejorar el rendimiento. Si actualizas los datos, reinicia la aplicación para que los cambios se reflejen correctamente.

## Contribuciones

Si deseas contribuir al proyecto, sigue estos pasos:

1. Crea un fork del repositorio.
2. Realiza tus cambios en una nueva rama:
   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```
3. Envía un pull request detallando los cambios realizados.


---

¡Gracias por usar la aplicación! Si tienes preguntas o sugerencias, abre un issue en el repositorio para discutirlas.

