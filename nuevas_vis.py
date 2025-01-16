import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm

# Configuración de la página
st.set_page_config(page_title="Análisis de Crímenes en LA", layout="wide")
st.title("Análisis de Crímenes en Los Ángeles")

# Cargar datos
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'crimen23.csv')
    return pd.read_csv(file_path)

df = load_data()

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Seleccione una visualización:",
    ["Mapa de Calor", "Matriz de Correlación", "Crímenes por Mes y Raza", 
     "Distribución por Sexo", "Crímenes por Día", "Análisis por Hora, Raza y Tipo de Arma", "Distribución de Tipo de Crímenes"]
)

# 1. Mapa de Calor con filtro por mes
# 1. Mapa de Calor
if page == "Mapa de Calor":
    st.header("¿Cúal es la concentración de los crímenes en las distintas calles de Los Ángeles?")
    
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df_valid = df.dropna(subset=['LAT', 'LON'])
    df_crime_counts = df_valid.groupby(['LAT', 'LON']).size().reset_index(name='Crimenes Totales')
    
    max_crimes = df_crime_counts['Crimenes Totales'].max()
    
    custom_color_scale = [
        [0, '#00ff00'],      # Verde brillante para valores bajos
        [0.25, '#66ff00'],   # Verde amarillento
        [0.5, '#ffff00'],    # Amarillo
        [0.75, '#ff6600'],   # Naranja
        [1, '#ff0000']       # Rojo brillante para valores altos
    ]
    
    fig = px.density_mapbox(df_crime_counts, 
                           lat='LAT', 
                           lon='LON', 
                           z='Crimenes Totales',
                           hover_name='Crimenes Totales', 
                           color_continuous_scale=custom_color_scale,
                           title="",
                           mapbox_style="open-street-map", 
                           opacity=0.5,
                           zoom=10, 
                           center={'lat': 34.0522, 'lon': -118.2437})
    
    # Actualizar la barra de color para mostrar intervalos de 50
    steps = list(range(0, int(max_crimes) + 50, 50))
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickmode='array',
            tickvals=steps,
            ticktext=[str(i) for i in steps]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 2. Crímenes por Mes y Raza con filtro interactivo
elif page == "Crímenes por Mes y Raza":
    st.header("Según el mes del año ¿Cúal es la cantidad de víctimas de las principales razas? ")

    # Filtro interactivo por Vict Descent
    razas = ["W", "B", "A", "H"]
    razas_seleccionadas = st.sidebar.multiselect(
        "Seleccione una o más razas de víctimas:", 
        options=razas, 
        default=razas
    )

    # Filtrar por razas seleccionadas
    df_filtered = df[df['Vict Descent'].isin(razas_seleccionadas)]
    df_filtered['fecha'] = pd.to_datetime(df_filtered['fecha'], errors='coerce')
    df_filtered['Month'] = df_filtered['fecha'].dt.month

    # Agrupar por mes y calcular el conteo total de crímenes
    crime_counts = df_filtered.groupby('Month').size().reset_index(name='Crime Count')

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=crime_counts, x='Month', y='Crime Count', marker='o', color='blue', linewidth=2)

    # Estética del gráfico
    plt.title(f"Número Total de Crímenes por Mes (Razas Seleccionadas: {', '.join(razas_seleccionadas)})")
    plt.xlabel('Mes')
    plt.ylabel('Número de Crímenes')
    plt.xticks(
        range(1, 13),
        ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    )
    plt.grid(axis='y', linestyle='--',alpha=0.7)
    
    st.pyplot(fig)

# 3. Distribución de Tipo de Crímenes
elif page == "Distribución de Tipo de Crímenes":
    st.header("¿Existe alguna diferencia en el recuento de los cinco principales tipos de crímenes cometidos en Los Ángeles, dependiendo del mes?")
    
    # Filtro por mes
    meses = df['fecha'].str.split('/', expand=True)[0].unique()
    mes_seleccionado = st.sidebar.selectbox("Seleccione un mes:", sorted(meses))

    # Filtro interactivo para categorías de crimen
    crime_categories = [
        'VEHICLE - STOLEN',
        'BATTERY - SIMPLE ASSAULT',
        'THEFT OF IDENTITY',
        'BURGLARY',
        'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)'
    ]
    crime_counts = df[df['Crm Cd Desc'].isin(crime_categories)]

    # Filtrar por mes
    crime_counts['fecha'] = pd.to_datetime(crime_counts['fecha'], errors='coerce')
    crime_counts['Month'] = crime_counts['fecha'].dt.month
    crime_counts = crime_counts[crime_counts['Month'] == int(mes_seleccionado)]

    crime_counts = crime_counts['Crm Cd Desc'].value_counts()

    # Preparar los datos para el radar chart
    values = [crime_counts.get(cat, 0) for cat in crime_categories]
    values += values[:1]  # Repetir el primer valor al final para cerrar el polígono

    # Calcular los ángulos para cada categoría
    angles = np.linspace(0, 2*np.pi, len(crime_categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Completar el círculo

    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Crear el radar chart con puntos en los vértices
    ax.plot(angles, values, 'o-')
    ax.fill(angles, values, alpha=0.25)

    # Configurar las etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(crime_categories, size=8, weight='bold', wrap=True)

    # Ajustar la posición de las etiquetas
    ax.tick_params(pad=40)

    # Añadir los valores en los vértices
    for i, value in enumerate(values[:-1]):
        angle = angles[i]
        radius_factor = 0.9 if crime_categories[i] == 'VEHICLE - STOLEN' else 0.9
        radius = value * radius_factor
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        ax.text(angle, radius, f'{int(value):,}', 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title('Distribución de Tipos de Crímenes', loc='left', pad=20, weight='bold', size=14)
    
    st.pyplot(fig)

# 4. Matriz de Correlación
elif page == "Matriz de Correlación":
    st.header("¿Existe alguna relación entre las variables de: Total de crímenes, el género de la víctima, la raza de la misma y su edad? ")
    
    df['Vict Sex_num'] = df['Vict Sex'].map({'M': 1, 'F': 0, 'U': 2})
    df['Vict Descent_num'] = df['Vict Descent'].map({
        'W': 0, 'H': 1, 'B': 2, 'A': 3, 'O': 4, 'U': 5, 'N': 6, 'L': 7, 'P': 8, 'S': 9
    })
    
    total_crimenes = df.groupby('Vict Age').size().reset_index(name='Total Crimes')
    df = pd.merge(df, total_crimenes, on='Vict Age', how='left')
    df_corr = df[['Vict Age', 'Total Crimes', 'Vict Sex_num', 'Vict Descent_num']].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_corr.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación de Variables')
    
    st.pyplot(fig)

# 5. Distribución por Sexo
elif page == "Distribución por Sexo":
    st.header("¿Cúal es el porcentaje de las víctimas de los distintos crímenes según su sexo?")
    
    df_filtered = df[df['Vict Sex'].isin(['M', 'F', 'X'])]
    crime_counts = df_filtered['Vict Sex'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = cm.Blues(np.linspace(0.5, 0.9, len(crime_counts)))
    
    wedges, texts, autotexts = ax.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%',
            startangle=140, colors=colors,
            textprops={'fontsize': 16, 'weight': 'bold', 'color': 'white'},
            wedgeprops={'edgecolor': 'Black', 'linewidth': 2})
    
    plt.title('Distribución de Crímenes por Sexo de la Víctima')
    ax.legend(wedges, crime_counts.index, title="Sexo de la Víctima", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)

# 6. Análisis por Hora, Raza y Tipo de Arma
elif page == "Análisis por Hora, Raza y Tipo de Arma":
    st.header("¿Qué distribuciones podemos ver con relación al género de la víctima, su edad y su raza?")
    
    def convertir_a_hora(x):
        try:
            if len(str(x)) == 4:
                horas = int(str(x)[:2])
                minutos = int(str(x)[2:])
                return horas + minutos / 60
            return None
        except ValueError:
            return None
    
    df_filtered = df[df['Status Desc'] == "Adult Arrest"].copy()
    df_filtered['TIME OCC'] = df_filtered['TIME OCC'].astype(str).str.zfill(4)
    df_filtered['Hora'] = df_filtered['TIME OCC'].apply(convertir_a_hora)
    df_filtered = df_filtered.dropna(subset=['Hora'])
    
    tipo_grafico = st.radio(
        "Seleccione el tipo de gráfico:",
        ["Histograma de Edad", "Histograma por Sexo", "Gráfico de Pirámide"]
    )
    
    categories = ['W', 'B', 'H', 'A']
    fig, axes = plt.subplots(2, 2, figsize=[12, 12])
    axes = axes.flatten()
    
    if tipo_grafico == "Histograma de Edad":
        for i, descent in enumerate(categories):
            df_subset = df_filtered[df_filtered['Vict Descent'] == descent]
            sns.histplot(df_subset['Vict Age'], kde=True, bins=20, ax=axes[i])
            axes[i].set_title(f'Histograma de Edad - {descent}')
            
    elif tipo_grafico == "Histograma por Sexo":
        df_filtered = df_filtered[df_filtered['Vict Sex'].isin(['M', 'F'])]
        palette = sns.color_palette("Set2", n_colors=2)
        
        for i, descent in enumerate(categories):
            df_subset = df_filtered[df_filtered['Vict Descent'] == descent]
            sns.histplot(df_subset, x='Hora', hue='Vict Sex', kde=True, bins=24,
                        ax=axes[i], multiple='stack', palette=palette)
            axes[i].set_title(f'Histograma por Sexo - {descent}')

    elif tipo_grafico == "Gráfico de Pirámide":
        top_weapons = df['Weapon Desc'].value_counts().nlargest(10).index.tolist()
        conteo = df[df['Weapon Desc'].isin(top_weapons)].groupby(['Weapon Desc', 'Vict Sex']).size().unstack(fill_value=0)

        if 'M' not in conteo.columns:
            conteo['M'] = 0
        if 'F' not in conteo.columns:
            conteo['F'] = 0

        conteo['total'] = conteo['M'] + conteo['F']
        conteo = conteo.sort_values('total', ascending=True)
        conteo = conteo.drop('total', axis=1)

        valores_masculinos = np.array(conteo['M'])
        valores_femeninos = np.array(conteo['F'])

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(top_weapons))
        ax.barh(y_pos, -valores_masculinos, align='center', color='skyblue', label='Masculino')
        ax.barh(y_pos, valores_femeninos, align='center', color='lightpink', label='Femenino')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(conteo.index, wrap=True)
        ax.set_xlabel('Número de Casos')
        ax.set_title('Top 10 Armas más Frecuentes por Género de la Víctima', pad=20)

        ax.axvline(x=0, color='black', linewidth=0.5)

        max_value = max(valores_masculinos.max(), valores_femeninos.max())
        ax.set_xlim(-max_value*1.25, max_value*1.25)

        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{abs(int(x)):,}' for x in xticks])

        ax.legend()

        for i, v in enumerate(valores_masculinos):
            ax.text(-v*1.05, i, f'{int(v):,}', 
                    verticalalignment='center',
                    horizontalalignment='right')
            ax.text(valores_femeninos[i]*1.05, i, f'{int(valores_femeninos[i]):,}',
                    verticalalignment='center',
                    horizontalalignment='left')

    plt.tight_layout()
    st.pyplot(fig)

# 7. Crímenes por día
elif page == "Crímenes por Día":
    st.header("¿Existe alguna relación entre la cantidad de crímenes con víctimas cometidos, con el día del mes?")
    
    # Filtro por Status Desc
    status_options = df['Status Desc'].unique()
    selected_status = st.sidebar.multiselect("Selecciona los Estados", status_options, default=status_options)

    # Filtrar el dataframe según el filtro de Status Desc
    df_filtered = df[df['Status Desc'].isin(selected_status)]

    # Calcular el promedio de crímenes por día
    df_filtered['day'] = pd.to_numeric(df_filtered['day'], errors='coerce')
    avg_crimes_per_day = df_filtered.groupby(['day', 'Status Desc']).size().reset_index(name='Average Crimes')

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=avg_crimes_per_day, x='day', y='Average Crimes',
                hue='Status Desc', palette='Set2', linewidth=2)

    # Títulos y etiquetas
    plt.title('Promedio de Crímenes por Día del Mes (Agrupado por Estado)')
    plt.xlabel('Día del Mes')
    plt.ylabel('Promedio de Crímenes')
    plt.xticks(range(1, 32))
    
    # Mostrar el gráfico
    st.pyplot(fig)
                    
