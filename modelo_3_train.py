"""
import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === Rutas de las carpetas ya extraídas ===
ruta_actividades = 'movement/'
ruta_niveles = 'uploads/'

# === Características que deben contener los CSV ===
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# === Función para cargar archivos CSV con una etiqueta ===
def cargar_datos(ruta, prefijo, etiqueta):
    archivos = glob.glob(os.path.join(ruta, f'{prefijo}_*.csv'))
    dataframes = []
    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
            if set(features).issubset(df.columns):
                df['label'] = etiqueta
                dataframes.append(df)
            else:
                print(f"[!] Ignorado (faltan columnas): {archivo}")
        except Exception as e:
            print(f"[!] Error leyendo {archivo}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# === Cargar datos para Modelo 1 (tipo de movimiento) ===
df_voluntario = cargar_datos(ruta_actividades, 'drinkin_water', 'voluntario')
df_involuntario = cargar_datos(ruta_actividades, 'drinkin_waterP', 'involuntario')
df_tipo = pd.concat([df_voluntario, df_involuntario], ignore_index=True)

# === Cargar datos para Modelo 2 (nivel de temblor) ===
df_bajo = cargar_datos(ruta_niveles, 'bajo', 'bajo')
df_medio = cargar_datos(ruta_niveles, 'medio', 'medio')
df_brusco = cargar_datos(ruta_niveles, 'brusco', 'brusco')
df_nivel = pd.concat([df_bajo, df_medio, df_brusco], ignore_index=True)

# === ENTRENAR MODELO 1 ===
X1 = df_tipo[features]
y1 = df_tipo['label']
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, stratify=y1, random_state=42)
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf1.fit(X1_train, y1_train)
y1_pred = clf1.predict(X1_test)

print("=== MODELO 1: Tipo de Movimiento ===")
print(classification_report(y1_test, y1_pred))
print("Matriz de Confusión:")
print(confusion_matrix(y1_test, y1_pred))

# Guardar modelo y escalador
joblib.dump(clf1, 'model_uno.pkl')
joblib.dump(scaler1, 'scale_uno.pkl')

# === ENTRENAR MODELO 2 ===
X2 = df_nivel[features]
y2 = df_nivel['label']
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, stratify=y2, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2.fit(X2_train, y2_train)
y2_pred = clf2.predict(X2_test)

print("\n=== MODELO 2: Nivel de Temblor ===")
print(classification_report(y2_test, y2_pred))
print("Matriz de Confusión:")
print(confusion_matrix(y2_test, y2_pred))

# Guardar modelo y escalador
joblib.dump(clf2, 'model_dos.pkl')
joblib.dump(scaler2, 'scale_dos.pkl')
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === RUTAS DE DATOS ===
ruta_actividades = 'movement/'
ruta_niveles = 'uploads/'

# === CARACTERÍSTICAS A USAR ===
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# === FUNCIÓN PARA CARGAR MOVIMIENTOS VOLUNTARIOS/INVOLUNTARIOS ===
def cargar_datos_voluntad(ruta):
    archivos = glob.glob(os.path.join(ruta, '*.csv'))
    normal, parkinson = [], []

    print(f"\n📂 Archivos encontrados en {ruta}:")
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        print(f"→ {nombre}")

        try:
            df = pd.read_csv(archivo)
            if not set(features).issubset(df.columns):
                print(f"[!] Ignorado (faltan columnas): {nombre}")
                continue

            if '_P' in nombre or 'P_' in nombre or nombre.split('_')[0].endswith('P'):
                df['label'] = 'involuntario'
                parkinson.append(df)
                print(f"✔️ Involuntario: {nombre}")
            else:
                df['label'] = 'voluntario'
                normal.append(df)
                print(f"✔️ Voluntario: {nombre}")

        except Exception as e:
            print(f"[!] Error leyendo {nombre}: {e}")

    df_voluntario = pd.concat(normal, ignore_index=True) if normal else pd.DataFrame()
    df_involuntario = pd.concat(parkinson, ignore_index=True) if parkinson else pd.DataFrame()
    return df_voluntario, df_involuntario

# === CARGA Y ENTRENAMIENTO MODELO 1 ===
df_voluntario, df_involuntario = cargar_datos_voluntad(ruta_actividades)
df_movimiento = pd.concat([df_voluntario, df_involuntario], ignore_index=True)

X_vol = df_movimiento[features]
y_vol = df_movimiento['label']

scaler_vol = StandardScaler()
X_vol_scaled = scaler_vol.fit_transform(X_vol)

X_train_vol, X_test_vol, y_train_vol, y_test_vol = train_test_split(X_vol_scaled, y_vol, test_size=0.2, stratify=y_vol, random_state=42)

clf_vol = RandomForestClassifier(n_estimators=100, random_state=42)
clf_vol.fit(X_train_vol, y_train_vol)

print("\n=== MODELO 1: Voluntario/Involuntario ===")
print(classification_report(y_test_vol, clf_vol.predict(X_test_vol)))
print(confusion_matrix(y_test_vol, clf_vol.predict(X_test_vol)))

joblib.dump(clf_vol, 'modelo_por_voluntad.pkl')
joblib.dump(scaler_vol, 'escalador_voluntad.pkl')


# === FUNCIÓN PARA CARGAR NIVELES DE TEMBLOR ===
def cargar_datos_nivel(ruta, prefijo, etiqueta):
    archivos = glob.glob(os.path.join(ruta, f'{prefijo}_*.csv'))
    dataframes = []
    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
            if set(features).issubset(df.columns):
                df['label'] = etiqueta
                dataframes.append(df)
            else:
                print(f"[!] Ignorado (faltan columnas): {archivo}")
        except Exception as e:
            print(f"[!] Error leyendo {archivo}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# === CARGA Y ENTRENAMIENTO MODELO 2 ===
df_bajo = cargar_datos_nivel(ruta_niveles, 'bajo', 'bajo')
df_medio = cargar_datos_nivel(ruta_niveles, 'medio', 'medio')
df_brusco = cargar_datos_nivel(ruta_niveles, 'brusco', 'brusco')

df_nivel = pd.concat([df_bajo, df_medio, df_brusco], ignore_index=True)

X_nivel = df_nivel[features]
y_nivel = df_nivel['label']

scaler_nivel = StandardScaler()
X_scaled_nivel = scaler_nivel.fit_transform(X_nivel)

X_train_nivel, X_test_nivel, y_train_nivel, y_test_nivel = train_test_split(X_scaled_nivel, y_nivel, test_size=0.2, stratify=y_nivel, random_state=42)

clf_nivel = RandomForestClassifier(n_estimators=100, random_state=42)
clf_nivel.fit(X_train_nivel, y_train_nivel)

print("\n=== MODELO 2: Nivel de Temblor ===")
print(classification_report(y_test_nivel, clf_nivel.predict(X_test_nivel)))
print(confusion_matrix(y_test_nivel, clf_nivel.predict(X_test_nivel)))

joblib.dump(clf_nivel, 'modelo_por_nivel.pkl')
joblib.dump(scaler_nivel, 'escalador_nivel.pkl')