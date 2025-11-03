from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder
import os
import gc

app = Flask(__name__)
plt.switch_backend('Agg')

def plot_to_base64():
    """Convertir plot a base64 para HTML"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=80)  # Reducir DPI
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    img.close()
    return plot_data

def load_dataset_safe():
    """Cargar dataset de forma segura y eficiente"""
    try:
        dataset_files = [f for f in os.listdir('.') if f.endswith(('.txt', '.arff', '.TXT')) and 'KDD' in f.upper()]
        
        if not dataset_files:
            return None
            
        dataset_path = dataset_files[0]
        print(f"📁 Cargando dataset: {dataset_path}")
        
        # Leer solo las primeras filas para desarrollo
        nrows = 1000  # Limitar filas
        
        if dataset_path.endswith('.arff'):
            from scipy.io import arff
            data, meta = arff.loadarff(dataset_path)
            df_orig = pd.DataFrame(data)
            # Convertir solo columnas necesarias
            for col in ['protocol_type', 'service', 'flag', 'class']:
                if col in df_orig.columns and df_orig[col].dtype == object:
                    df_orig[col] = df_orig[col].str.decode('utf-8')
        else:
            # Leer con tipos de datos optimizados
            df_orig = pd.read_csv(dataset_path, header=None, nrows=nrows)
            
        return df_orig
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None

def create_sample_data():
    """Crear datos de ejemplo más livianos"""
    np.random.seed(42)
    n_samples = 300  # Reducir muestras
    sample_data = {
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
        'src_bytes': np.random.randint(0, 5000, n_samples, dtype=np.int16),
        'dst_bytes': np.random.randint(0, 2500, n_samples, dtype=np.int16),
        'duration': np.random.randint(0, 50, n_samples, dtype=np.int8),
        'same_srv_rate': np.random.uniform(0, 1, n_samples).astype(np.float32),
        'dst_host_srv_count': np.random.randint(0, 50, n_samples, dtype=np.int8),
        'dst_host_same_srv_rate': np.random.uniform(0, 1, n_samples).astype(np.float32),
        'class': np.random.choice(['normal', 'anomaly'], n_samples, p=[0.7, 0.3])
    }
    return pd.DataFrame(sample_data)

def optimize_dataframe(df):
    """Optimizar tipos de datos para reducir memoria"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == object:
            # Convertir a categoría si tiene pocos valores únicos
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
                
        elif col_type in ['int64', 'int32']:
            # Optimizar enteros
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                else:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
                    
        elif col_type == 'float64':
            # Convertir a float32
            df[col] = df[col].astype(np.float32)
            
    return df

@app.route('/')
def index():
    plots = {}
    
    try:
        # 1. CARGA OPTIMIZADA DE DATOS
        df_orig = load_dataset_safe()
        if df_orig is None:
            df_orig = create_sample_data()
            dataset_source = 'Datos de ejemplo'
        else:
            dataset_source = 'Dataset NSL-KDD'
        
        # Limitar tamaño del dataset
        if len(df_orig) > 2000:
            df_orig = df_orig.sample(n=2000, random_state=42)
        
        # Optimizar dataframe
        df = optimize_dataframe(df_orig.copy())
        
        # Asignar nombres de columnas (solo las necesarias)
        essential_columns = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                           'duration', 'same_srv_rate', 'dst_host_srv_count', 
                           'dst_host_same_srv_rate', 'class']
        
        for i, col in enumerate(df.columns):
            if i < len(essential_columns):
                df = df.rename(columns={col: essential_columns[i]})
            else:
                break
        
        # 2. PREPROCESAMIENTO SELECTIVO
        labelencoder = LabelEncoder()
        for col in ['class', 'protocol_type', 'service', 'flag']:
            if col in df.columns:
                df[col] = labelencoder.fit_transform(df[col].astype(str))
        
        # 3. GENERAR VISUALIZACIONES UNA POR UNA (liberando memoria)
        
        # Visualización 1: Distribución de protocol_type
        plt.figure(figsize=(8, 5))
        if 'protocol_type' in df_orig.columns:
            protocol_counts = df_orig['protocol_type'].value_counts().head(5)
            plt.bar(protocol_counts.index, protocol_counts.values, 
                   color=['#3498db', '#9b59b6', '#e67e22'])
            plt.title('Distribución de Protocolos')
            plt.xticks(rotation=45)
        plots['protocol_hist'] = plot_to_base64()
        plt.close()
        gc.collect()
        
        # Visualización 2: Solo 2 histogramas principales
        plt.figure(figsize=(10, 4))
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col in ['src_bytes', 'dst_bytes', 'duration', 'same_srv_rate']]
        
        for i, col in enumerate(numeric_cols[:2]):
            plt.subplot(1, 2, i + 1)
            df[col].hist(bins=20, alpha=0.7, color='#3498db', edgecolor='black')
            plt.title(f'Distribución de {col}')
            plt.tight_layout()
        plots['multiple_hist'] = plot_to_base64()
        plt.close()
        gc.collect()
        
        # Visualización 3: Heatmap simplificado
        plt.figure(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            important_cols = [col for col in numeric_df.columns if col in 
                            ['same_srv_rate', 'dst_host_same_srv_rate', 'class']]
            if len(important_cols) < 2:
                important_cols = numeric_df.columns[:4]
            
            correlation_matrix = numeric_df[important_cols].corr()
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       fmt='.2f')
            plt.title('Correlaciones')
        plots['correlation_heatmap'] = plot_to_base64()
        plt.close()
        gc.collect()
        
        # Visualización 4: Distribución de clases
        plt.figure(figsize=(6, 4))
        if 'class' in df_orig.columns:
            class_distribution = df_orig['class'].value_counts().head(3)
            colors = ['#2ecc71' if 'normal' in str(x).lower() else '#e74c3c' 
                     for x in class_distribution.index]
            plt.bar(range(len(class_distribution)), class_distribution.values, color=colors)
            plt.title('Distribución de Clases')
            plt.xticks(range(len(class_distribution)), 
                      [str(x)[:15] for x in class_distribution.index], 
                      rotation=45)
        plots['class_distribution'] = plot_to_base64()
        plt.close()
        gc.collect()
        
        # 4. DATOS PARA TEMPLATE (solo esenciales)
        table_head = df.head(8).to_dict('records')  # Menos filas
        
        dtype_info = []
        for col in df.columns[:6]:  # Solo primeras 6 columnas
            dtype_info.append({
                'columna': col,
                'tipo': str(df[col].dtype),
                'no_nulos': df[col].notnull().sum(),
                'unicos': df[col].nunique()
            })
        
        stats_data = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            'dataset_source': dataset_source
        }
        
        # Liberar memoria
        del df_orig, numeric_df
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
        # Datos de fallback mínimos
        plots = {f'plot_{i}': '' for i in range(4)}
        table_head = []
        dtype_info = []
        stats_data = {'total_rows': 0, 'total_columns': 0, 'dataset_source': 'Error'}
    
    return render_template('index.html', 
                         plots=plots,
                         stats=stats_data,
                         table_head=table_head,
                         dtype_info=dtype_info,
                         columns=list(plots.keys()))

@app.route('/health')
def health_check():
    """Endpoint simple para verificar que la app está funcionando"""
    return "OK"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Iniciando NSL-KDD Dashboard en puerto {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
