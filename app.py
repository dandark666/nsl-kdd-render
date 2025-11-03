from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import os
import traceback

app = Flask(__name__)
plt.switch_backend('Agg')

def plot_to_base64():
    """Convertir plot a base64 para HTML"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def load_dataset():
    """Cargar dataset NSL-KDD con manejo robusto de errores"""
    try:
        # Buscar archivos del dataset
        dataset_files = [f for f in os.listdir('.') if f.endswith(('.txt', '.csv', '.arff', '.TXT')) and 'KDD' in f.upper()]
        
        if dataset_files:
            dataset_path = dataset_files[0]
            print(f"📁 Cargando dataset: {dataset_path}")
            
            if dataset_path.endswith('.arff'):
                from scipy.io import arff
                data, meta = arff.loadarff(dataset_path)
                df_orig = pd.DataFrame(data)
                # Decodificar bytes a strings
                for col in df_orig.columns:
                    if df_orig[col].dtype == object:
                        df_orig[col] = df_orig[col].str.decode('utf-8')
            else:
                # Intentar diferentes separadores
                try:
                    df_orig = pd.read_csv(dataset_path, header=None)
                except:
                    df_orig = pd.read_csv(dataset_path, header=None, sep='\t')
            
            return df_orig, dataset_path
            
        else:
            raise FileNotFoundError("No se encontró archivo del dataset NSL-KDD")
            
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None, None

def create_enhanced_sample_data():
    """Crear datos de ejemplo realistas y mejorados del NSL-KDD"""
    print("📊 Generando datos de ejemplo NSL-KDD mejorados...")
    np.random.seed(42)
    n_samples = 2000
    
    sample_data = {
        'duration': np.random.exponential(10, n_samples).astype(int),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'dns', 'ssh', 'telnet', 'pop_3', 'other'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'SH', 'RSTR'], n_samples, p=[0.6, 0.15, 0.1, 0.08, 0.05, 0.02]),
        'src_bytes': np.random.exponential(1000, n_samples).astype(int),
        'dst_bytes': np.random.exponential(500, n_samples).astype(int),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.poisson(0.1, n_samples),
        'urgent': np.random.poisson(0.01, n_samples),
        'hot': np.random.poisson(0.5, n_samples),
        'num_failed_logins': np.random.poisson(0.1, n_samples),
        'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'num_compromised': np.random.poisson(0.05, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'num_root': np.random.poisson(0.1, n_samples),
        'num_file_creations': np.random.poisson(0.05, n_samples),
        'num_shells': np.random.poisson(0.02, n_samples),
        'num_access_files': np.random.poisson(0.03, n_samples),
        'num_outbound_cmds': np.random.poisson(0.001, n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'serror_rate': np.random.beta(2, 5, n_samples),
        'srv_serror_rate': np.random.beta(2, 5, n_samples),
        'rerror_rate': np.random.beta(5, 2, n_samples),
        'srv_rerror_rate': np.random.beta(5, 2, n_samples),
        'same_srv_rate': np.random.beta(2, 5, n_samples),
        'diff_srv_rate': np.random.beta(5, 2, n_samples),
        'srv_diff_host_rate': np.random.beta(5, 2, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(8, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(2, 5, n_samples),
        'dst_host_same_src_port_rate': np.random.beta(5, 2, n_samples),
        'dst_host_srv_diff_host_rate': np.random.beta(2, 5, n_samples),
        'dst_host_serror_rate': np.random.beta(2, 5, n_samples),
        'dst_host_srv_serror_rate': np.random.beta(2, 5, n_samples),
        'dst_host_rerror_rate': np.random.beta(5, 2, n_samples),
        'dst_host_srv_rerror_rate': np.random.beta(5, 2, n_samples),
        'class': np.random.choice(['normal', 'neptune', 'portsweep', 'satan', 'ipsweep', 'smurf', 'back', 'teardrop'], 
                                n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]),
        'difficulty': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(sample_data)
    return df

def assign_column_names(df):
    """Asignar nombres de columnas estándar del NSL-KDD"""
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]
    
    # Asignar nombres disponibles
    for i, col in enumerate(df.columns):
        if i < len(column_names):
            df = df.rename(columns={col: column_names[i]})
    
    return df

@app.route('/')
def index():
    try:
        # 1. CARGAR DATOS
        df_orig, dataset_path = load_dataset()
        
        if df_orig is None:
            print("🔄 Usando datos de ejemplo...")
            df_orig = create_enhanced_sample_data()
            dataset_source = 'NSL-KDD Dataset Simulation'
        else:
            df_orig = assign_column_names(df_orig)
            dataset_source = dataset_path
        
        df = df_orig.copy()
        
        # 2. PREPROCESAMIENTO
        labelencoder = LabelEncoder()
        categorical_columns = []
        
        for col in ['class', 'protocol_type', 'service', 'flag']:
            if col in df.columns:
                df[col] = labelencoder.fit_transform(df[col].astype(str))
                categorical_columns.append(col)
        
        # 3. GENERAR VISUALIZACIONES MEJORADAS
        plots = {}
        
        # 3.1 Distribución de clases (MEJORADA)
        plt.figure(figsize=(14, 8))
        if 'class' in df_orig.columns:
            class_distribution = df_orig['class'].value_counts()
            colors = ['#2ecc71' if 'normal' in str(x).lower() else '#e74c3c' for x in class_distribution.index]
            
            bars = plt.bar(range(len(class_distribution)), class_distribution.values, 
                         color=colors, alpha=0.8, edgecolor='black')
            plt.title('Distribución de Clases - NSL-KDD', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Tipo de Conexión', fontsize=12, fontweight='bold')
            plt.ylabel('Cantidad de Registros', fontsize=12, fontweight='bold')
            plt.xticks(range(len(class_distribution)), 
                      [str(x) for x in class_distribution.index], 
                      rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            
            # Añadir valores y porcentajes
            total = len(df_orig)
            for i, (bar, v) in enumerate(zip(bars, class_distribution.values)):
                percentage = (v / total) * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_distribution.values)*0.01, 
                        f'{v}\n({percentage:.1f}%)', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
        
        plots['class_distribution'] = plot_to_base64()
        plt.close()
        
        # 3.2 Distribución de protocolos (MEJORADA)
        plt.figure(figsize=(12, 7))
        if 'protocol_type' in df_orig.columns:
            protocol_counts = df_orig['protocol_type'].value_counts()
            colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']
            bars = plt.bar(protocol_counts.index, protocol_counts.values, 
                         color=colors[:len(protocol_counts)], alpha=0.8, edgecolor='black')
            plt.title('Distribución de Protocolos de Red', fontsize=16, fontweight='bold')
            plt.xlabel('Protocolo', fontsize=12, fontweight='bold')
            plt.ylabel('Frecuencia', fontsize=12, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            for bar, v in zip(bars, protocol_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(protocol_counts.values)*0.01, 
                        f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plots['protocol_hist'] = plot_to_base64()
        plt.close()
        
        # 3.3 Heatmap de correlaciones (MEJORADA)
        plt.figure(figsize=(16, 12))
        numeric_cols = ['src_bytes', 'dst_bytes', 'duration', 'count', 'srv_count', 
                       'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'diff_srv_rate']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if len(available_numeric) >= 3:
            correlation_matrix = df[available_numeric].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True, 
                       fmt='.2f',
                       cbar_kws={"shrink": .8},
                       annot_kws={"size": 10, 'weight': 'bold'})
            plt.title('Mapa de Calor - Correlaciones entre Variables', 
                     fontsize=18, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
        
        plots['correlation_heatmap'] = plot_to_base64()
        plt.close()
        
        # 3.4 Distribución de bytes (MEJORADA)
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        if 'src_bytes' in df.columns:
            plt.hist(df['src_bytes'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
            plt.title('Distribución de Bytes de Origen', fontweight='bold', fontsize=12)
            plt.xlabel('Bytes', fontweight='bold')
            plt.ylabel('Frecuencia', fontweight='bold')
            plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if 'dst_bytes' in df.columns:
            plt.hist(df['dst_bytes'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
            plt.title('Distribución de Bytes de Destino', fontweight='bold', fontsize=12)
            plt.xlabel('Bytes', fontweight='bold')
            plt.ylabel('Frecuencia', fontweight='bold')
            plt.grid(alpha=0.3)
        
        plt.suptitle('Análisis de Tráfico de Red - Bytes Transmitidos', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plots['bytes_hist'] = plot_to_base64()
        plt.close()
        
        # 3.5 Servicios más comunes (MEJORADA)
        plt.figure(figsize=(14, 8))
        if 'service' in df_orig.columns:
            service_counts = df_orig['service'].value_counts().head(10)
            colors = plt.cm.Set3(np.linspace(0, 1, len(service_counts)))
            bars = plt.bar(service_counts.index, service_counts.values, 
                         color=colors, alpha=0.8, edgecolor='black')
            plt.title('Top 10 Servicios Más Utilizados', fontsize=16, fontweight='bold')
            plt.xlabel('Servicio', fontsize=12, fontweight='bold')
            plt.ylabel('Frecuencia', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            
            for bar, v in zip(bars, service_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(service_counts.values)*0.01, 
                        f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plots['services'] = plot_to_base64()
        plt.close()
        
        # 3.6 Histogramas múltiples (NUEVA)
        plt.figure(figsize=(16, 12))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Seleccionar columnas más importantes
            important_cols = [col for col in numeric_cols if any(keyword in col for keyword in 
                                ['bytes', 'count', 'rate', 'duration'])]
            if len(important_cols) > 6:
                important_cols = important_cols[:6]
            
            n_cols = 3
            n_rows = (len(important_cols) + n_cols - 1) // n_cols
            
            for i, col in enumerate(important_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.hist(df[col], bins=30, alpha=0.7, color='#27ae60', edgecolor='black')
                plt.title(f'Distribución de {col}', fontweight='bold', fontsize=11)
                plt.xlabel('Valor', fontsize=9)
                plt.ylabel('Frecuencia', fontsize=9)
                plt.grid(alpha=0.3)
        
        plt.suptitle('Distribuciones de Variables Numéricas Principales', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plots['multiple_hist'] = plot_to_base64()
        plt.close()
        
        # 4. PREPARAR DATOS PARA LA PLANTILLA
        stats_data = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(categorical_columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'dataset_source': dataset_source,
            'null_values': df.isnull().sum().sum(),
            'attack_percentage': f"{(len(df_orig[df_orig['class'] != 'normal']) / len(df_orig) * 100):.1f}%" if 'class' in df_orig.columns else 'N/A'
        }
        
        # Datos para tabla
        table_head = df_orig.head(15).to_dict('records')
        columns = df_orig.columns.tolist()
        
        # Información de tipos de datos
        dtype_info = []
        for col in df_orig.columns[:12]:  # Solo primeras 12 columnas para mejor visualización
            dtype_info.append({
                'columna': col,
                'tipo': str(df_orig[col].dtype),
                'no_nulos': df_orig[col].notnull().sum(),
                'nulos': df_orig[col].isnull().sum(),
                'unicos': df_orig[col].nunique()
            })
        
        # Estadísticas numéricas
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            numeric_stats = df.describe().round(3).to_dict()
        else:
            numeric_stats = {}
        
        return render_template('index.html', 
                             plots=plots,
                             stats=stats_data,
                             table_head=table_head,
                             dtype_info=dtype_info,
                             numeric_stats=numeric_stats,
                             columns=columns)
                             
    except Exception as e:
        print(f"❌ Error crítico en la aplicación: {e}")
        traceback.print_exc()
        return f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 2rem; background: #f8f9fa;">
                <div style="max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
                    <h1 style="color: #e74c3c;">🚨 Error en el Dashboard NSL-KDD</h1>
                    <p style="color: #7f8c8d;"><strong>Detalles:</strong> {str(e)}</p>
                    <p style="color: #95a5a6;">El sistema está generando datos de ejemplo. Por favor, verifica la configuración.</p>
                    <a href="/" style="display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 1rem;">
                        Reintentar
                    </a>
                </div>
            </body>
        </html>
        """

@app.route('/health')
def health():
    """Endpoint de salud para monitoreo"""
    return {
        'status': 'healthy', 
        'message': 'NSL-KDD Dashboard running',
        'timestamp': pd.Timestamp.now().isoformat()
    }

@app.route('/api/stats')
def api_stats():
    """Endpoint API para estadísticas"""
    try:
        df_orig, _ = load_dataset()
        if df_orig is None:
            df_orig = create_enhanced_sample_data()
        
        basic_stats = {
            'total_records': len(df_orig),
            'total_features': len(df_orig.columns),
            'memory_usage_mb': round(df_orig.memory_usage(deep=True).sum() / 1024**2, 2),
            'null_values': int(df_orig.isnull().sum().sum())
        }
        
        if 'class' in df_orig.columns:
            class_counts = df_orig['class'].value_counts().to_dict()
            basic_stats['class_distribution'] = class_counts
        
        return basic_stats
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Iniciando NSL-KDD Dashboard en puerto {port}...")
    print("📊 Sistema optimizado para producción")
    print(f"🌐 Accede en: http://localhost:{port}")
    print("🔧 Endpoints disponibles: /, /health, /api/stats")
    app.run(host='0.0.0.0', port=port, debug=False)
