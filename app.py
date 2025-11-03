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

app = Flask(__name__)
plt.switch_backend('Agg')

def plot_to_base64():
    """Convertir plot a base64 para HTML"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def create_sample_data():
    """Crear datos de ejemplo realistas del NSL-KDD"""
    print("📊 Generando datos de ejemplo NSL-KDD...")
    np.random.seed(42)
    n_samples = 1500
    
    sample_data = {
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'dns', 'ssh', 'telnet'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'SH'], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'src_bytes': np.random.exponential(1000, n_samples).astype(int),
        'dst_bytes': np.random.exponential(500, n_samples).astype(int),
        'duration': np.random.exponential(10, n_samples).astype(int),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'same_srv_rate': np.random.beta(2, 5, n_samples),
        'diff_srv_rate': np.random.beta(5, 2, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(8, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(2, 5, n_samples),
        'class': np.random.choice(['normal', 'neptune', 'portsweep', 'satan', 'ipsweep', 'smurf'], n_samples, 
                                p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
    }
    
    df = pd.DataFrame(sample_data)
    return df

@app.route('/')
def index():
    try:
        # Usar datos de ejemplo (más confiable que cargar archivos externos)
        print("🎯 Inicializando dashboard NSL-KDD...")
        df_orig = create_sample_data()
        df = df_orig.copy()
        
        # Preprocesamiento
        labelencoder = LabelEncoder()
        categorical_columns = []
        
        for col in ['class', 'protocol_type', 'service', 'flag']:
            if col in df.columns:
                df[col] = labelencoder.fit_transform(df[col].astype(str))
                categorical_columns.append(col)
        
        # Generar visualizaciones
        plots = {}
        
        # 1. Distribución de clases
        plt.figure(figsize=(12, 6))
        class_distribution = df_orig['class'].value_counts()
        colors = ['#2ecc71' if 'normal' in str(x).lower() else '#e74c3c' for x in class_distribution.index]
        
        bars = plt.bar(range(len(class_distribution)), class_distribution.values, color=colors, alpha=0.8)
        plt.title('Distribución de Clases - NSL-KDD', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tipo de Conexión', fontweight='bold')
        plt.ylabel('Cantidad de Registros', fontweight='bold')
        plt.xticks(range(len(class_distribution)), 
                  [str(x) for x in class_distribution.index], 
                  rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, v) in enumerate(zip(bars, class_distribution.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_distribution.values)*0.01, 
                    f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plots['class_distribution'] = plot_to_base64()
        plt.close()
        
        # 2. Distribución de protocolos
        plt.figure(figsize=(10, 6))
        protocol_counts = df_orig['protocol_type'].value_counts()
        colors = ['#3498db', '#9b59b6', '#e67e22']
        bars = plt.bar(protocol_counts.index, protocol_counts.values, color=colors, alpha=0.8)
        plt.title('Distribución de Protocolos de Red', fontsize=14, fontweight='bold')
        plt.xlabel('Protocolo', fontweight='bold')
        plt.ylabel('Frecuencia', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, v in zip(bars, protocol_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(protocol_counts.values)*0.01, 
                    f'{v}', ha='center', va='bottom', fontweight='bold')
        
        plots['protocol_hist'] = plot_to_base64()
        plt.close()
        
        # 3. Heatmap de correlaciones
        plt.figure(figsize=(12, 10))
        numeric_cols = ['src_bytes', 'dst_bytes', 'duration', 'count', 'srv_count', 
                       'dst_host_count', 'dst_host_srv_count']
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
                       annot_kws={"size": 10})
            plt.title('Mapa de Calor - Correlaciones entre Variables', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plots['correlation_heatmap'] = plot_to_base64()
        plt.close()
        
        # 4. Distribución de bytes
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if 'src_bytes' in df.columns:
            plt.hist(df['src_bytes'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            plt.title('Bytes de Origen', fontweight='bold')
            plt.xlabel('Bytes')
            plt.ylabel('Frecuencia')
            plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if 'dst_bytes' in df.columns:
            plt.hist(df['dst_bytes'], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
            plt.title('Bytes de Destino', fontweight='bold')
            plt.xlabel('Bytes')
            plt.ylabel('Frecuencia')
            plt.grid(alpha=0.3)
        
        plt.suptitle('Distribución de Tráfico de Red', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots['bytes_hist'] = plot_to_base64()
        plt.close()
        
        # 5. Servicios más comunes
        plt.figure(figsize=(12, 6))
        if 'service' in df_orig.columns:
            service_counts = df_orig['service'].value_counts().head(8)
            colors = plt.cm.Set3(np.linspace(0, 1, len(service_counts)))
            bars = plt.bar(service_counts.index, service_counts.values, color=colors, alpha=0.8)
            plt.title('Top 8 Servicios Más Utilizados', fontsize=14, fontweight='bold')
            plt.xlabel('Servicio', fontweight='bold')
            plt.ylabel('Frecuencia', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            for bar, v in zip(bars, service_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(service_counts.values)*0.01, 
                        f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plots['services'] = plot_to_base64()
        plt.close()
        
        # Preparar datos para la plantilla
        stats_data = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(categorical_columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'dataset_source': 'NSL-KDD Dataset Simulation',
            'null_values': df.isnull().sum().sum(),
            'attack_percentage': f"{(len(df_orig[df_orig['class'] != 'normal']) / len(df_orig) * 100):.1f}%"
        }
        
        # Datos para tabla
        table_head = df_orig.head(12).to_dict('records')
        columns = df_orig.columns.tolist()
        
        # Información de tipos de datos
        dtype_info = []
        for col in df_orig.columns[:10]:  # Solo primeras 10 columnas
            dtype_info.append({
                'columna': col,
                'tipo': str(df_orig[col].dtype),
                'no_nulos': df_orig[col].notnull().sum(),
                'nulos': df_orig[col].isnull().sum(),
                'unicos': df_orig[col].nunique()
            })
        
        return render_template('index.html', 
                             plots=plots,
                             stats=stats_data,
                             table_head=table_head,
                             dtype_info=dtype_info,
                             columns=columns)
                             
    except Exception as e:
        print(f"❌ Error en la aplicación: {e}")
        import traceback
        traceback.print_exc()
        return f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 2rem; background: #f8f9fa;">
                <div style="max-width: 800px; margin: 0 auto; background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h1 style="color: #e74c3c;">🚨 Error en la aplicación</h1>
                    <p><strong>Detalles:</strong> {str(e)}</p>
                    <p>Por favor, verifica que todos los archivos estén correctamente configurados.</p>
                </div>
            </body>
        </html>
        """

@app.route('/health')
def health():
    """Endpoint de salud"""
    return {'status': 'healthy', 'message': 'NSL-KDD Dashboard running'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Iniciando NSL-KDD Dashboard en puerto {port}...")
    print("📊 Usando datos de ejemplo del NSL-KDD")
    print(f"🌐 Accede en: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
