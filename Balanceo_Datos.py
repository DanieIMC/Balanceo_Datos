import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Cargar el dataset
file_path = r"C:\Users\s2dan\OneDrive\Documentos\WorkSpace\Proyect_AI\ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(file_path)

# Separar las características numéricas y categóricas
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Escalar las variables numéricas
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Codificar las variables categóricas
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Verificar el balanceo de las clases
class_distribution = data['NObeyesdad'].value_counts(normalize=True) * 100
print("Distribución de clases antes del balanceo:")
print(class_distribution)

# Si fuera necesario balancear los datos, aplicamos SMOTE (oversampling)
X = data.drop('NObeyesdad', axis=1)  # Características
y = data['NObeyesdad']  # Etiqueta

# Aplicamos SMOTE si es necesario
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verificar la distribución de clases después del balanceo
resampled_class_distribution = pd.Series(y_resampled).value_counts(normalize=True) * 100
print("\nDistribución de clases después del balanceo:")
print(resampled_class_distribution)
