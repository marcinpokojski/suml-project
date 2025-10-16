import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('lung_cancer_dataset.csv')

#usuniecie patient_id
if 'patient_id' in df.columns:
    df = df.drop(columns=['patient_id'])
    print("Usunięto kolumnę 'patient_id'")

#konwersja pack_years na int
if 'pack_years' in df.columns:

    df['pack_years'] = pd.to_numeric(df['pack_years'], errors='coerce')
    df['pack_years'] = df['pack_years'].fillna(0).astype(int)

#przedzialy wiekowe
    # bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, np.inf]
    # labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '50+']
    # df['pack_years_group'] = pd.cut(df['pack_years'], bins=bins, labels=labels, right=True)

#zamiana danych na binarne w przypadku yes/no dla kolumn asbestos_exposure, copd_diagnosis,family_history
binary_mappings = {
    'asbestos_exposure': {'yes': 1, 'no': 0},
    'copd_diagnosis': {'yes': 1, 'no': 0},
    'family_history': {'yes': 1, 'no': 0},
    'secondhand_smoke_exposure': {'yes': 1, 'no': 0},
    'lung_cancer': {'yes': 1, 'no': 0},

}
for col, mapping in binary_mappings.items():
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map(mapping)
        df[col] = df[col].fillna(0).astype(int)
        print(f"Zamieniono kolumnę '{col}' na wartości binarne 0/1")


#rzutowanie alcohol_consumption na 0,1,2 w zaleznosci od poziomu
df['alcohol_consumption'] = df['alcohol_consumption'].astype(str).str.strip().str.lower()

mapping = {
    'none': 0,
    'moderate': 1,
    'heavy': 2}

df['alcohol_consumption'] = df['alcohol_consumption'].map(mapping)

df['alcohol_consumption'] = df['alcohol_consumption'].fillna(0).astype(int)

#rzutowanie radon_exposure na 0,1,2 w zaleznosci od poziomu
df['radon_exposure'] = df['radon_exposure'].astype(str).str.strip().str.lower()

mapping = {
    'low': 0,
    'medium': 1,
    'high': 2}

df['radon_exposure'] = df['radon_exposure'].map(mapping)

df['radon_exposure'] = df['radon_exposure'].fillna(0).astype(int)

#analiza rozkladu cech

# Rozkład wieku pacjentów
sns.histplot(df['age'], kde=True)
plt.title('Rozkład wieku pacjentów')
plt.show()

# Rozkład liczby wypalonych paczek papierosów (pack_years)
sns.histplot(df['pack_years'], kde=True, bins=20)
plt.title('Rozkład pack_years')
plt.show()

# Proporcje płci wśród pacjentów
sns.countplot(x='gender', data=df)
plt.title('Rozkład płci')
plt.show()

# Proporcje diagnozy raka płuc (zmienna docelowa)
sns.countplot(x='lung_cancer', data=df)
plt.title('Rozkład występowania raka płuc')
plt.show()

# Zależność między wiekiem a rakiem płuc
sns.boxplot(x='lung_cancer', y='age', data=df)
plt.title('Zależność między wiekiem a rakiem płuc')
plt.xlabel('Lung Cancer (0 = brak, 1 = tak)')
plt.ylabel('Wiek')
plt.show()

#ew wyrownanie lung_cancer



print(min(df['age']))
pd.set_option('display.max_columns', None)  # pokaż wszystkie kolumny
pd.set_option('display.width', None)        # dopasuj szerokość do terminala
pd.set_option('display.max_colwidth', None) # nie skracaj długich wartości tekstowych


#podzial danych na treningowe i testowe (80/20)
X = df.drop('lung_cancer', axis=1)
y = df['lung_cancer']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # zachowuje proporcje klas 0/1
)

print(df.head(10))