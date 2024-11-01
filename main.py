import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi okuma
df = pd.read_csv('train.csv')

# 1. PANDAS İLE VERİ ANALİZİ
print("1. Verinin ilk 5 satırı:")
print(df.head())

print("\n2. Veri hakkında genel bilgi:")
print(df.info())

print("\n3. Hayatta kalanların sayısı:")
print(df['Survived'].value_counts())

print("\n4. Yolcu sınıflarına göre dağılım:")
print(df['Pclass'].value_counts())

# 5. Eksik verileri kontrol edelim
print("\n5. Eksik veriler:")
print(df.isnull().sum())

# 2. NUMPY İLE HESAPLAMALAR
print("\n6. Yaş istatistikleri:")
print("Ortalama yaş:", np.mean(df['Age'].dropna()))
print("Medyan yaş:", np.median(df['Age'].dropna()))
print("Standart sapma:", np.std(df['Age'].dropna()))

# 3. MATPLOTLIB İLE GÖRSELLEŞTİRME
plt.figure(figsize=(10, 6))
df['Survived'].value_counts().plot(kind='bar')
plt.title('Hayatta Kalan vs Kalmayan Yolcu Sayısı')
plt.xlabel('Hayatta Kalma (0: Hayır, 1: Evet)')
plt.ylabel('Yolcu Sayısı')
plt.show()

# 4. SEABORN İLE GELİŞMİŞ GÖRSELLEŞTİRME
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df)
plt.title('Yolcu Sınıfı ve Yaşa Göre Hayatta Kalma Durumu')
plt.show()

# 5. İLGİNÇ ANALİZLER
# Cinsiyete göre hayatta kalma oranı
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Cinsiyete Göre Hayatta Kalma Oranı')
plt.show()

# Bilet sınıfına göre hayatta kalma oranı
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Bilet Sınıfına Göre Hayatta Kalma Oranı')
plt.show()

# Yaş dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=30)
plt.title('Yolcuların Yaş Dağılımı')
plt.show()

# Aile büyüklüğüne göre analiz
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Aile Büyüklüğüne Göre Hayatta Kalma Oranı')
plt.show()