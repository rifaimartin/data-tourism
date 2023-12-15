import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Membaca dataset
users_df = pd.read_csv('/content/user.csv')
tourism_rating_df = pd.read_csv('/content/tourism_rating.csv')

merged_df = users_df.merge(tourism_rating_df, on='User_Id', how='inner')

tourism_with_id_df = pd.read_csv('/content/tourism_with_id.csv')
final_merged_df = merged_df.merge(tourism_with_id_df, on='Place_Id', how='inner')
final_merged_df.fillna(value=0, inplace=True)

# Preprocessing data - konversi variabel kategorikal menjadi numerik
label_encoders = {}
categorical_columns = ['Location', 'Place_Name', 'Description', 'Category', 'City']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    final_merged_df[col] = label_encoders[col].fit_transform(final_merged_df[col])

# Pemilihan fitur
features = ['User_Id', 'Location',  'Place_Id', 'Time_Minutes']  # Sesuaikan dengan fitur yang ingin digunakan

target = 'Place_Ratings'

# Pembagian data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(final_merged_df[features], final_merged_df[target], test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi model
test_score = model.score(X_test, y_test)
print(f"Test Score: {test_score}")