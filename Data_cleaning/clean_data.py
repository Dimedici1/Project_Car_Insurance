import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('ml_gw_car_insurance.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
print(df.head())

# Get the object columns except for policy_id
object_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'policy_id']

# Perform label encoding on object columns
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Apply StandardScaler to numerical columns
numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'policy_id' and col != 'is_claim']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop rows with nan values
df.dropna(inplace=True)
print(df.head())

# Save the cleaned dataset
df.to_csv('ml_gw_car_insurance_cleaned.csv', index=False)
