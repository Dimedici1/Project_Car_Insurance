import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer

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

# Drop rows with nan values and policy_id
df.dropna(inplace=True)
df = df.drop('policy_id', axis=1)

# Adapt two columns
pt=PowerTransformer(method='yeo-johnson')
df.loc[:,['age_of_car','age_of_policyholder']] = pt.fit_transform(pd.DataFrame(df.loc[:,['age_of_car','age_of_policyholder']]))

# Reset index
df = df.reset_index(drop=True)

# Save the cleaned dataset
df.to_csv('ml_gw_car_insurance_cleaned.csv', index=False)
