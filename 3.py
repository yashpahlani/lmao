#Dataset:-https://drive.google.com/file/d/18GiWGBte1DslmaS8xix_up-YK6JpD5fl/view?usp=drive_link

import pandas as pd

data = pd.read_csv('/content/marketing_campaign_dataset.csv')


print(data.head())


print(data.describe())

data['Acquisition_Cost'] = data['Acquisition_Cost'].str.replace(',', '').str.replace('$', '', regex=False).astype(float)


missing_values = data.isnull().sum()
print("\nMissing values before handling:")
print(missing_values)

# Replace missing values in numeric columns with mean
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Replace missing values in categorical columns with mode
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Drop rows with missing values that cannot be replaced
data.dropna(subset=['Acquisition_Cost', 'Impressions'], inplace=True)

missing_values_after = data.isnull().sum()
print("\nMissing values after handling:")
print(missing_values_after)


unwanted_columns = ['Campaign_ID']  # List of unwanted columns
data.drop(columns=unwanted_columns, inplace=True)


initial_rows = data.shape[0]
data.drop_duplicates(inplace=True)
duplicate_rows = initial_rows - data.shape[0]
print("\nNumber of duplicate rows removed:", duplicate_rows)


print("Number of rows before removing NaN values:", len(data))

# Remove rows with NaN values
data = data.dropna(axis=0, how='any')

# Display the number of rows after removing NaN values
print("Number of rows after removing NaN values:", len(data))

data.to_csv('cleaned_marketing_data.csv', index=False)
