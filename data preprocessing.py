print("Dataset Shape: ",df.shape)
df.info()

print(df.isnull.sum())

df=df.dropna()
print("Dataset Shape after removing missing values: ",df.shape)

print("Duplicate Rows: ",df.duplicated().sum())

df=df.drop_duplicates()

print("Dataset Shape after removing duplicates: ",df.shape)

print(df.dtypes)
df.describe()

print(df.nunique())
