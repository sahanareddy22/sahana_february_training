from preprocessing import *
from categorical_encoding import *
from scaling import *
df = load_data("data/train.csv")

df = handle_missing_values(df)
df = fix_datatypes(df)
df = remove_duplicates(df)
df = remove_irrelevant_columns(df)

df = treat_outliers_iqr(df, 'SalePrice')

df = transform_skewed_feature(df, 'SalePrice')

df = label_encoding(df, 'Neighborhood')

df = frequency_encoding(df, 'Neighborhood')
df = target_encoding(df, 'Neighborhood', 'SalePrice')

df = one_hot_encoding(df, ['MSZoning'])

quality_order = ['Po','Fa','TA','Gd','Ex']
df = ordinal_encoding(df, 'ExterQual', quality_order)

df = min_max_scaling(df, 'SalePrice')
df = standard_scaling(df, 'GrLivArea')

print("Preprocessing Completed Successfully!")
print(df.head())