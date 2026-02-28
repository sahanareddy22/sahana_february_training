# Conclusion

## Missing Value Handling
Median worked best for numerical features because the dataset contains outliers.
Mode was effective for categorical features.

## Categorical Encoding
- One-Hot Encoding worked best for nominal variables like MSZoning.
- Ordinal Encoding worked well for ordered quality features.
- Target Encoding improved performance for high-cardinality features like Neighborhood.
- Label Encoding is suitable for tree-based models.

## Feature Scaling
Standardization performed best for normally distributed features.
Min-Max scaling works well when the algorithm is sensitive to feature range.

## Outliers
IQR method effectively removed extreme SalePrice values.

## Skewness
Log transformation significantly reduced right skewness of SalePrice.