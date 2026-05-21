
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


products_df = pd.read_csv(r'C:/blinkit datasets/blinkit_products.csv')
orders_df   = pd.read_csv(r'C:/blinkit datasets/blinkit_orders.csv')
marketing_df = pd.read_csv(r'C:/blinkit datasets/blinkit_marketing_performance.csv')

# 1. Explore data structure
print("=== PRODUCTS ===")
print(products_df.info())
print(products_df.head())
print(products_df.describe(include='all'))

print("\n=== ORDERS ===")
print(orders_df.info())
print(orders_df.head())
print(orders_df.describe(include='all'))

print("\n=== MARKETING ===")
print(marketing_df.info())
print(marketing_df.head())
print(marketing_df.describe(include='all'))


# 2. Identify trends, patterns, anomalies 

orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
marketing_df['date']    = pd.to_datetime(marketing_df['date'])

# a) Histograms for distributions
for col in ['price','margin_percentage','shelf_life_days']:
    products_df[col].hist(bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

orders_df['order_total'].hist(bins=30)
plt.title('Order Total Distribution')
plt.xlabel('Order Total')
plt.ylabel('Count')
plt.show()

# b) Correlation heatmap (numeric variables)
num_df = pd.concat([
    products_df.select_dtypes(include=['number']),
    orders_df[['order_total','delivery_delay_mins']] if 'delivery_delay_mins' in orders_df else pd.DataFrame(),
    marketing_df.select_dtypes(include=['number'])
], axis=1).dropna(axis=1, how='all')
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# c) Time series trends
monthly_orders = orders_df.resample('M', on='order_date').size()
monthly_orders.plot(marker='o')
plt.title('Monthly Order Volume')
plt.ylabel('Order Count')
plt.show()

monthly_spend = marketing_df.resample('M', on='date')['spend'].sum()
monthly_spend.plot(marker='o', color='orange')
plt.title('Monthly Marketing Spend')
plt.ylabel('Total Spend')
plt.show()

# d) Boxplots to detect outliers
sns.boxplot(data=products_df, x='category', y='price')
plt.title('Price by Category (Outliers)')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=orders_df, y='order_total')
plt.title('Order Total Boxplot')
plt.show()

# e) Categorical patterns
for col in ['category','brand']:
    print(f"\nTop 5 {col} by count:")
    print(products_df[col].value_counts().head())
    
print("\nTop 5 payment methods:")
print(orders_df['payment_method'].value_counts().head())


#Testing Hypotheses & Validating Assumptions
from scipy.stats import ttest_ind, pearsonr

#Hypothesis 1: Orders with card payments have higher average order total than those with cash
card_orders = orders_df[orders_df['payment_method'] == 'Card']['order_total']
cash_orders = orders_df[orders_df['payment_method'] == 'Cash']['order_total']
t_stat, p_val = ttest_ind(card_orders, cash_orders)

# Hypothesis 2: Higher spend leads to higher revenue in marketing campaigns
corr_spend_revenue, corr_p_val = pearsonr(marketing_df['spend'], marketing_df['revenue_generated'])

# Detecting missing values
missing_values = {
    'products': products_df.isnull().sum(),
    'orders': orders_df.isnull().sum(),
    'marketing': marketing_df.isnull().sum()
}

# Detecting duplicate records
duplicate_counts = {
    'products': products_df.duplicated().sum(),
    'orders': orders_df.duplicated().sum(),
    'marketing': marketing_df.duplicated().sum()
}

# Detecting outliers using IQR for order_total
Q1 = orders_df['order_total'].quantile(0.25)
Q3 = orders_df['order_total'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (orders_df['order_total'] < (Q1 - 1.5 * IQR)) | (orders_df['order_total'] > (Q3 + 1.5 * IQR))
outlier_count = outlier_mask.sum()

# Output all results
print("\n Hypothesis 1: Card vs Cash Avg Order Total")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print("Significant difference?" , "Yes" if p_val < 0.05 else "No")

print("\ Hypothesis 2: Spend vs Revenue Correlation")
print(f"Correlation: {corr_spend_revenue:.4f}")
print(f"P-value: {corr_p_val:.4f}")
print("Significant correlation?", "Yes" if corr_p_val < 0.05 else "No")

print("\n Missing Values:")
for df_name, missing in missing_values.items():
    print(f"\n{df_name.capitalize()}:\n{missing}")

print("\n Duplicate Records:")
for df_name, count in duplicate_counts.items():
    print(f"{df_name.capitalize()}: {count} duplicates")

    print(f"\n Outliers in 'order_total': {outlier_count}")




