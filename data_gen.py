import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

platforms = ['Blinkit', 'Zepto', 'Swiggy']
brands = ['Surf_Excel', 'Dove', 'Knorr', 'Cadbury', 'Oreo', 'Sunfeast', 'Bingo', 'Ariel']
campaign_types = ['Sponsored_Products', 'Sponsored_Display', 'Sponsored_Brands']

df = pd.DataFrame({
    'platform': np.random.choice(platforms, n),
    'brand': np.random.choice(brands, n),
    'campaign_type': np.random.choice(campaign_types, n),
    'spend': np.random.uniform(500, 50000, n).round(2),
    'impressions': np.random.randint(1000, 500000, n),
    'clicks': np.random.randint(50, 10000, n),
    'orders': np.random.randint(5, 1000, n),
    'revenue': np.random.uniform(1000, 200000, n).round(2),
    'day_of_week': np.random.randint(0, 7, n),
    'is_weekend': np.random.randint(0, 2, n),
})

df['CTR'] = (df['clicks'] / df['impressions']).round(4)
df['ROAS'] = (df['revenue'] / df['spend']).round(2)
df['CPC'] = (df['spend'] / df['clicks']).round(2)

df['is_risky'] = (df['ROAS'] < 3).astype(int)

df.to_csv('campaign_data.csv', index=False)
print(f"Data generated! Shape: {df.shape}")
print(df['is_risky'].value_counts())