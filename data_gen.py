import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

platforms = ['Blinkit', 'Zepto', 'Swiggy']
brands = ['Surf_Excel', 'Dove', 'Knorr', 'Cadbury', 'Oreo', 'Sunfeast', 'Bingo', 'Ariel']
campaign_types = ['Sponsored_Products', 'Sponsored_Display', 'Sponsored_Brands']

# Step 1: Generate categorical variables
platform_choice = np.random.choice(platforms, n)
brand_choice = np.random.choice(brands, n)
campaign_type_choice = np.random.choice(campaign_types, n)
day_of_week = np.random.randint(0, 7, n)
is_weekend = (day_of_week >= 5).astype(int)

# Step 2: Generate spend
spend = np.random.uniform(500, 50000, n).round(2)

# Step 3: CPM (Cost Per Thousand Impressions) settings
cpm_bases = {
    'Sponsored_Products': 130.0,
    'Sponsored_Brands': 200.0,
    'Sponsored_Display': 80.0
}
platform_cpm_mult = {
    'Blinkit': 1.0,
    'Zepto': 1.1,
    'Swiggy': 0.9
}

# Step 4: Base CTR settings
ctr_bases = {
    'Sponsored_Products': 0.035,
    'Sponsored_Brands': 0.020,
    'Sponsored_Display': 0.008
}

# Step 5: Base Conversion Rate (CR) settings
cr_bases = {
    'Sponsored_Products': 0.08,
    'Sponsored_Brands': 0.05,
    'Sponsored_Display': 0.02
}
platform_cr_mult = {
    'Blinkit': 1.1,
    'Zepto': 1.0,
    'Swiggy': 0.9
}

# Step 6: Brand-specific Average Order Value (AOV) settings
brand_aovs = {
    'Surf_Excel': 750.0,
    'Dove': 550.0,
    'Knorr': 150.0,
    'Cadbury': 250.0,
    'Oreo': 120.0,
    'Sunfeast': 180.0,
    'Bingo': 90.0,
    'Ariel': 680.0
}

# Run sequential simulation
impressions = np.zeros(n, dtype=int)
clicks = np.zeros(n, dtype=int)
orders = np.zeros(n, dtype=int)
revenue = np.zeros(n)

for i in range(n):
    p = platform_choice[i]
    b = brand_choice[i]
    ct = campaign_type_choice[i]
    sp = spend[i]
    
    # Generate CPM with some random variance
    cpm = cpm_bases[ct] * platform_cpm_mult[p] * np.random.uniform(0.85, 1.15)
    
    # Calculate Impressions (with noise)
    imp = int(round((sp / cpm) * 1000))
    imp = max(imp, 100)  # Ensure a minimum number of impressions
    impressions[i] = imp
    
    # Calculate CTR (with noise) and weekend boost
    weekend_ctr_mult = 1.2 if is_weekend[i] == 1 else 1.0
    ctr = ctr_bases[ct] * weekend_ctr_mult * np.random.uniform(0.7, 1.3)
    ctr = max(0.001, min(ctr, 0.15))
    
    # Calculate Clicks
    clk = int(round(imp * ctr))
    clk = max(clk, 1)    # Ensure at least 1 click if there is spend
    clicks[i] = clk
    
    # Calculate Conversion Rate (with noise) and weekend boost
    weekend_cr_mult = 1.3 if is_weekend[i] == 1 else 1.0
    cr = cr_bases[ct] * platform_cr_mult[p] * weekend_cr_mult * np.random.uniform(0.65, 1.35)
    cr = max(0.005, min(cr, 0.25))
    
    # Calculate Orders
    ord_cnt = int(round(clk * cr))
    orders[i] = ord_cnt
    
    # Calculate Brand-specific AOV
    aov = brand_aovs[b] * np.random.uniform(0.9, 1.1)
    
    # Calculate Revenue
    rev = ord_cnt * aov * np.random.uniform(0.95, 1.05)
    revenue[i] = round(max(0.0, rev), 2)

# Build DataFrame
df = pd.DataFrame({
    'platform': platform_choice,
    'brand': brand_choice,
    'campaign_type': campaign_type_choice,
    'spend': spend,
    'impressions': impressions,
    'clicks': clicks,
    'orders': orders,
    'revenue': revenue,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend
})

df['CTR'] = (df['clicks'] / df['impressions']).round(4)
df['ROAS'] = (df['revenue'] / df['spend']).round(2)
df['CPC'] = (df['spend'] / df['clicks']).round(2)

df['is_risky'] = (df['ROAS'] < 3.0).astype(int)

df.to_csv('campaign_data.csv', index=False)
print(f"Data generated! Shape: {df.shape}")
print("is_risky class distribution:")
print(df['is_risky'].value_counts())