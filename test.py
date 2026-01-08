import geopandas as gpd
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import os
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
 
 
# %%
import redshift_connector
import pandas as pd
 
conn = redshift_connector.connect(
    host='',
    database='',
    user='',
    password='',
    port=
)
 
query = """
WITH active_customers AS (
    -- All active loan customers for a specific country/region (your base of customers)
    SELECT DISTINCT customer_id
    FROM your_database.loan_accounts  -- REPLACE with your actual loan account table
    WHERE state = 'active'
      AND customer_id LIKE 'your_prefix%'  -- e.g., 'ug%' or whatever prefix identifies the target group
      -- Optional: add a single customer filter for testing, e.g., AND customer_id = 'test_123'
),

payment_metrics AS (
    -- 6-month (180-day) window for payments and PVE calculation
    SELECT
        t.customer_id,
        COUNT(DISTINCT t.primary_id)                                      AS all_time_payment_frequency,
        SUM(t.amount / 100.0)                                             AS all_time_total_paid,
        MAX(t.added_at_utc)                                               AS last_payment_date_all_time,
        EXTRACT(DAY FROM CURRENT_DATE - MAX(t.added_at_utc))               AS recency_days_all_time,
        MAX(t.daily_rate) / 100.0                                          AS current_daily_rate,
        (MAX(t.daily_rate) / 100.0) * 180                                  AS expected_pay_180d,

        -- PVE based on last 6 months only
        CASE
            WHEN MAX(t.daily_rate) = 0 THEN NULL
            ELSE ROUND( (SUM(t.amount)/100.0) / ((MAX(t.daily_rate)/100.0) * 180) * 100 , 2)
        END                                                               AS pve_percent_last_180d,

        -- PVE category (6 months)
        CASE
            WHEN MAX(t.daily_rate) = 0 THEN 'No Rate'
            WHEN SUM(t.amount) / NULLIF((MAX(t.daily_rate)/100.0)*180, 0) >= 100 THEN '100%+ (Super Payer)'
            WHEN SUM(t.amount) / NULLIF((MAX(t.daily_rate)/100.0)*180, 0) >= 90  THEN '90–99% (Excellent)'
            WHEN SUM(t.amount) / NULLIF((MAX(t.daily_rate)/100.0)*180, 0) >= 70  THEN '70–89% (Good)'
            WHEN SUM(t.amount) / NULLIF((MAX(t.daily_rate)/100.0)*180, 0) >= 50  THEN '50–69% (Average)'
            ELSE 'Below 50% (At Risk)'
        END                                                               AS pve_category_180d

    FROM your_database.loan_transactions t  -- REPLACE with your actual transactions table
    WHERE t.country = 'YOUR_COUNTRY_CODE'      -- e.g., 'UG' – replace with your target country
      AND t.transaction_type IN ('Payment', 'OverPayment Reducing', 'Migration Amount')
      AND t.added_at_utc >= CURRENT_DATE - INTERVAL '180 days'
      AND t.customer_id IN (SELECT customer_id FROM active_customers)
    GROUP BY t.customer_id
),

demographics AS (
    SELECT
        cs.customer_id,
        COALESCE(
            EXTRACT(YEAR FROM cs.birthdate)::INTEGER,
            pd.birth_year,
            ps.year_of_birth
        ) AS year_of_birth,
        pd.gender,
        pd.occupation,
        pd.area1,
        pd.area2
    FROM your_sensitive_schema.customers_sensitive cs          -- REPLACE with your sensitive customer table
    LEFT JOIN your_database.customer_details pd                -- REPLACE with your public/customer details table
           ON pd.customer_id = cs.customer_id
    LEFT JOIN your_sensitive_schema.person_sensitive ps        -- REPLACE with any additional sensitive person table
           ON ps.primary_phone = cs.primary_phone_number
    WHERE cs.customer_id LIKE 'your_prefix%'                   -- Same prefix as above
)

-- Final output: ALL active customers + 6-month PVE + demographics
SELECT
    a.customer_id,
    COALESCE(p.all_time_payment_frequency, 0)          AS all_time_payment_frequency,
    COALESCE(p.all_time_total_paid, 0)                 AS all_time_total_paid,
    p.last_payment_date_all_time,
    p.recency_days_all_time,
    COALESCE(p.current_daily_rate, 0)                  AS current_daily_rate,
    COALESCE(p.expected_pay_180d, 0)                   AS expected_pay_last_6m,
    p.pve_percent_last_180d,
    COALESCE(p.pve_category_180d, 'No Payment in 180d') AS pve_category_last_6m,
    
    d.year_of_birth,
    d.gender,
    d.occupation,
    d.area1,
    d.area2
FROM active_customers a
LEFT JOIN payment_metrics p ON p.customer_id = a.customer_id
LEFT JOIN demographics d   ON d.customer_id = a.customer_id
ORDER BY p.pve_percent_last_180d DESC NULLS LAST

"""
 
df = pd.read_sql(query, conn)
print(df.head(5))
 
 
conn.close()
 
 
 
# %%
 
df.head(5)
 
# %% [markdown]
# # Exploratory Data Analysis (EDA)
 
# %%
#Check the number of rows and columns.
df.shape
 
 
# %%
duplicate_counts = df['customer_id'].value_counts()
print(duplicate_counts)
 
# %%
df.drop_duplicates(subset=['customer_id'], inplace=True)
 
# %%
df.info()
 
# %%
df.isnull().sum()
 
# %% [markdown]
# # Data Cleaning
#
 
# %%
df.dropna(subset=['last_payment_date_all_time'], inplace=True)
 
# %%
df.isnull().sum()
 
# %%
df['year_of_birth'] = df['year_of_birth'].ffill()
 
 
# %%
current_year = pd.to_datetime('now').year
df['age'] = current_year - df['year_of_birth']
df['age'] = df['age'].astype('Int64')
 
 
# %%
df['age'] = df['age'].replace(2025, np.nan)
# Calculate the mean of 'age' after replacing 2025 with NaN
mean_age = df['age'].mean()
# Round the mean age to the nearest integer before filling NaNs, then cast to Int64
df['age'] = df['age'].fillna(round(mean_age)).astype('Int64')
df.sample(5)
 
# %%
df.isnull().sum()
 
# %%
df['gender_encoded'] = np.where(df['gender'] == 'male', 0, 1)
df['gender_encoded'] = np.where(df['gender'].isnull(), 0, df['gender_encoded'])
df.sample(5)
 
# %%
df['occupation'] = df['occupation'].fillna('Unknown')
 
# %%
df['area1'] = df['area1'].fillna('Eastern')
 
# %%
conditions = [
    df['area1'] == 'Eastern',
    df['area1'] == 'Northern',
    df['area1'] == 'Central'
]
choices = [0, 1, 2]
df['area1_encoded'] = np.select(conditions, choices, default=3)
df.sample(5)
 
 
 
# %%
df.isnull().sum()
 
# %%
df['occupation'].unique()
 
# %%
df['newoccupation'] = np.select(
    [
        df['occupation'].str.contains(r'farm|agric|cattle|fish|far|fam', case=False, na=False),
        df['occupation'].str.contains(r'business|shop|market|trader|mm\s*agent|broker|money\s*changer|tailor|butcher|hair|Casual|Brick|Saloon|Welding|Cashier|Sale|Money|Self'
        , case=False, na=False),
 
        df['occupation'].str.contains(r'accountant|health|teacher|bank|religious|ngo|journal|tour\s*guide|Pastor|nurse|school|Mid-Wife|medical|attendant|Clinic|Bursar|Lawyer', case=False, na=False),
 
        df['occupation'].str.contains(r'office|government|civic|civil\s*servant', case=False, na=False),
 
        df['occupation'].str.contains(r'boda|taxi|driver|Transportation', case=False, na=False),
        df['occupation'].str.contains(r'engineer|construct|mechanic|technician|carpenter|electrician|Builder', case=False, na=False),
        df['occupation'].str.contains(r'police|updf|prisons|security\s*guard|soldier|Force|MILTARY', case=False, na=False),
        df['occupation'].str.contains(r'herbal|other|not\s*disclosed|not\s*employed', case=False, na=False),
    ],
    [
        'Agriculture/Farming',
        'Business/Entrepreneurship',
        'Service Sector',
        'Service Sector',
        'Transportation',
         'Service Sector',
        'Service Sector',
        'Other',
    ],
    default='Other'
)
df.sample(5)
 
# %%
#occupation mapping
occupation_mapping = {
    'Agriculture/Farming': 0,
    'Business/Entrepreneurship': 1,
    'Service Sector': 2,
    'Transportation': 3,
    'Other': 4
}
df['occupation_numeric'] = df['newoccupation'].map(occupation_mapping)
display(df[['newoccupation', 'occupation_numeric']].head(10))
 
# %%
df['newoccupation'].value_counts()
 
# %% [markdown]
# ### Encoded data
 
# %%
data = df[['customer_id','all_time_payment_frequency','all_time_total_paid','pve_percent_last_180d','recency_days_all_time','pve_percent_last_180d','age','gender_encoded','area1_encoded','occupation_numeric']]
data.head(10)
 
# %% [markdown]
# ### **Recency Score calculator**
 
# %%
# ...existing code...
# 1. Recency score (lower days = better → reverse the quintiles)
# Use ranks to avoid qcut dropping bins when there are many duplicate values,
# then create 5 quantile bins labeled 5..1 (5 = most recent). Result is Int64 with NA preserved.
ranks = df['recency_days_all_time'].rank(method='first', na_option='keep')
df['R_score'] = pd.qcut(ranks, q=5, labels=[5, 4, 3, 2, 1])
df['R_score'] = df['R_score'].astype('Int64')
display(df[['recency_days_all_time', 'R_score']].head(10))
# ...existing code...
 
# %%
# distribution of R_score
df['R_score'].value_counts()
 
# %%
#average by Cluster distribution of days by F_score
 
average_recency_by_r_score = df.groupby('R_score')['recency_days_all_time'].mean()
print(average_recency_by_r_score)
 
# %%
data = df[['customer_id','all_time_payment_frequency','all_time_total_paid','pve_percent_last_180d','recency_days_all_time','pve_percent_last_180d','age','gender_encoded','area1_encoded','occupation_numeric', 'R_score']]
data.head(10)
 
# %% [markdown]
# ### **Frequency Score calculator**
 
# %%
# Option A: Standard 1–5 score (1 = lowest frequency, 5 = highest frequency)
# Create the quintiles, allowing duplicates to be dropped if necessary.
# pd.qcut will return a Categorical object, whose codes can be used as 0-indexed labels.
qcut_categories = pd.qcut(
    df['all_time_payment_frequency'], # Apply to df directly
    q=5,
    duplicates='drop'
)
 
# Convert the categorical codes to 1-indexed integer scores
df['F_score'] = qcut_categories.cat.codes + 1 # Add to df
 
# Display the head of the relevant columns
display(df[['all_time_payment_frequency', 'F_score']].head())
 
# %%
# Option A: Standard 1–5 score (1 = lowest frequency, 5 = highest frequency)
# To ensure all 5 scores are present, we'll use pd.cut to create 5 equal-width bins
# as pd.qcut with duplicates='drop' might result in fewer than 5 unique scores.
 
# Determine the min and max values for binning
min_val = df['all_time_payment_frequency'].min()
max_val = df['all_time_payment_frequency'].max()
 
# Handle the edge case where all frequencies are the same (cannot create 5 bins)
if min_val == max_val:
    df['F_score'] = 3 # Assign a default middle score if all frequencies are the same
else:
    # Create 5 equal-width bins and assign scores 1-5
    df['F_score'] = pd.cut(
        df['all_time_payment_frequency'],
        bins=5,
        labels=[1, 2, 3, 4, 5], # Explicitly label 1-5
        include_lowest=True,
        right=True # Ensure bins are inclusive on the right (except for the first bin's lower bound)
    ).astype(int)
 
# Display the head of the relevant columns
display(df[['all_time_payment_frequency', 'F_score']].head())
# Display value counts to confirm that all 5 scores are present
display(df['F_score'].value_counts().sort_index())
 
# %%
df['F_score'].value_counts()
 
# %%
#distribution of days by F_score
average_frequency_by_F_score = df.groupby('F_score')['all_time_payment_frequency'].mean()
print(average_frequency_by_F_score)
 
# %%
data = df[['customer_id','all_time_payment_frequency','all_time_total_paid','pve_percent_last_180d','recency_days_all_time','pve_percent_last_180d','age','gender_encoded','area1_encoded','occupation_numeric', 'R_score','F_score']]
data.head(10)
 
# %% [markdown]
# ### **PVE Score calculator**
 
# %%
# 3. PVE score (use PVE, not total collect — because of different daily rates)
 
df['V_score'] = pd.qcut(
    df['pve_percent_last_180d'],
    q=5,
    labels=[1,2,3,4,5],
    duplicates='drop'
)
 
display(df[['pve_percent_last_180d', 'V_score']].head())
 
# %%
df['V_score'].value_counts()
 
# %%
#distribution of days by V_score
average_pve_by_V_score = df.groupby('V_score')['pve_percent_last_180d'].mean()
print(average_pve_by_V_score)
 
# %%
data = df[['customer_id','all_time_total_paid','pve_percent_last_180d','all_time_payment_frequency','recency_days_all_time','age','gender_encoded','area1_encoded','occupation_numeric', 'R_score','F_score','V_score']]
data.head(10)
 
# %% [markdown]
# ### **RFV Combination string**
 
# %%
# 4. Combine into RFV string
df['RFV_score'] = df ['R_score'].astype(str) + df ['F_score'].astype(str) + df ['V_score'].astype(str)
 
# %%
data = df[['customer_id','all_time_payment_frequency','all_time_total_paid','pve_percent_last_180d','recency_days_all_time','age','gender','gender_encoded','area1_encoded','occupation_numeric', 'R_score', 'F_score', 'V_score', 'RFV_score', 'area2']]
data.head(10)
 
# %% [markdown]
# ### **Clustering by combination**
 
# %%
## this
 
# RFV
# 111 → 555 combination is correctly placed
 
def rfm_6_segments(rfv_score):
    """
    Input: '555', '341', '112', etc. (string of 3 digits)
    Output: One of 6 standard segments
    """
 
    if not isinstance(rfv_score, str) or len(rfv_score) != 3:
        return 'Low-Value / Others'
 
    try:
        r = int(rfv_score[0])
        f = int(rfv_score[1])
        v = int(rfv_score[2])
    except:
        return 'Low-Value / Others'
 
    if r >= 4 and f >= 4:
        return 'Champions'
 
    elif r >= 4:
        return 'Potential Loyalists'
 
    elif (r <= 3 and f >= 4) or (r <= 2 and v >= 4):
        return 'At-Risk'
 
    elif v >= 4 and f <= 3:
        return 'Big Spenders'
 
    elif r <= 2 and v <= 3:
        return 'Hibernating'
 
    else:
        return 'Low-Value / Others'
 
 
data['Customer_Segment'] = data['RFV_score'].apply(rfm_6_segments)
 
segment_priority = {
    'Champions': 1,
    'Potential Loyalists': 2,
    'At-Risk': 3,
    'Big Spenders': 4,
    'Hibernating': 5,
    'Low-Value / Others': 6
}
data['Segment_Order'] = data['Customer_Segment'].map(segment_priority)
 
# Quick validation — should show exactly 6 segments and 100% coverage
print(data['Customer_Segment'].value_counts().sort_index())
print("\nTotal customers segmented:", len(data))
 
# %%
data['RFV_score_numeric'] = pd.to_numeric(data['RFV_score'], errors='coerce').fillna(0).astype(int)
average_rfv_by_segment = data.groupby('Customer_Segment')['RFV_score_numeric'].mean()
print(average_rfv_by_segment)
 
# %%
data['R_score'] = pd.to_numeric(data['R_score'], errors='coerce').fillna(0).astype(int)
data['F_score'] = pd.to_numeric(data['F_score'], errors='coerce').fillna(0).astype(int)
data['V_score'] = pd.to_numeric(data['V_score'], errors='coerce').fillna(0).astype(int)
 
average_scores_by_segment = data.groupby('Customer_Segment')[['R_score', 'F_score', 'V_score']].mean()
print(average_scores_by_segment)
 
# %%
data['Customer_Segment'].value_counts()
 
# %%
data.head(10)
 
# %% [markdown]
# ## **Demographic Segmentation**
 
# %% [markdown]
#
 
# %%
# Age groups
bins = [0, 25, 35, 45, 100]
labels = ['18-25', '26-35', '36-45', '46+']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, include_lowest=True)
data.head(1)
 
# %%
# 1. Gender + Segment → females will dominate Golden
pd.crosstab(data['gender_encoded'], data['Customer_Segment'], normalize='columns')
 
# 2. Age group + Segment → 26–35 and 36–45 will dominate Golden & Win-Back
pd.crosstab(data['age_group'], data['Customer_Segment'], normalize='columns')
 
# 3. Occupation + Segment → Business/Entrepreneurship & Service Sector = your whales
pd.crosstab(data['occupation_numeric'], data['Customer_Segment'], normalize='columns')
 
# 4. Region + Segment → Central (Kampala) will have 60–80% of Golden customers
pd.crosstab(data['area1_encoded'], data['Customer_Segment'], normalize='columns')
 
# %%
# 3. FINAL SEGMENTATION (11 segments – best practice 2025)
conditions = [
    # 1–2: Service Sector = golden segment (salaried teachers, nurses, etc.)
    (data['occupation_numeric'] == 2),
 
    # 3–4: Young Male Boda/Transport → highest risk
    (data['occupation_numeric'] == 3) & (data['gender'] == 'male') & (data['age_group'].isin(['18-25', '26-35'])),
 
    # 5: Young Female Business (18–35) → excellent repayment
    (data['occupation_numeric'] == 1) & (data['gender'] == 'female') & (data['age_group'].isin(['18-25', '26-35'])),
 
    # 6: Mature Business Owner (46+) → high value
    (data['occupation_numeric'] == 1) & (data['age_group'] == '46+'),
 
    # 7–8: Female vs Male Farmers
    (data['occupation_numeric'] == 0) & (data['gender'] == 'female'),
    (data['occupation_numeric'] == 0) & (data['gender'] == 'male'),
 
    # 9: Older Transport (36+) → much better than young bodas
    (data['occupation_numeric'] == 3) & (data['age_group'].isin(['36-45', '46+'])),
 
    # 10: Male Business (all ages except mature)
    (data['occupation_numeric'] == 1) & (data['gender'] == 'male'),
 
    # 11: Everything else (Other occupation + rare combos)
    (data['occupation_numeric'] == 4)
]
 
choices = [
    'Service/Salaried',                  # Best segment
    'Young Male Transport (18-35)',      # Worst segment
    'Young Female Business (18-35)',     # Excellent
    'Mature Business Owner (46+)',       # High ARPU
    'Female Farmer',                     # Very good
    'Male Farmer',                       # Average to weak
    'Mature Transport (36+)',            # Acceptable
    'Male Business',                     # Medium
    'Other Occupation'                   # Neutral
]
 
data['demo_segment'] = np.select(conditions, choices, default='Other Occupation')
 
# Optional: Add short codes for CRM/agents
segment_code = {
    'Service/Salaried'              : 'S1',
    'Young Male Transport (18-35)'  : 'T1',  # High risk
    'Young Female Business (18-35)' : 'B1',  # Star segment
    'Mature Business Owner (46+)'   : 'B2',
    'Female Farmer'                 : 'F1',
    'Male Farmer'                   : 'F2',
    'Mature Transport (36+)'        : 'T2',
    'Male Business'                 : 'B3',
    'Other Occupation'              : 'XX'
}
data['segment_code'] = data['demo_segment'].map(segment_code)
 
# Quick validation - removed non-existent columns
data.groupby('demo_segment').agg({'customer_id': 'count'}).round(3)
 
# %%
data.head(10)
 
# %% [markdown]
# # Geographic segmentation
 
# %%
# CLEAN UGANDAN DISTRICT TIERS – NO DUPLICATES – SOLAR READY
district_tiers = {
    'Greater Kampala': ['Kampala', 'Wakiso', 'Mukono'],
 
    'Major Towns': ['Jinja', 'Mbale', 'Gulu', 'Lira', 'Mbarara', 'Masaka', 'Hoima',
                             'Arua', 'Soroti', 'Tororo', 'Iganga', 'Fort Portal', 'Kasese',
                             'Masindi', 'Bushenyi', 'Kabale', 'Rukungiri', 'Ntungamo'],
 
    'Secondary Towns': ['Mityana', 'Kamuli', 'Pallisa', 'Busia', 'Kapchorwa', 'Kabarole',
                                 'Nebbi', 'Yumbe', 'Moyo', 'Adjumani', 'Kitgum', 'Moroto', 'Kween',
                                 'Kayunga', 'Isingiro', 'Kiruhura', 'Lyantonde', 'Kanungu', 'Kisoro',
                                 'Ibanda', 'Sheema', 'Bushenyi', 'Mubende', 'Kalangala','Mpigi', 'Buikwe', 'Luwero'],
 
    'Rural Accessible': ['Manafwa', 'Budaka', 'Bududa', 'Sironko', 'Bulambuli', 'Kumi',
                                  'Ngora', 'Serere', 'Katakwi', 'Amuria', 'Napak', 'Abim', 'Kotido',
                                  'Kaabong', 'Amudat', 'Nakapiripirit', 'Ntoroko', 'Kyegegwa',
                                  'Kyankwanzi', 'Kiboga', 'Nakaseke', 'Luuka', 'Buyende', 'Kaliro',
                                  'Namutumba', 'Bugiri', 'Mayuge', 'Kamwenge', 'Kyenjojo', 'Kibaale',
                                  'Kagadi', 'Kakumiro', 'Buliisa', 'Kiryandongo'],
 
    'Deep Rural & Remote': ['Amolatar', 'Dokolo', 'Oyam', 'Apac', 'Kole', 'Otuke',
                                     'Alebtong', 'Agago', 'Pader', 'Lamwo', 'Nwoya', 'Omoro',
                                     'Zombo', 'Pakwach', 'Maracha', 'Koboko', 'Obongi', 'Madi-Okollo',
                                     'Terego', 'Karenga', 'Buhweju', 'Rubirizi', 'Bundibugyo',
                                     'Sembabule', 'Lwengo', 'Kalungu', 'Bukomansimbi', 'Gomba']
}
 
# Islands (treated as Tier 5 for logistics)
island_districts = ['Buvuma', 'Kalangala', 'Namayingo', 'Lake Victoria', 'Lake Albert']
for d in island_districts:
    district_tiers['Deep Rural & Remote'].append(d)
 
# Create mapping (no duplicates!)
district_to_tier = {}
for tier_name, districts in district_tiers.items():
    for d in districts:
        district_to_tier[d.strip().title()] = tier_name
 
# Apply to your data
data['district_clean'] = data['area2'].str.strip().str.title()
data['location_tier'] = data['district_clean'].map(district_to_tier).fillna('Deep Rural & Remote')
 
# Quick check – you should see NO errors and clean counts
print("Location tiers (no duplicates!)")
print(data['location_tier'].value_counts().sort_index())
 
print("\nAny districts not recognised? (should be very few or zero)")
print(data[data['location_tier'].str.contains('Deep Rural')]['district_clean'].unique())
 
# %%
data.head()
 
# %%
segment = data[['customer_id','all_time_payment_frequency','recency_days_all_time','pve_percent_last_180d','R_score', 'F_score', 'V_score', 'RFV_score','age_group','gender_encoded','occupation_numeric','Customer_Segment','demo_segment', 'location_tier']]
segment.head(10)
 
# %%
segment.to_excel('drive/My Drive/Colab Notebooks/cvm.xlsx', index=False)
print('DataFrame saved to cvm.xlsx in Google Drive')
## reloader

