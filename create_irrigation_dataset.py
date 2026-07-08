import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/Crop_recommendation.csv')

# Drop the crop label - we only want the real sensor features
df = df.drop('label', axis=1)

# Derive irrigation schedule from real agronomic rules
def assign_irrigation(row):
    rain   = row['rainfall']
    temp   = row['temperature']
    humid  = row['humidity']

    if rain > 200 and humid > 75:
        return 'No Irrigation'
    elif rain < 80 and temp > 28 and humid < 55:
        return 'Daily'
    elif rain < 120 and temp >= 25:
        return 'Weekly'
    elif rain >= 120 and rain <= 220:
        return 'Bi-weekly'
    else:
        return 'Monthly'

df['irrigation_schedule'] = df.apply(assign_irrigation, axis=1)

print(df['irrigation_schedule'].value_counts())
print(df.shape)
df.to_csv('Datasets/Irrigation_recommendation.csv', index=False)
print("Saved Datasets/Irrigation_recommendation.csv")
