import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data from result3_1.py
meta = {
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_size": 128,
    "dtype": "bfloat16",
    "block_index (top-k)": 15,
    "git": "417e9f27794e2316ab145b5bdcb8e91a0d8f851f"
}

data = [
    {
        "block size": "64",
        "prefix": "1_000",
        "total": "5_000",
        "minf": 0.01786496639251709,
        "minf with paged attention": 0.022961735725402832,
    },
    {
        "block size": "64",
        "prefix": "2_000",
        "total": "5_000",
        "minf": 0.01786496639251709,
        "minf with paged attention": 0.02103297710418701,
    },
    {
        "block size": "64",
        "prefix": "3_000",
        "total": "5_000",
        "minf": 0.01786496639251709,
        "minf with paged attention": 0.020474720001220702,
    },
    {
        "block size": "64",
        "prefix": "4_000",
        "total": "5_000",
        "minf": 0.01786496639251709,
        "minf with paged attention": 0.019861006736755372,
    },


    {
        "block size": "64",
        "prefix": "10_000",
        "total": "50_000",
        "minf": 0.0346158504486084,
        "minf with paged attention": 0.03162274360656738,
    },
    {
        "block size": "64",
        "prefix": "20_000",
        "total": "50_000",
        "minf": 0.0346158504486084,
        "minf with paged attention": 0.0280362606048584,
    },
    {
        "block size": "64",
        "prefix": "30_000",
        "total": "50_000",
        "minf": 0.0346158504486084,
        "minf with paged attention": 0.024527955055236816,
    },
    {
        "block size": "64",
        "prefix": "40_000",
        "total": "50_000",
        "minf": 0.0346158504486084,
        "minf with paged attention": 0.02142155170440674,
    },


    {
        "block size": "64",
        "prefix": "10_000",
        "total": "100_000",
        "minf": 0.05748934745788574,
        "minf with paged attention": 0.05591483116149902,
    },
    {
        "block size": "64",
        "prefix": "20_000",
        "total": "100_000",
        "minf": 0.057665324211120604,
        "minf with paged attention": 0.049153804779052734
    },
    {
        "block size": "64",
        "prefix": "30_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.04763076305389404
    },
    {
        "block size": "64",
        "prefix": "40_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.04140341281890869
    },
    {
        "block size": "64",
        "prefix": "50_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.037438583374023435
    },
    {
        "block size": "64",
        "prefix": "60_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.024165630340576172
    },
    {
        "block size": "64",
        "prefix": "70_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.023226022720336914
    },
    {
        "block size": "64",
        "prefix": "80_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.02142157554626465
    },
    {
        "block size": "64",
        "prefix": "90_000",
        "total": "100_000",
        "minf": 0.05771183967590332,
        "minf with paged attention": 0.021250438690185548
    },


    {
        "block size": "64",
        "prefix": "10_000",
        "total": "500_000",
        "minf": 0.2937402486801147,
        "minf with paged attention": 0.3365923404693604
    },
    {
        "block size": "64",
        "prefix": "20_000",
        "total": "500_000",
        "minf": 0.2937402486801147,
        "minf with paged attention": 0.3201284408569336
    },
    {
        "block size": "64",
        "prefix": "30_000",
        "total": "500_000",
        "minf": 0.2937402486801147,
        "minf with paged attention": 0.31941022872924807
    },
    {
        "block size": "64",
        "prefix": "100_000",
        "total": "500_000",
        "minf": 0.2937402486801147,
        "minf with paged attention": 0.2742668867111206
    },
    {
        "block size": "64",
        "prefix": "200_000",
        "total": "500_000",
        "minf": 0.2937402486801147,
        "minf with paged attention": 0.2239511251449585
    },
    {
        "block size": "64",
        "prefix": "300_000",
        "total": "500_000",
        "minf": 0.2937402486801147,     # sometimes insufficient memory
        "minf with paged attention": 0.15587420463562013
    },


    {
        "block size": "64",
        "prefix": "200_000",
        "total": "600_000",
        "minf": None,       # memory insufficience
        "minf with paged attention": 0.2734508991241455
    },
]

# 1. Prepare the DataFrame
df = pd.DataFrame(data)
df['total_int'] = df['total'].str.replace('_', '').astype(int)
df['prefix_int'] = df['prefix'].str.replace('_', '').astype(int)

# 2. Filter for a specific Context Window
total_window = 5000
subset = df[df['total_int'] == total_window].sort_values('prefix_int').copy()

# 3. Plotting
plt.figure(figsize=(10, 7)) # Increased height slightly

# Plot Baseline
plt.plot(subset['prefix_int'], subset['minf'], label='MInference (No Caching)', 
         color='#A9A9A9', linestyle='--', marker='o', linewidth=2)

# Plot Paged Attention
plt.plot(subset['prefix_int'], subset['minf with paged attention'], 
         label='MInference + Prefix Caching', 
         color='#50C878', marker='D', linewidth=4, markersize=8)

# 4. Improved Annotations
y_range = subset['minf with paged attention'].max() - subset['minf with paged attention'].min()
offset = y_range * 0.15 if y_range > 0 else 0.002 # Fallback offset

for i, row in subset.iterrows():
    # Annotate start and end points
    if row['prefix_int'] in [subset['prefix_int'].min(), subset['prefix_int'].max()]:
        diff = row['minf with paged attention'] - row['minf']
        percent_change = (diff / row['minf']) * 100
        
        # Decide if text goes above or below based on the value
        # For 5k window, PA is actually slower (positive diff), so we move text up
        direction = 1 if diff > 0 else -1
        
        plt.annotate(
            f'{percent_change:+.1f}% Latency', 
            xy=(row['prefix_int'], row['minf with paged attention']),
            xytext=(0, 15 * direction), # 15 points offset
            textcoords="offset points",
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            ha='center', va='bottom' if direction > 0 else 'top',
            fontweight='bold', color='firebrick',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none')
        )

# 5. Styling
plt.title(f"Impact of Prefix Caching (Total Context: {total_window//1000}k)\nNote: Caching overhead visible at small scales", fontsize=14, pad=20)
plt.xlabel("Prefix Length (Tokens Cached)", fontsize=12)
plt.ylabel("Latency (Seconds)", fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# Format X-axis
plt.xticks(subset['prefix_int'], [f"{int(p)//1000}k" for p in subset['prefix_int']])

# Set Y-axis limits with some padding to prevent text clipping
plt.ylim(subset['minf'].min() * 0.9, subset['minf with paged attention'].max() * 1.2)

# Use tight_layout with padding
plt.tight_layout(pad=3.0)
plt.show()
