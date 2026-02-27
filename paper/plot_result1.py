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

df = pd.DataFrame(data)

# 1. Load your data
# (Using the data structure you provided in the previous turn)
# Ensure 'total_int' and 'prefix_int' are numeric for plotting
df['total_int'] = df['total'].str.replace('_', '').astype(int)
df['prefix_int'] = df['prefix'].str.replace('_', '').astype(int)

# 2. Setup Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Define the context window sizes we want to compare
# We'll filter for 50k, 100k, and 500k as they have multiple prefix data points
target_totals = [50000, 100000, 500000]
colors = ['#4A90E2', '#50C878', '#FF7F50'] # Blue, Green, Coral

for total, color in zip(target_totals, colors):
    subset = df[df['total_int'] == total].sort_values('prefix_int')
    
    # Plotting Prefix Size vs Latency for MInference with Paged Attention
    plt.plot(subset['prefix_int'], subset['minf with paged attention'], 
             label=f'Total Context: {total//1000}k', 
             color=color, marker='o', linewidth=2.5, markersize=8)

# 3. Aesthetics & Labels
plt.title("Impact of Prefix Caching on MInference Latency", fontsize=15, pad=15)
plt.xlabel("Prefix Size (Tokens Cached)", fontsize=12)
plt.ylabel("Latency (Seconds)", fontsize=12)

# Log scale for X-axis often helps if prefix sizes vary by orders of magnitude
plt.xscale('log') 
plt.xticks([1000, 10000, 100000, 300000], ['1k', '10k', '100k', '300k'])

plt.legend(title="Context Window", fontsize=10)
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()
