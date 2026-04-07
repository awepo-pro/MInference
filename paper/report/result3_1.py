import pandas as pd



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

# Reorder columns for better readability
cols = list(data[0].keys())
df = df[cols]

# Replace NaN with 'None'
df_filled = df.fillna("None")

# Generate Markdown
markdown_table = df_filled.to_markdown(index=False)
print(markdown_table)
