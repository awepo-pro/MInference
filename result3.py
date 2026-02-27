import pandas as pd

# Input data parsed from the provided text
data = [
    {
        "block_size": 16,
        "prefix": "2_000",
        "total": "10_000",
        "minf without": 0.008814573287963867,
        "minf with": 0.012950444221496582,
    },
    {
        "block_size": 16,
        "prefix": "10_000",
        "total": "100_000",
        "minf without": 0.01593472957611084,
        "minf with": 0.02128915786743164,
    },
    {
        "block_size": 16,
        "prefix": "2_000",
        "total": "100_000",
        "minf without": 0.01597435474395752,
        "minf with": 0.02111937999725342,
    },
    {
        "block_size": 16,
        "prefix": "20_000",
        "total": "100_000",
        "minf without": 0.015974974632263182,
        "minf with": 0.020994114875793456,
    },
    {
        "block_size": 16,
        "prefix": "30_000",
        "total": "100_000",
        "minf without": 0.015818333625793456,
        "minf with": 0.02088320255279541,
    },
    {
        "block_size": 32,
        "prefix": "30_000",
        "total": "100_000",
        "minf without": 0.015818333625793456,
        "minf with": 0.02016923427581787,
    },
    {
        "block_size": 64,
        "prefix": "30_000",
        "total": "100_000",
        "minf without": 0.015818333625793456,
        "minf with": 0.019948506355285646,
    },
    {
        "block_size": 64,
        "prefix": "30_000",
        "total": "1_000_000",
        "minf without": 0.19187586307525634,
        "minf with": 0.2646173000335693,
    },
    {
        "block_size": 64,
        "prefix": "50_000",
        "total": "1_000_000",
        "minf without": 0.19187586307525634,
        "minf with": 0.22118089199066163,
    },
    {
        "block_size": 64,
        "prefix": "100_000",
        "total": "1_000_000",
        "minf without": 0.19187586307525634,
        "minf with": 0.22092649936676026,
    },
    {
        "block_size": 64,
        "prefix": "300_000",
        "total": "1_000_000",
        "minf without": 0.19187586307525634,
        "minf with": 0.22034864425659179,
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Define column order
cols = list(data[0].keys())


# Reorder and handle missing values
df = df[cols].fillna("None")

# Output as Markdown
print(df.to_markdown(index=False))
