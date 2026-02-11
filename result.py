import pandas as pd

# Input data parsed from the provided text
data = [
    {
        "context-window": "17_000",
        "max_model_len": 24576,
        "output token": 1,
        "minf without": 0.7284053087234497,
        "minf with": 3.0434834241867064,
        "normal without": 4.5304972410202025,
        "normal with": 0.08895010948181152,
    },
    {
        "context-window": "20_000",
        "max_model_len": 24576,
        "output token": 1,
        "minf without": 4.715971827507019,
        "minf with": 0.7262516736984252,
        "normal without": 5.71290156841278,
        "normal with": 0.08533780574798584,
    },
    {
        "context-window": "24_000",
        "max_model_len": 24576,
        "output token": 1,
        "minf without": 5.650742697715759,
        "minf with": 0.8083413124084473,
        "normal without": 7.337783217430115,
        "normal with": 0.1666180372238159,
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Define column order
cols = [
    "context-window", 
    "max_model_len", 
    "output token", 
    "minf with", 
    "minf without", 
    "normal with", 
    "normal without"
]

# Reorder and handle missing values
df = df[cols].fillna("None")

# Output as Markdown
print(df.to_markdown(index=False))
