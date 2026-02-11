import pandas as pd

# Input data parsed from the provided text
data = [
    {
        "context-window": "17_000",
        "max_model_len": 24576,
        "output token": 1,
        "block_size": 16,
        "prefix": "4_000",
        "compute": "12_000",
        "minf without": 1.4903662204742432,
        "minf with": 1.6751995086669922,
        "normal without": 1.0289995670318604,
        "normal with": 0.7779994010925293,
    },
    {
        "context-window": "20_000",
        "max_model_len": 24576,
        "output token": 1,
        "block_size": 32,
        "prefix": "4_000",
        "compute": "15_000",
        "minf without": 1.6228184700012207,
        "minf with": 1.8437342643737793,
        "normal without": 1.1821706295013428,
        "normal with": 0.9907801151275635,
    },
    {
        "context-window": "24_000",
        "max_model_len": 24576,
        "output token": 1,
        "block_size": 32,
        "prefix": "4_000",
        "compute": "19_000",
        "minf without": 1.8255112171173096,
        "minf with": 2.113152265548706,
        "normal without": 1.439767599105835,
        "normal with": 1.2752225399017334,
    },
    {
        "context-window": "24_000",
        "max_model_len": 24576,
        "output token": 1,
        "block_size": 32,
        "prefix": "9_000",
        "compute": "14_000",
        "minf without": 1.8318281173706055,
        "minf with": 1.817716360092163,
        "normal without": 1.4441075325012207,
        "normal with": 0.9856750965118408,
    },
    {
        "context-window": "24_000",
        "max_model_len": 24576,
        "output token": 1,
        "block_size": 32,
        "prefix": "13_000",
        "compute": "10_000",
        "minf without": 1.8195648193359375,
        "minf with": 1.5577092170715332,
        "normal without": 1.4451830387115479,
        "normal with": 0.7132761478424072,
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
