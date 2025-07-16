# %%
# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
# %%
# --- Constants ---
CROSSCODER_TYPE = "L1"
LAYER = 15

# --- Data Paths ---
# DATA_PATH = f"steering_{CROSSCODER_TYPE.lower()}_crosscoder_proto_including_reference.csv"
DATA_PATH = f"multiple_prompts_steering_{CROSSCODER_TYPE.lower()}_crosscoder.csv"
ATTRIBUTION_PATH = f"../results/{CROSSCODER_TYPE}-Crosscoder/L{LAYER}R/attributions_mean_last_token_data.json"

# --- Plotting Constants ---
PLOT_FONT_FAMILY = "EB Garamond"
PLOT_FONT_SIZE = 20
COLOR_BOTTOM = "salmon"
COLOR_TOP = "skyblue"
EDGE_COLOR = "white"  # Original value, we will override in plot calls directly

plt.rc('font', family=PLOT_FONT_FAMILY, size=PLOT_FONT_SIZE)
plt.rcParams['axes.titlesize'] = PLOT_FONT_SIZE
plt.rcParams['axes.labelsize'] = PLOT_FONT_SIZE
plt.rcParams['xtick.labelsize'] = PLOT_FONT_SIZE
plt.rcParams['ytick.labelsize'] = PLOT_FONT_SIZE
plt.rcParams['legend.fontsize'] = PLOT_FONT_SIZE

# Set default patch properties to have no edge
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams['patch.linewidth'] = 0

# %%
# --- Load Data ---
df = pd.read_csv(DATA_PATH)
with open(ATTRIBUTION_PATH, 'r') as f:
    attributions = json.load(f)


# %%
# --- Extract top and bottom features from attributions ---
# Sort top features by absolute value
sorted_top_features = sorted(
    attributions["top"], key=lambda x: abs(x["value"]), reverse=True)
sorted_top_indices = [item["index"] for item in sorted_top_features]

# Sort bottom features by absolute value
sorted_bottom_features = sorted(
    attributions["bottom"], key=lambda x: abs(x["value"]), reverse=True)
sorted_bottom_indices = [item["index"] for item in sorted_bottom_features]

# --- Match features in df to top and bottom with rankings ---


def categorize_feature(feature_idx):
    if feature_idx in sorted_top_indices:
        rank = sorted_top_indices.index(feature_idx) + 1
        return f"top{rank}"
    elif feature_idx in sorted_bottom_indices:
        rank = sorted_bottom_indices.index(feature_idx) + 1
        return f"bottom{rank}"
    else:
        return None


# Add attribution_group column
df['attribution_group'] = df['feature_idx'].apply(categorize_feature)

# Display summary
print(f"Added 'attribution_group' column with feature categorization")
print(f"Top features: {len(sorted_top_indices)}")
print(f"Bottom features: {len(sorted_bottom_indices)}")
print(f"Rows with top features: {(df['attribution_group'] == 'top').sum()}")
print(
    f"Rows with bottom features: {(df['attribution_group'] == 'bottom').sum()}")
print(
    f"Rows with uncategorized features: {df['attribution_group'].isna().sum()}")


# %%
# --- Find characters before 'wait' ---
def count_chars_before_wait(text):
    if pd.isna(text):
        return None

    # Convert to lowercase for case-insensitive search
    text_lower = text.lower()

    # Find the position of 'wait'
    wait_pos = text_lower.find('wait')

    # Return the number of characters before 'wait' or -1 if not found
    return wait_pos if wait_pos >= 0 else -1


# Apply the function to create the new column
df['chars_before_wait'] = df['text_after_wait'].apply(count_chars_before_wait)

# Display summary
print(f"Added 'chars_before_wait' column with counts of characters before 'wait' appears")
print(
    f"Number of rows with 'wait' found: {(df['chars_before_wait'] >= 0).sum()}")
print(
    f"Number of rows without 'wait': {(df['chars_before_wait'] == -1).sum()}")
print(
    f"Number of rows with 'wait' as NOT the first token: {(df['chars_before_wait'] > 0).sum()}")


# %%
# --- Histogram of characters before 'wait' ---

# Include all rows, including those where 'wait' wasn't found (-1)
all_counts = df['chars_before_wait'].fillna(-1)

plt.figure(figsize=(10, 6))
bins = [-1.5, -0.5, 0.5] + list(range(1, 21))
# Explicitly set edgecolor to 'none' and linewidth to 0 for the histogram
plt.hist(all_counts, bins=bins, edgecolor='none', linewidth=0)
plt.title('Number of Characters Before "wait"')
plt.xlabel('Number of Characters (-1 = "wait" not found)')
plt.ylabel('Frequency')
plt.grid(alpha=0)
plt.tight_layout()
plt.show()

# Calculate statistics only for valid counts (where 'wait' was found)
valid_counts = df['chars_before_wait'][df['chars_before_wait'] >= 0]
print(f"Mean chars before 'wait': {valid_counts.mean():.2f}")
print(f"Median chars before 'wait': {valid_counts.median():.2f}")


# %%
# --- Constants for filtering ---
FEATURE_GROUP = "bottom"
STRENGTH_VALUE = 0.5
K = 50

# --- Filter for features with specified group and strength ---


def filter_and_analyze_features(df, feature_group, strength_value, k=K):
    """
    Filter dataframe for features with specified group and strength and analyze them.

    Args:
        df: DataFrame containing the data
        feature_group: String indicating which feature group to filter for (e.g., "top", "bottom")
        strength_value: Float value for the strength to filter by
        k: Integer indicating the number of top/bottom features to consider (1 to k)

    Returns:
        filtered_features: DataFrame containing only the filtered rows
        analysis_data: Dictionary containing analysis results
    """
    # Create pattern to match feature_group followed by number between 1 and k
    pattern = f"{feature_group}[1-{k}]$|{feature_group}[1-{k}][0-9]$"
    filtered_features = df[df['attribution_group'].str.match(pattern, na=False) &
                           (df['strength'] == strength_value)]

    analysis_data = {
        "num_rows": len(filtered_features),
        "features_represented": 0,
        "wait_found": 0,
        "wait_percentage": 0,
        "mean_chars": None,
        "median_chars": None,
        "feature_details": [],
        "all_counts": None,
        "valid_counts": None
    }

    # Check if there are any rows matching the criteria
    if len(filtered_features) > 0:
        # Get summary statistics
        analysis_data["features_represented"] = filtered_features['attribution_group'].nunique()

        # Extract unique attribution groups and sort them by the numeric part
        features = filtered_features['attribution_group'].unique()

        # Custom sort function to extract and sort by the numeric part
        def get_feature_number(feature_name):
            # Extract the number from strings like "bottom10", "bottom2", etc.
            try:
                return int(feature_name.split(feature_group)[1])
            except (IndexError, ValueError):
                return 0  # Default value if parsing fails

        sorted_features = sorted(features, key=get_feature_number)

        # Collect feature details
        for feature in sorted_features:
            feature_rows = filtered_features[filtered_features['attribution_group'] == feature]
            chars_values = feature_rows['chars_before_wait'].tolist()
            feature_idx = feature_rows['feature_idx'].iloc[0]
            analysis_data["feature_details"].append({
                "name": feature,
                "feature_idx": feature_idx,
                "chars_values": chars_values
            })

        # Check for 'wait' patterns
        analysis_data["wait_found"] = (
            filtered_features['chars_before_wait'] >= 0).sum()
        if len(filtered_features) > 0:
            analysis_data["wait_percentage"] = analysis_data["wait_found"] / \
                len(filtered_features) * 100

        # Include all rows, including those where 'wait' wasn't found (-1)
        analysis_data["all_counts"] = filtered_features['chars_before_wait'].fillna(
            -1)

        # If there are rows with 'wait', analyze character positions
        if analysis_data["wait_found"] > 0:
            analysis_data["valid_counts"] = filtered_features['chars_before_wait'][filtered_features['chars_before_wait'] >= 0]
            analysis_data["mean_chars"] = analysis_data["valid_counts"].mean()
            analysis_data["median_chars"] = analysis_data["valid_counts"].median()

    return filtered_features, analysis_data


# Apply the filter and analysis for bottom features with strength 0.5
filtered_features, analysis_data = filter_and_analyze_features(
    df, FEATURE_GROUP, STRENGTH_VALUE)

# Display analysis results
print(
    f"\nAnalysis of rows with {FEATURE_GROUP} features (1-{K}) and strength = {STRENGTH_VALUE}:")
print(f"Number of rows: {analysis_data['num_rows']}")

if analysis_data['num_rows'] > 0:
    print(f"Features represented: {analysis_data['features_represented']}")

    print(f"\nFeature chars_before_wait values:")
    for feature_detail in analysis_data['feature_details']:
        print(
            f"  {feature_detail['name']} (feature_idx: {feature_detail['feature_idx']}): {feature_detail['chars_values']}")

    print(
        f"\n'wait' found in {analysis_data['wait_found']} rows ({analysis_data['wait_percentage']:.1f}% of filtered rows)")

    # Plot histogram for this subset, including -1 values
    plt.figure(figsize=(10, 6))
    # Custom bins to properly show -1 separately
    bins = [-1.5, -0.5, 0.5] + list(range(1, 21))
    # Explicitly set edgecolor to 'none' and linewidth to 0 for this histogram
    plt.hist(analysis_data['all_counts'], bins=bins,
             edgecolor='none', linewidth=0)
    plt.title(
        f'Number of Characters Before "wait" ({FEATURE_GROUP.capitalize()} Features 1-{K}, Strength={STRENGTH_VALUE})')
    plt.xlabel('Number of Characters (-1 = "wait" not found)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0)
    plt.tight_layout()
    plt.show()

    # If there are rows with 'wait', display statistics
    if analysis_data['wait_found'] > 0:
        print(f"Mean chars before 'wait': {analysis_data['mean_chars']:.2f}")
        print(
            f"Median chars before 'wait': {analysis_data['median_chars']:.2f}")
else:
    print(
        f"No rows found matching the criteria ({FEATURE_GROUP} features 1-{K} with strength = {STRENGTH_VALUE})")

# %%
# --- Comparative analysis across different strengths and feature groups ---
print(
    f"\nComparative analysis across different strengths and feature groups (1-{K}):")

# Define the feature groups and strengths to analyze
feature_groups = ["bottom", "top"]
strength_values = [-1.5, -1.25, -1.0, 1.0, 1.25, 1.5,]
colors = {"bottom": "blue", "top": "red"}

# Create a figure for the comparison
plt.figure(figsize=(12, 8))

# Store results for plotting
results = []

# Analyze each combination
for feature_group in feature_groups:
    for strength in strength_values:
        filtered_df, analysis = filter_and_analyze_features(
            df, feature_group, strength, K)

        # Get the maximum length of text samples for this subset
        max_text_length = 0
        if 'text_after_wait' in filtered_df.columns:
            text_lengths = filtered_df['text_after_wait'].str.len()
            max_text_length = text_lengths.max() if not text_lengths.empty and not all(
                pd.isna(text_lengths)) else 0
        else:
            # Fallback
            max_text_length = filtered_df['chars_before_wait'].max() * 2

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        filtered_df_copy = filtered_df.copy()

        # Create adjusted column for analysis using proper pandas methods
        filtered_df_copy.loc[:,
                             'chars_until_wait_adjusted'] = filtered_df_copy['chars_before_wait'].copy()
        filtered_df_copy.loc[filtered_df_copy['chars_before_wait']
                             == -1, 'chars_until_wait_adjusted'] = max_text_length

        # Calculate the mean
        mean_chars = filtered_df_copy['chars_until_wait_adjusted'].mean() if len(
            filtered_df_copy) > 0 else 0

        # Store the results
        results.append({
            "feature_group": feature_group,
            "strength": strength,
            "mean_chars": mean_chars,
            "num_rows": len(filtered_df_copy)
        })

        print(f"{feature_group.capitalize()} features 1-{K}, Strength={strength}: Mean chars={mean_chars:.2f}, Rows={len(filtered_df_copy)}")

# Plot the results
bar_width = 0.35
index = np.arange(len(strength_values))

for i, feature_group in enumerate(feature_groups):
    group_results = [r["mean_chars"]
                     for r in results if r["feature_group"] == feature_group]
    color = COLOR_BOTTOM if feature_group == "bottom" else COLOR_TOP
    # Explicitly set edgecolor to 'none' and linewidth to 0 for the bar chart
    plt.bar(index + i*bar_width, group_results, bar_width,
            label=f'{feature_group.capitalize()} Features 1-{K}',
            color=color, alpha=1, edgecolor='none', linewidth=0)

plt.xlabel('Strength Value')
plt.ylabel('Average number of characters until "wait"')
plt.title(
    f'Average number of characters until "wait" by Feature Rank (1-{K}) and Intervention Strength')
plt.xticks(index + bar_width/2, strength_values)
plt.legend()
plt.grid(axis='y', alpha=0.0)
plt.tight_layout()
plt.savefig(f'feature_intervention_wait_analysis_k{K}.pdf', format='pdf')
plt.show()

# %%
