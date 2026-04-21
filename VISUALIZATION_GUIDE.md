# Dataset & Results Visualization Guide

## Overview

Two visualization scripts have been created to analyze the HuMob dataset and model results:

1. **`visualize_dataset.py`** — Dataset representation and analysis
2. **`visualize_results.py`** — Model performance comparison and GEO-BLEU analysis

---

## visualize_dataset.py

Generates comprehensive visualizations of the HuMob dataset characteristics.

### Usage

```bash
python visualize_dataset.py --cities A B C D --output_dir dataset_analysis
```

### Generated Plots

| Plot | Description |
|------|-------------|
| **(a) data_completeness.png** | Bar chart of data completeness (%) per city |
| **(b) dataset_seasonality.png** | 4-panel plot showing daily movement count seasonality for each city with training/test period markers |
| **(c) dataset_mobility_reduction.png** | Comparison of average daily movement count: training period (pre-emergency) vs test period (emergency) |
| **(d) dataset_user_distribution.png** | Bar chart of number of users per city |
| **(e) dataset_records_per_user.png** | Violin plot showing distribution of records per user for each city |
| **(f) dataset_temporal_coverage.png** | 4-panel plot of temporal data completeness (%) over all 75 days |
| **dataset_overview_dashboard.png** | Combined 3×3 dashboard view of all statistics |
| **dataset_summary.csv** | Summary statistics table (users, records, completeness, etc.) |

### Key Statistics Generated

For each city:
- Total users
- Total records (overall, training, test)
- Average records per user
- Data completeness rate (%)
- Daily movement count seasonality
- Temporal coverage over 75 days

### Example Output

```
════════════════════════════════════════════════════════════════════════════════════════════════
DATASET SUMMARY STATISTICS
════════════════════════════════════════════════════════════════════════════════════════════════
 City     Users   Total Records   Train Records    Test Records  Avg Records/User  Data Completeness
    A    150,000      87,042,418       67,862,502      19,179,916             580.3             92.5%
    B     15,000       8,200,000        6,500,000       1,700,000             546.7             88.3%
    C     12,500       6,850,000        5,400,000       1,450,000             548.0             85.6%
    D     10,000       5,400,000        4,200,000       1,200,000             540.0             82.1%
════════════════════════════════════════════════════════════════════════════════════════════════
```

### Interpretation Guide

**Seasonality (Plot b):**
- Shows daily movement count over 75 days
- Red dashed line marks training/test boundary (day 60→61)
- Green shaded region: training period (days 1-60)
- Orange shaded region: test period (days 61-75)
- Patterns visible: weekly cycles, emergency-period reduction

**Mobility Reduction (Plot c):**
- Green bars: normal mobility period (training)
- Red bars: emergency period (test)
- Typically shows 20-40% reduction in movement during test period
- Indicates real-world event impact on mobility

**Data Completeness (Plot a):**
- Percentage of users who have data on a given day
- Higher values = more reliable dataset
- Typically 80-95% for HuMob challenge data

---

## visualize_results.py

Generates visualizations comparing model predictions and GEO-BLEU scores.

### Usage

```bash
python visualize_results.py \
  --lpbert_dir predictions \
  --geoformer_dir predictions \
  --cities A B C D \
  --output_dir results_analysis
```

### Required Data

The script expects GEO-BLEU score CSV files with structure:

```csv
uid,geobleu
12345,0.0234
12346,0.0156
...
```

Naming convention:
- `{output_dir}/geobleu_scores_city{A|B|C|D}.csv`

### Generated Plots

| Plot | Description |
|------|-------------|
| **geobleu_comparison_cityX.png** | 2-panel: (left) bar chart of mean GEO-BLEU with error bars, (right) box plot of score distribution |
| **geobleu_distribution_cityX.png** | 2-panel: (left) CDF of scores, (right) histogram of score distribution |
| **geobleu_heatmap.png** | Heatmap showing mean GEO-BLEU for each model×city combination |
| **geobleu_summary.csv** | Summary table with mean GEO-BLEU for all models and cities |

### Example Output

```
════════════════════════════════════════════════════════════════════════════════════════════════
GEO-BLEU SCORE SUMMARY
════════════════════════════════════════════════════════════════════════════════════════════════
        Model  City A      City B      City C      City D
      LP-BERT  0.006296    0.084300    0.065900    0.034800
     GeoFormer 0.062957    0.073864    0.065303    0.074077
      MoE+BERT 0.162200    0.084300    0.065900    0.034800
════════════════════════════════════════════════════════════════════════════════════════════════
```

### Interpretation Guide

**GEO-BLEU Comparison (bar chart):**
- Mean score with error bars (±1 std dev)
- Higher is better
- Error bars show prediction confidence/consistency

**Distribution (box plot):**
- Shows median, quartiles, outliers
- Wider boxes = more variable predictions
- Outliers = users with very good/poor predictions

**CDF (cumulative distribution function):**
- X-axis: GEO-BLEU score threshold
- Y-axis: fraction of users achieving at least that score
- Steeper curves = better overall performance

**Score Range Interpretation:**
- 0.0-0.01: Very poor predictions (random or baseline)
- 0.01-0.05: Poor predictions (location often wrong)
- 0.05-0.10: Moderate predictions (some correct locations)
- 0.10+: Good predictions (often within short distance)

---

## Command-Line Options

### visualize_dataset.py

```
--cities A B C D       Cities to analyze (default: all)
--data_dir DATA        Path to data folder (default: Data)
--output_dir DIR       Output directory for plots (default: dataset_analysis)
```

### visualize_results.py

```
--lpbert_dir DIR       Directory with LP-BERT results (default: predictions)
--geoformer_dir DIR    Directory with GeoFormer results (default: predictions)
--cities A B C D        Cities to visualize (default: all)
--output_dir DIR       Output directory for plots (default: results_analysis)
```

---

## Integration with Existing Code

### After Training LP-BERT:

```bash
# Train
python run_lpbert.py train --city A --epochs 200

# Generate predictions
python run_lpbert.py generate --checkpoint checkpoints/lpbert_cityA_paper_best.pt --city A

# Evaluate
python run_lpbert.py evaluate --predictions predictions/lpbert_predictions_cityA.csv --city A

# Visualize dataset
python visualize_dataset.py --cities A --output_dir analysis/dataset

# Visualize results
python visualize_results.py --lpbert_dir predictions --cities A --output_dir analysis/results
```

### For Full Pipeline:

```bash
# Run visualizations for all cities after all models are trained
python visualize_dataset.py --cities A B C D --output_dir analysis/dataset
python visualize_results.py --lpbert_dir predictions --cities A B C D --output_dir analysis/results
```

---

## Output Directory Structure

```
dataset_analysis/
├── dataset_completeness.png
├── dataset_seasonality.png
├── dataset_mobility_reduction.png
├── dataset_user_distribution.png
├── dataset_records_per_user.png
├── dataset_temporal_coverage.png
├── dataset_overview_dashboard.png
└── dataset_summary.csv

results_analysis/
├── geobleu_comparison_cityA.png
├── geobleu_comparison_cityB.png
├── geobleu_comparison_cityC.png
├── geobleu_comparison_cityD.png
├── geobleu_distribution_cityA.png
├── geobleu_distribution_cityB.png
├── geobleu_distribution_cityC.png
├── geobleu_distribution_cityD.png
├── geobleu_heatmap.png
└── geobleu_summary.csv
```

---

## Customization

### Modify Colors

In both scripts, change the color palettes:

```python
# In visualize_dataset.py
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # Red, Blue, Green, Orange

# Replace with your preferred palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Alternative
```

### Adjust Plot Sizes

```python
# In visualize_dataset.py
fig, ax = plt.subplots(figsize=(10, 6))  # width, height in inches

# Make larger
fig, ax = plt.subplots(figsize=(14, 8))
```

### Change DPI (resolution)

```python
# Default
plt.savefig(output_dir / "plot.png", dpi=300)

# Higher quality (larger file)
plt.savefig(output_dir / "plot.png", dpi=600)

# Lower quality (smaller file)
plt.savefig(output_dir / "plot.png", dpi=150)
```

---

## Troubleshooting

### "File not found" error

**Solution:** Ensure CSV files are in the correct directory and follow the naming convention:
- `{output_dir}/geobleu_scores_city{A|B|C|D}.csv`

### Empty plots

**Possible causes:**
1. Data not loaded correctly → Check file path and format
2. Column names mismatch → Verify CSV has `uid` and `geobleu` columns
3. No data for specified cities → Check available cities in data directory

**Solution:** Run with debug output:
```bash
python visualize_dataset.py --cities A --output_dir test 2>&1 | tee debug.log
```

### Plots not showing

If using remote system, save plots to files (already done by default):
```bash
# Transfer files locally
scp -r user@remote:~/analysis ./local_analysis
```

---

## Dependencies

Required packages (already in requirements.txt):
- `pandas`: Data manipulation
- `numpy`: Numerical computation
- `matplotlib`: Plotting
- `tqdm`: Progress bars

Installation:
```bash
pip install -r requirements.txt
```

---

## Examples

### Quick dataset overview (single city):

```bash
python visualize_dataset.py --cities A --output_dir quick_analysis
```

### Full analysis (all cities):

```bash
python visualize_dataset.py --cities A B C D --output_dir full_analysis
```

### Compare models across cities:

```bash
python visualize_results.py --lpbert_dir predictions --geoformer_dir predictions --cities A B C D --output_dir comparison
```

---

## References

- **GEO-BLEU metric**: Geospatial BLEU score for trajectory evaluation
- **HuMob Challenge 2023**: Human mobility prediction task
- **Seasonality patterns**: Weekly and event-driven mobility cycles
- **Data completeness**: Coverage rate indicating data quality

---

**Last updated:** 2026-04-21
**Author:** Dataset Visualization Suite
