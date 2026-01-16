# Data Filtering Guide

## Overview

JADE-DLS includes interactive filtering dialogs that allow you to visually inspect and exclude bad measurements. This ensures data quality before analysis.

## Workflow

After loading data files, two filtering steps are automatically presented:

1. **Count Rate Filtering** - Inspect detector count rates
2. **Correlation Filtering** - Inspect correlation functions

## Count Rate Filtering

### What to Look For

**Good Count Rates:**
- Stable over time
- No sudden jumps or drops
- All detector slots show similar trends
- Smooth lines without noise spikes

**Bad Count Rates (exclude these):**
- Large fluctuations
- Sudden drops or spikes
- Detector slots showing very different behavior
- Noisy or erratic patterns

### How to Filter

1. **View Individual Files:**
   - Click on a filename in the left panel
   - Inspect the count rate plot on the right
   - Look for anomalies

2. **View All Files at Once:**
   - Click "📊 Show All Plots"
   - Get overview of all measurements
   - Identify outliers quickly

3. **Mark Bad Files:**
   - Select file(s) in the list (Ctrl+Click for multiple)
   - Click "❌ Mark Selected as Bad"
   - File turns red and is marked for exclusion

4. **Unmark Files:**
   - Select marked file(s)
   - Click "✓ Unmark Selected"

5. **Apply Filter:**
   - Click "Apply Filter" to proceed
   - Statistics show how many files excluded

### Example

```
Count Rate Plot showing:
- Time [s] on x-axis
- Count Rate [kHz] on y-axis
- 4 detector slots plotted

Good measurement:
  All 4 lines smooth, parallel, stable

Bad measurement:
  Sudden spike at t=50s
  Detector 3 drops to zero
  → EXCLUDE THIS FILE
```

## Correlation Filtering

### What to Look For

**Good Correlations (g²-1):**
- Smooth exponential decay
- Starts near 1.0 at short times
- Decays smoothly to baseline (near 0)
- No negative values or oscillations
- Logarithmic x-axis shows smooth curve

**Bad Correlations (exclude these):**
- Noisy data
- Non-exponential decay
- Baseline not at zero
- Negative values
- Artifacts or discontinuities

### How to Filter

Same workflow as Count Rate Filtering:

1. View individual or all plots
2. Mark bad files
3. Apply filter

### Example

```
Correlation Plot showing:
- Time [ms] on log scale (x-axis)
- g² - 1 on linear scale (y-axis)

Good measurement:
  Smooth decay from ~0.8 to 0
  No oscillations
  Reaches baseline

Bad measurement:
  Noisy data with scatter
  Doesn't reach baseline
  Negative values
  → EXCLUDE THIS FILE
```

## Tips

### Efficient Filtering

1. **Use "Show All Plots" First:**
   - Get overview of all data
   - Quickly identify obvious outliers
   - Red background = already excluded

2. **Multi-Select:**
   - Hold Ctrl and click multiple files
   - Mark/unmark several at once

3. **Conservative Approach:**
   - When in doubt, exclude
   - Better to have less data that's high quality
   - Can always re-run without excluding

### Common Issues

**All count rates look different?**
- Check if detector alignment changed
- Consider excluding all from that session
- Or re-calibrate detector settings

**Correlation baseline not at zero?**
- Usually indicates problems
- Check if normalization is correct
- Consider excluding

**Only a few files are good?**
- Check experimental conditions
- Temperature stability?
- Sample preparation?
- May need to repeat experiment

## Quality Metrics

### Count Rates

**Acceptable:**
- Stability: < 5% variation
- All detectors within 20% of each other
- No spikes > 2× average

**Reject:**
- Variation > 10%
- Detectors differ by > 50%
- Any sudden jumps

### Correlations

**Acceptable:**
- R² > 0.99 after fitting
- Smooth decay
- Baseline within ±0.05 of zero

**Reject:**
- Noisy data (visual inspection)
- Baseline > 0.1
- Non-physical behavior

## Status Bar Messages

During filtering, the status bar shows:

```
[14:35:30] Successfully loaded 39 files (completed in 8.2s)
[14:35:31] Filtering count rates
[14:35:45] Excluded 3 files based on count rates
[14:35:46] Filtering correlations
[14:36:02] Excluded 2 files based on correlations
[14:36:03] Ready
```

## Final Summary

After filtering, you'll see a summary:

```
┌─ Data Ready ──────────────────────┐
│                                    │
│ Data loading and filtering         │
│ complete!                          │
│                                    │
│ Original files: 39                 │
│ Excluded: 5                        │
│ Final dataset: 34 files            │
│                                    │
│ Base data: 34 entries              │
│ Correlations: 34 datasets          │
│ Count rates: 34 datasets           │
│                                    │
│              [ OK ]                │
└────────────────────────────────────┘
```

## Skipping Filtering

- **Count Rate Dialog:** Click "Cancel" to use all data
- **Correlation Dialog:** Click "Cancel" to use all data
- Status bar will note: "Filtering cancelled - using all data"

## Best Practices

1. **Always Filter:**
   - Even if you think data is good
   - Quick visual check catches issues
   - Takes only 1-2 minutes

2. **Document Exclusions:**
   - Note why files were excluded
   - Keep lab notebook entry
   - Helps with reproducibility

3. **Consistent Criteria:**
   - Use same criteria for all experiments
   - Be systematic, not arbitrary
   - Train eye to recognize bad data

4. **Re-Filter if Needed:**
   - Can always reload data
   - Try different exclusion criteria
   - Compare results

## Scientific Rationale

### Why Filter?

Bad data points can:
- Skew average values
- Increase error bars artificially
- Lead to incorrect size distributions
- Cause poor fit quality
- Result in non-physical parameters

### Impact on Results

Example with 39 files:
- Without filtering: Rh = 125 ± 45 nm, R² = 0.85
- With filtering (5 bad files excluded): Rh = 112 ± 8 nm, R² = 0.997

**Conclusion:** Filtering improves both accuracy and precision!

### Transparency

JADE-DLS shows you:
- All data before filtering
- Which files you excluded
- Why (via visual inspection)
- Final dataset composition

This ensures **scientific transparency** and **reproducibility**.

---

**Remember:** Good analysis starts with good data. Take time to filter properly! 🔬
