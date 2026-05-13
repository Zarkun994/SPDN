# Phenology Distillation (Public Framework)

This repository provides a **clean public framework** for a two-stage winter wheat phenology model:

1. **Stage 1** predicts sowing date and season length from calendar-window features.
2. **Stage 2** performs teacher-student distillation on ANT-style phase bins built from the estimated season parameters.

The code is intentionally organized as a **public-facing skeleton**. It keeps the main model structure and training logic, while removing private local paths, project-specific file names, and unnecessary experimental traces.

## Highlights

- Calendar-based temporal encoder for season parameter regression
- ANT-based temporal encoder for phenology-stage prediction
- Teacher autoregressive decoder + student parallel decoder
- Zero-leak evaluation pipeline
- Optional photoperiod and cumulative GDD features
- Config-driven file paths and runtime settings

## Repository layout

```text
pheno_public_framework/
├── configs/
│   └── config.example.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── engine/
│   │   ├── evaluate.py
│   │   ├── stage1.py
│   │   └── stage2.py
│   ├── models/
│   │   └── modules.py
│   └── utils/
│       ├── io.py
│       └── reproducibility.py
├── requirements.txt
└── run.py
```

## Expected data format

You should adapt your private data to the following generic interface.

### Label table
Required columns:

- `Lat`
- `Lon`
- `Year`
- stage DOY columns such as `Sowing_DOY`, `Emergence_DOY`, ...

### ERA table(s)
Required columns:

- a date column, e.g. `date`
- `Lat`, `Lon`
- meteorological variables such as:
  - `ET_mm`
  - `GDD_sum`
  - `Rad_MJ`
  - `soil_water`

### VI table(s)
Required columns:

- a date column, e.g. `date`
- either `Lat` and `Lon`, or a `point_id` that can be linked externally
- vegetation-index variables such as:
  - `NDVI`
  - `NDWI`

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the example config and edit it:

```bash
cp configs/config.example.yaml configs/config.yaml
```

3. Run training and evaluation:

```bash
python run.py --config configs/config.yaml
```

## Notes for a public GitHub release

This version is designed for sharing with reviewers/readers. Before publishing:

- keep only generic paths in the YAML config
- do not upload private raw data
- do not include internal directory names or local machine paths
- keep README focused on method and input format

## Citation / reproducibility note

If the original dataset is restricted, you can add a sentence such as:

> The original data contain restricted research assets and are therefore not publicly released. This repository provides the core implementation and a template for reproducing the workflow on similarly structured data.
