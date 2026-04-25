# Hugging Face Model License Index

[![Update Index](https://github.com/YOUR_USER/open-model-license-index/actions/workflows/update.yml/badge.svg)](https://github.com/YOUR_USER/open-model-license-index/actions/workflows/update.yml)

> **Important:** this project does **not** classify, score, or interpret licenses.
> It only mirrors the license identifier/name/link exposed by Open model
> metadata, tags, and repository license files. Always review the linked original
> source before using a model.

## What is this?

A daily-updated index of license metadata for popular Open models.

- **Data sources**: Hugging Face Hub API — model card metadata, hub tags, repository files
- **Update frequency**: Daily via GitHub Actions
- **GitHub Pages**: Browse the interactive table at your Pages URL

## Table

<!-- MODEL_LICENSE_TABLE:START -->

> 0 models tracked · updated 2026-04-25 13:26 UTC

| Model | Org | License |
|---|---|---|
<!-- MODEL_LICENSE_TABLE:END -->

## Files

| File | Description |
|---|---|
| `data/models.json` | Full JSON with license evidence |
| `data/models.csv` | Flat CSV for spreadsheets |
| `public/models.json` | JSON served by GitHub Pages |
| `public/index.html` | Interactive browse UI |

## Configuration

Edit `config/sources.yml` to add or remove tracked organizations.

## License

This project is licensed under the [MIT License](LICENSE).
