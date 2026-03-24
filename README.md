# SeniorThesis Workspace

Active work is now consolidated in:

- `secondary_world_thesis/`

Start here:

- `secondary_world_thesis/README.md`
- `secondary_world_thesis/PROJECT_MAP.md`
- `secondary_world_thesis/handoffs/SHAREABLE_PROJECT_UPDATE_2026-03-23.md`

## Current Canonical Layout

```text
SeniorThesis/
├── secondary_world_thesis/
│   ├── data/
│   │   ├── corpus_txt/
│   │   ├── raw_epubs/
│   │   └── results/
│   ├── src/
│   ├── scripts/
│   ├── outputs_13books_fallback/
│   ├── outputs_13books_spacy/
│   ├── outputs_compare_13books_spacy_vs_fallback/
│   ├── figures_13books_fallback/
│   ├── figures_13books_spacy/
│   ├── figures_13books_extended/
│   └── handoffs/
└── archive/
    ├── 2026-03-05_focus_cleanup/
    └── 2026-03-23_project_cleanup/
```

## Quick Run

```bash
cd secondary_world_thesis
python3 run.py --spacy-model en_core_web_lg --spacy-batch-size 1 --spacy-chunk-size 80000 --chapter-world-mode fallback_only --output-dir outputs_13books_spacy
python3 validate_outputs.py --output-dir outputs_13books_spacy
```
