├── data/
│   ├── corpus_txt/          # Cleaned text files of the novels
│   ├── processed_data/      # Neo4j nodes, relationships, and CSVs
│   └── results/             # Final statistical output (CSVs)
├── scripts/
│   ├── analyze_influence.py # MAIN SCRIPT: Runs LDA, TF-IDF, and Semantic Shift
│   ├── gemini_labeler.py    # Uses Google Gemini API for high-fidelity NER
│   ├── train_fantasy_ner.py # Trains custom spaCy models
│   ├── export_neo4j.py      # Exports social graphs to Neo4j
│   └── CleanText.py         # Pre-processing pipeline
├── .env                     # API Keys (Not included in repo)
└── README.md
