import pandas as pd
from itertools import combinations
import os


def export_gold_to_neo4j():
    """
    Reads ai_gold_labels.csv and prepares node/relationship files for Neo4j.
    Utilizes the AI-generated labels (CHARACTER, LOCATION, FACTION, ARTIFACT).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, "..", "data", "processed_data"))

    input_file = os.path.join(base_path, "ai_gold_labels.csv")
    output_nodes = os.path.join(base_path, "gold_nodes.csv")
    output_rels = os.path.join(base_path, "gold_relationships.csv")

    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find {input_file}. Run apply_fantasy_ner.py first!")
        return

    print("Loading Gold Standard AI labels...")
    df = pd.read_csv(input_file)

    node_data = {}  # Using dict to handle entity resolution/labels
    edge_data = []

    print("Processing entities and co-occurrences...")
    for _, row in df.iterrows():
        book = row['book_id']
        labeled_str = str(row.get('labeled_entities', ''))

        if not labeled_str or labeled_str == 'nan':
            continue

        # Split the labeled entities: "Name|LABEL,Name|LABEL"
        raw_pairs = [p.split('|') for p in labeled_str.split(',') if '|' in p]

        # Unique entities in this specific chunk
        chunk_ents = []
        for name, label in raw_pairs:
            name = name.strip()
            if not name or len(name) < 2: continue

            # Store node info (keeping the most frequent label if there's a conflict)
            if name not in node_data:
                node_data[name] = label
            chunk_ents.append(name)

        unique_chunk_ents = list(set(chunk_ents))

        # Create edges for everything appearing together in this ~7-sentence window
        if len(unique_chunk_ents) > 1:
            for pair in combinations(unique_chunk_ents, 2):
                s, t = sorted(pair)
                edge_data.append({"source": s, "target": t, "book": book})

    # 1. SAVE NODES
    nodes_list = [{"id": name, "name": name, "label": label} for name, label in node_data.items()]
    pd.DataFrame(nodes_list).to_csv(output_nodes, index=False)

    # 2. SAVE RELATIONSHIPS
    edges_df = pd.DataFrame(edge_data)
    if not edges_df.empty:
        # Aggregate weight: how many chunks do these two entities appear in together?
        edges_df = edges_df.groupby(['source', 'target', 'book']).size().reset_index(name='weight')
    edges_df.to_csv(output_rels, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"Nodes: {len(nodes_list)} | Relationships: {len(edges_df)}")
    print(f"Import files saved to: {base_path}")


if __name__ == "__main__":
    export_gold_to_neo4j()