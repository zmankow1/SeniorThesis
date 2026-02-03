import pandas as pd
from itertools import combinations
import os
import re


def clean_fantasy_entity(ent):
    """Strips dialogue artifacts and noise while preserving valid names."""
    if not ent or not isinstance(ent, str): return None
    # Strip non-alphanumeric from ends (catches me?”Bran -> Bran)
    ent = re.sub(r"^[^\w]+|[^\w]+$", "", ent).strip()

    # Noise that should just be deleted
    garbage = {"yes", "no", "lied", "said", "looked", "toward", "find", "and", "the"}
    if ent.lower() in garbage or len(ent) < 3: return None

    # Punctuation check: If it still has a quote or question mark inside, it's noise
    if any(char in ent for char in ['?', '!', '”', '“', '—']): return None

    return ent


def export_to_neo4j():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, "..", "data", "processed_data"))

    input_manual = os.path.join(base_path, "manual_labels.csv")
    input_auto = os.path.join(base_path, "automated_labels.csv")
    output_nodes = os.path.join(base_path, "neo4j_nodes.csv")
    output_rels = os.path.join(base_path, "neo4j_relationships.csv")

    print("Loading data for high-fidelity export...")
    df_man = pd.read_csv(input_manual)
    df_auto = pd.read_csv(input_auto)
    df = pd.merge(df_man, df_auto[['book_id', 'chunk_id', 'key_entities']], on=['book_id', 'chunk_id'], how='left')

    # --- KNOWLEDGE BASES ---

    # These are ALWAYS Characters
    titles = {"Lord", "Lady", "Maester", "Highprince", "Brightlord", "Brightlady", "King", "Queen", "Prince",
              "Princess", "Ser", "Khal", "Septa", "Warden"}
    # Known people who often get mislabeled as places
    known_people = {"Kaladin", "Dalinar", "Shallan", "Adolin", "Sansa", "Arya", "Tyrion", "Perrin", "Rand", "Mat",
                    "Moiraine", "Egwen", "Ned", "Jon"}

    # These are ALWAYS Locations
    geography_suffixes = ["land", "shire", "ford", "port", "field", "keep", "tower", "city", "valley", "mount", "river",
                          "sea", "lake", "rock", "plain", "castle", "bridge", "peak"]
    known_locs = {
        "Winterfell", "King's Landing", "Castle Black", "Eyrie", "Riverrun", "Pyke", "Highgarden", "Dorne", "Braavos",
        "Pentos", "Meereen", "Qarth", "The Wall",
        "Shattered Plains", "Kharbranth", "Kholinar", "Hearthstone", "Urithiru", "Shadesmar", "Thaylen City", "Bavland",
        "Emond's Field", "Two Rivers", "Baerlon", "Caemlyn", "Tar Valon", "Shienar", "Fal Dara",
        "Shire", "Rivendell", "Moria", "Isengard", "Mordor", "Gondor", "Rohan", "Buckkeep", "Six Duchies"
    }

    # These are ALWAYS Factions
    factions = {"House", "Bridge Four", "Watch", "Clan", "Order", "Sedai", "Kingsguard", "Radiants", "Stark",
                "Lannister", "Targaryen"}

    node_data = []
    edge_data = []
    seen_nodes = set()

    for _, row in df.iterrows():
        book = row['book_id']
        combined = str(row.get('key_entities', '')).split(',') + str(row.get('manual_entities', '')).split(',')

        chunk_ents = []
        for e in combined:
            cleaned = clean_fantasy_entity(e)
            if cleaned: chunk_ents.append(cleaned)

        unique_ents = list(set(chunk_ents))

        for ent in unique_ents:
            if ent not in seen_nodes:
                ent_low = ent.lower()
                ent_words = ent.split()

                # CLASSIFICATION PRIORITY ENGINE
                label = "Character"  # Default

                # 1. Check Factions
                if any(f.lower() in ent_low for f in factions):
                    label = "Faction"
                # 2. Check Locations (Whitelist or Suffixes)
                elif ent in known_locs or any(ent_low.endswith(s) or f" {s}" in ent_low for s in geography_suffixes):
                    label = "Location"
                # 3. Check for obvious Titles (force to Character)
                elif any(t in ent for t in titles) or ent in known_people:
                    label = "Character"
                # 4. Fallback: If it's a single word and not capitalized, it's probably noise/location junk
                elif not ent[0].isupper():
                    continue  # Skip noise like 'bush' or 'ironwood'

                node_data.append({"id": ent, "name": ent, "label": label})
                seen_nodes.add(ent)

        if len(unique_ents) > 1:
            for pair in combinations(unique_ents, 2):
                s, t = sorted(pair)
                edge_data.append({"source": s, "target": t, "book": book})

    # Save outputs
    pd.DataFrame(node_data).to_csv(output_nodes, index=False)
    edges_df = pd.DataFrame(edge_data).groupby(['source', 'target', 'book']).size().reset_index(name='weight')
    edges_df.to_csv(output_rels, index=False)

    print(f"✅ Export Successful. Unique Nodes: {len(node_data)}")
    print(f"Files saved to: {os.path.abspath(base_path)}")


if __name__ == "__main__":
    export_to_neo4j()