import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(script_dir, "..", "data", "results")
IMG_DIR = os.path.join(script_dir, "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")  # Bigger fonts for Thesis


def plot_lexical_decay():
    """Bar chart of Vocabulary Similarity"""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "influence_metrics.csv"))
        plt.figure(figsize=(10, 6))

        # Color palette: Green (Root), Blue (Successor), Red (Modern)
        colors = ['#2e8b57', '#4682b4', '#b22222']

        sns.barplot(x="Corpus", y="Lexical Diffusion Score", data=df, palette=colors)
        plt.title("The Decay of 'Tolkien-Speak' (Lexical Diffusion)")
        plt.ylabel("Vocabulary Similarity Score")
        plt.xlabel("")
        plt.ylim(400, 800)  # Zoom in to show difference

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "1_Lexical_Decay.png"), dpi=300)
        print("✅ Generated Lexical Decay Chart")
    except Exception as e:
        print(f"Skipping Lexical Chart: {e}")


def plot_thematic_shift():
    """Grouped Bar Chart for Topic 3 vs Topic 4"""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "topic_distribution.csv"))

        # We want to compare Topic 3 (Nature/Journey) and Topic 4 (Politics/State)
        # Melt the dataframe for plotting
        plot_df = df.melt(id_vars=["Corpus"], value_vars=["Topic 3 Share", "Topic 4 Share"],
                          var_name="Topic", value_name="Prevalence")

        # Rename for clarity
        plot_df["Topic"] = plot_df["Topic"].replace({
            "Topic 3 Share": "The Elemental Journey (Nature)",
            "Topic 4 Share": "Martial Civilization (Politics)"
        })

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Corpus", y="Prevalence", hue="Topic", data=plot_df, palette="muted")

        plt.title("Thematic Displacement: From Nature to Politics")
        plt.ylabel("Topic Prevalence (%)")
        plt.xlabel("")

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "2_Thematic_Shift.png"), dpi=300)
        print("✅ Generated Thematic Shift Chart")
    except Exception as e:
        print(f"Skipping Thematic Chart: {e}")


def plot_moral_scatter():
    """Scatter Plot: Authority vs. Care(Vice)"""
    try:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "moral_foundations.csv"))

        plt.figure(figsize=(12, 8))

        # Map colors to groups
        group_colors = {
            "Tolkien (Root)": "#2e8b57",  # Green
            "Successors (80s/90s)": "#4682b4",  # Blue
            "Modern (Deconstruction)": "#b22222"  # Red
        }

        sns.scatterplot(
            data=df,
            x="Authority (Virtue)",
            y="Care (Vice)",
            hue="Group",
            palette=group_colors,
            s=300,  # Dot size
            alpha=0.8
        )

        # Label specific characters
        labels_to_show = ["Aragorn", "Frodo", "Ned", "Cersei", "Kaladin", "Jaime", "Rand", "Gandalf"]

        for i, row in df.iterrows():
            if row['Character'].strip() in labels_to_show:
                plt.text(
                    row['Authority (Virtue)'] + 0.5,
                    row['Care (Vice)'],
                    row['Character'],
                    fontsize=12,
                    weight='bold'
                )

        plt.title("The Moral Landscape: Political Power vs. Violence")
        plt.xlabel("Language of Authority (Politics)")
        plt.ylabel("Language of Harm (Grittiness)")

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "3_Moral_Scatter.png"), dpi=300)
        print("✅ Generated Moral Scatter Plot")
    except Exception as e:
        print(f"Skipping Moral Scatter: {e}")


if __name__ == "__main__":
    plot_lexical_decay()
    plot_thematic_shift()
    plot_moral_scatter()
    print(f"\n✨ All images saved to: {IMG_DIR}")