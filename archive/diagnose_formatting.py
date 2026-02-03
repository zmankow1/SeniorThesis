import os


def find_file(filename, search_dirs):
    """
    Tries to find the file in a list of potential directories.
    """
    for directory in search_dirs:
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
    return None


def diagnose_formatting(file_path, label):
    """
    Reads the first 2000 characters of a file and prints a visual representation
    of whitespace characters (newlines, tabs, etc.) to diagnose chunking issues.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"Error: File for '{label}' not found.")
        return

    print(f"\n{'=' * 20} DIAGNOSING: {label} {'=' * 20}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Path: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(2000)

    # Create a visible representation of whitespace
    visible_content = content.replace('\n', '[\\n]\n').replace('\r', '[\\r]')

    print("--- VISUAL WHITESPACE REPRESENTATION (First 2000 chars) ---")
    print(visible_content)
    print("-" * 60)

    # Analysis of line breaks
    line_count = content.count('\n')
    double_newline_count = content.count('\n\n')
    carriage_return_count = content.count('\r')

    print(f"Statistics:")
    print(f"- Total Characters Read: {len(content)}")
    print(f"- Single Newlines (\\n): {line_count}")
    print(f"- Double Newlines (\\n\\n): {double_newline_count}")
    print(f"- Carriage Returns (\\r): {carriage_return_count}")

    if double_newline_count == 0 and line_count > 0:
        print("\n[RESULT] Potential 'Hard Wrap' detected: Newlines exist but no paragraph breaks (\\n\\n).")
    elif line_count == 0:
        print("\n[RESULT] No newlines detected: File may be a single continuous string.")
    else:
        print("\n[RESULT] Formatting appears standard (contains double newlines).")


if __name__ == "__main__":
    # Define potential search directories based on your error log
    search_paths = [
        ".",  # Current directory
        "corpus_txt",  # Subdirectory
        "/Users/zmankowitz2022/PycharmProjects/NLP/corpus_txt",  # Absolute path check
        "../corpus_txt"  # Parent's subdirectory
    ]

    # File names to check
    natural_filename = "TheReturnofTheKing.txt"
    forced_filename = "TheTwoTowers.txt"

    # Locate files
    natural_path = find_file(natural_filename, search_paths)
    forced_path = find_file(forced_filename, search_paths)

    diagnose_formatting(natural_path, "NATURAL CHUNKING")
    diagnose_formatting(forced_path, "FORCED WINDOWING")