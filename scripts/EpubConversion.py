import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import os
import glob


def convert_epub_to_text(epub_path):
    """
    Reads an EPUB file, extracts the content from each chapter,
    strips HTML, and returns the combined clean text.
    """
    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        print(f"Error reading EPUB {epub_path}: {e}")
        return ""

    content = []

    for item in book.get_items():
        # Only process content items (chapters, sections)
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Decode the HTML content
            html_content = item.get_content().decode('utf-8')

            # Use BeautifulSoup to parse and strip HTML tags
            soup = BeautifulSoup(html_content, 'html.parser')

            # Get text and clean up whitespace/newlines
            text = soup.get_text()

            # Use regex to replace multiple newlines/spaces with a single space
            clean_text = re.sub(r'\s{2,}', ' ', text).strip()

            # Only append if we have meaningful text
            if clean_text:
                content.append(clean_text)

    # Join all chapter contents with a single marker (e.g., two newlines)
    full_text = '\n\n'.join(content)

    return full_text


def batch_convert_epubs(input_dir, output_dir):
    """
    Finds all EPUB files in the input directory, converts them to text,
    and saves the resulting TXT files in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Use glob to find all files ending in '.epub' recursively (**)
    epub_files = glob.glob(os.path.join(input_dir, '*.epub'), recursive=False)

    if not epub_files:
        print(f"No EPUB files found in {input_dir}. Check your path.")
        return []

    converted_files = []

    print(f"Found {len(epub_files)} EPUBs to convert...")

    for epub_path in epub_files:
        novel_text = convert_epub_to_text(epub_path)

        # 1. Create a safe output filename
        base_filename = os.path.basename(epub_path)
        txt_filename = base_filename.replace('.epub', '.txt')
        output_path = os.path.join(output_dir, txt_filename)

        # 2. Save the converted text
        if novel_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(novel_text)
            print(f"  -> Successfully converted and saved: {txt_filename}")
            converted_files.append(output_path)
        else:
            print(f"  -> Conversion failed or text was empty for: {base_filename}")

    return converted_files


# --- Execution Block ---
if __name__ == '__main__':
    INPUT_DIR = '../data/raw_epubs/'  # Folder where your 13 .epub files are located
    OUTPUT_DIR = '../data/corpus_txt/'  # Folder where the converted .txt files will be saved

    # Run the batch conversion
    txt_file_list = batch_convert_epubs(INPUT_DIR, OUTPUT_DIR)

    # This list (txt_file_list) is what you will pass to the load_corpus function in CleanText.py
    print(f"\nTotal files ready for processing: {len(txt_file_list)}")