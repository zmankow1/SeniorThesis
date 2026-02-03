import os
import glob

INPUT_DIR = '../data/corpus_txt/'


def audit_files():
    txt_files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))

    print(f"{'File Name':<30} | {'Size (MB)':<10} | {'Chars Read':<12} | {'Start Marker Found?'}")
    print("-" * 80)

    start_markers = ["PROLOGUE", "PRELUDE", "BOOK ONE", "Chapter 1", "Chapter One"]

    for path in txt_files:
        file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        file_name = os.path.basename(path)

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                chars_read = len(content)

                # Check for null bytes that might be stopping the read
                has_null = '\x00' in content

                # Check markers
                found_marker = "No"
                for m in start_markers:
                    if m.lower() in content.lower():
                        found_marker = f"Yes ({m})"
                        break

                null_status = " [!] HAS NULL BYTES" if has_null else ""
                print(f"{file_name[:30]:<30} | {file_size:<10.2f} | {chars_read:<12} | {found_marker}{null_status}")

                if chars_read < 10000 and file_size > 0.1:
                    print(
                        f"   ⚠️ WARNING: File is {file_size:.2f}MB but only read {chars_read} characters. Check encoding!")

        except Exception as e:
            print(f"{file_name[:30]:<30} | ERROR: {str(e)}")


if __name__ == "__main__":
    audit_files()