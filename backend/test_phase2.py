import os
import sys
from pathlib import Path

# Add the backend directory to the path if running from project root
backend_path = Path(__file__).parent
if backend_path.name != 'backend':
    backend_path = backend_path / 'backend'
sys.path.insert(0, str(backend_path))

from utils.text_cleaner import clean_ocr_text
from utils.section_extractor import extract_legal_sections


def main():
    """Test the text cleaning and section extraction pipeline."""
    
    # Define the input file path
    input_file = "sample_ocr_output.txt"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        print(f"Current directory: {os.getcwd()}")
        print("Please ensure the sample_ocr_output.txt file exists in the current directory.")
        sys.exit(1)
    
    try:
        # Read the raw OCR text
        with open(input_file, "r", encoding="utf-8") as f:
            raw = f.read()
        
        if not raw.strip():
            print("Warning: Input file is empty.")
            return
        
        print(f"Raw text length: {len(raw)} characters\n")
        
        # Clean the OCR text
        cleaned = clean_ocr_text(raw)
        print(f"Cleaned text length: {len(cleaned)} characters\n")
        
        # Extract legal sections
        sections = extract_legal_sections(cleaned)
        print(f"Total Sections Found: {len(sections)}\n")
        
        if not sections:
            print("No sections found. Check if the text contains 'Sec. X-X' patterns.")
            print("\nFirst 500 characters of cleaned text:")
            print("-" * 50)
            print(cleaned[:500])
            return
        
        # Display first 3 sections (or fewer if less than 3 exist)
        num_to_display = min(3, len(sections))
        print(f"Displaying first {num_to_display} section(s):\n")
        
        for i, s in enumerate(sections[:num_to_display], 1):
            print(f"Section {i}:")
            print(f"Number: {s['section_number']}")
            print(f"Title: {s['title']}")
            print("-" * 50)
            
            # Handle content display safely
            content = s['content']
            if len(content) > 500:
                print(f"{content[:500]}...")
            else:
                print(content)
            
            print("\n" + "=" * 70 + "\n")
        
        # Optional: Save cleaned text and extracted sections
        save_output = input("Save cleaned text and sections to files? (y/n): ").strip().lower()
        if save_output == 'y':
            # Save cleaned text
            with open("cleaned_output.txt", "w", encoding="utf-8") as f:
                f.write(cleaned)
            print("✓ Saved cleaned text to: cleaned_output.txt")
            
            # Save sections summary
            with open("sections_summary.txt", "w", encoding="utf-8") as f:
                for i, s in enumerate(sections, 1):
                    f.write(f"Section {i}: {s['section_number']}\n")
                    f.write(f"{s['title']}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{s['content']}\n\n")
                    f.write("=" * 70 + "\n\n")
            print("✓ Saved sections to: sections_summary.txt")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()