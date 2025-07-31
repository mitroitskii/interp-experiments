import json
import re
from collections import defaultdict

def extract_sections(text):
    """Extract sections from the annotated thinking text."""
    # Pattern to match sections like ["section-name"] content ["end-section"]
    pattern = r'\[\"([^\"]+)\"\]\s*(.*?)\s*\[\"end-section\"\]'
    sections = re.findall(pattern, text, re.DOTALL)
    return sections

def analyze_sections(json_file_path):
    """Analyze sections from the JSON file."""
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Counters for each section type
    section_counts = defaultdict(int)
    sections_with_wait = defaultdict(int)
    
    print(f"Processing {len(data)} entries...")
    
    # Process each string in the JSON array
    for i, text in enumerate(data):
        if i % 100 == 0:  # Progress indicator
            print(f"Processed {i} entries...")
            
        # Extract sections from this text
        sections = extract_sections(text)
        
        for section_type, content in sections:
            # Count this section type
            section_counts[section_type] += 1
            
            # Check if content contains 'wait' (case insensitive)
            if 'wait' in content.lower():
                sections_with_wait[section_type] += 1
    
    print(f"\nFinished processing {len(data)} entries.")
    print(f"Found {len(section_counts)} different section types.")
    
    # Calculate results
    print("\n" + "="*80)
    print("SECTION ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n{'Section Type':<25} {'Count':<10} {'With Wait':<12} {'Wait %':<10}")
    print("-" * 65)
    
    # Sort by count (descending)
    sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
    
    for section_type, count in sorted_sections:
        wait_count = sections_with_wait[section_type]
        wait_percentage = (wait_count / count * 100) if count > 0 else 0
        
        print(f"{section_type:<25} {count:<10} {wait_count:<12} {wait_percentage:.1f}%")
    
    # Summary statistics
    total_sections = sum(section_counts.values())
    total_with_wait = sum(sections_with_wait.values())
    overall_wait_percentage = (total_with_wait / total_sections * 100) if total_sections > 0 else 0
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total sections found: {total_sections}")
    print(f"Total sections containing 'wait': {total_with_wait}")
    print(f"Overall percentage with 'wait': {overall_wait_percentage:.1f}%")
    print(f"Number of unique section types: {len(section_counts)}")
    
    return section_counts, sections_with_wait

if __name__ == "__main__":
    json_file_path = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/data/responses_deepseek-r1-distill-llama-8b_-extracted_annotated_thinking.json"
    
    try:
        section_counts, sections_with_wait = analyze_sections(json_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find file {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error: {e}") 