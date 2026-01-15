import os
import re
from pathlib import Path
import csv
import argparse


def parse_arch_summary_files(archsumms_dir):
    """
    Parse all architecture summary text files and extract structured data.
    Returns a list of dicts with columns: run_name, iteration_number, data_representation, model_architecture
    """
    results = []
    archsumms_path = Path(archsumms_dir)
    
    # Find all .txt files (excluding the csv output itself)
    txt_files = sorted(archsumms_path.glob("*.txt"))
    
    for txt_file in txt_files:
        run_name = txt_file.stem
        
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split by "Iteration iteration_X:" pattern
        iteration_pattern = r'Iteration (iteration_\d+):'
        iterations = re.split(iteration_pattern, content)
        
        # Skip the header (Architecture Summary for Run: ...)
        if len(iterations) > 1:
            iterations = iterations[1:]  # Remove the initial split before first iteration
        
        # Process pairs of (iteration_name, iteration_content)
        for i in range(0, len(iterations), 2):
            if i + 1 < len(iterations):
                iteration_name = iterations[i].strip()
                iteration_content = iterations[i + 1]
                
                # Extract iteration number
                match = re.search(r'iteration_(\d+)', iteration_name)
                if match:
                    iteration_number = int(match.group(1))
                else:
                    continue
                
                # Extract data representation
                data_repr_match = re.search(
                    r'Data Representation:\s*(.+?)(?=\n  Model Architecture:|$)',
                    iteration_content,
                    re.DOTALL
                )
                data_repr = data_repr_match.group(1).strip() if data_repr_match else "Missing DataRepresentation or ModelArchitecture information."
                
                # Extract model architecture
                model_arch_match = re.search(
                    r'Model Architecture:\s*(.+?)$',
                    iteration_content,
                    re.DOTALL
                )
                model_arch = model_arch_match.group(1).strip() if model_arch_match else "Missing DataRepresentation or ModelArchitecture information."
                
                # Clean up text (remove extra whitespace, preserve structure)
                data_repr = ' '.join(data_repr.split())
                model_arch = ' '.join(model_arch.split())
                
                results.append({
                    'run_name': run_name,
                    'iteration_number': iteration_number,
                    'data_representation': data_repr,
                    'model_architecture': model_arch,
                })
    
    return results


def save_to_csv(data, output_csv):
    """
    Save structured data to CSV file.
    """
    if not data:
        print(f"Warning: No data to write to {output_csv}")
        return
    
    # Sort by run_name and iteration_number for consistency
    data = sorted(data, key=lambda x: (x['run_name'], x['iteration_number']))
    
    fieldnames = ['run_name', 'iteration_number', 'data_representation', 'model_architecture']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Architecture summary CSV written to {output_csv}")
    print(f"Total entries: {len(data)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert architecture summary text files to CSV'
    )
    parser.add_argument(
        '--archsumms-dir',
        type=str,
        default='./archsumms',
        help='Directory containing architecture summary text files'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Output CSV file path (default: archsumms_dir/architecture_summaries.csv)'
    )
    
    args = parser.parse_args()
    
    if args.output_csv is None:
        args.output_csv = os.path.join(args.archsumms_dir, 'architecture_summaries.csv')
    
    # Parse text files
    data = parse_arch_summary_files(args.archsumms_dir)
    
    # Save to CSV
    save_to_csv(data, args.output_csv)


if __name__ == '__main__':
    main()
