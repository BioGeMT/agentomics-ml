import re
import argparse
from pathlib import Path

def parse_structured_output(content):
    """
    Parse structured output string into a list of dictionaries.
    Each dictionary has the class name as key and the object attributes as value.
    """
    result = []

    # Pattern to match ClassName(attr1=value1, attr2=value2, ...)
    # This handles nested parentheses and quoted strings
    pattern = r'(\w+)\(((?:[^()]|\([^()]*\))*)\)'

    matches = re.finditer(pattern, content)

    for match in matches:
        class_name = match.group(1)
        args_str = match.group(2)

        # Parse the keyword arguments
        attributes = {}

        # Pattern to match key=value pairs, handling quoted strings and lists
        arg_pattern = r'(\w+)=(?:\'([^\']*?)\'|"([^"]*?)"|(\[.*?\])|(\{.*?\})|([^,\s\)]+))'

        # Use a more robust approach: manually parse through the string
        i = 0
        while i < len(args_str):
            # Skip whitespace and commas
            while i < len(args_str) and args_str[i] in [' ', ',']:
                i += 1
            if i >= len(args_str):
                break

            # Find the key
            key_match = re.match(r'(\w+)=', args_str[i:])
            if not key_match:
                break

            key = key_match.group(1)
            i += len(key_match.group(0))

            # Find the value
            value = None
            if i < len(args_str):
                if args_str[i] == "'":
                    # Single-quoted string - find the closing quote, handling escaped quotes
                    i += 1
                    start = i
                    while i < len(args_str):
                        if args_str[i] == "'" and (i == start or args_str[i-1] != '\\'):
                            value = args_str[start:i]
                            i += 1
                            break
                        i += 1
                elif args_str[i] == '"':
                    # Double-quoted string
                    i += 1
                    start = i
                    while i < len(args_str):
                        if args_str[i] == '"' and (i == start or args_str[i-1] != '\\'):
                            value = args_str[start:i]
                            i += 1
                            break
                        i += 1
                elif args_str[i] == '[':
                    # List - find matching closing bracket
                    start = i
                    bracket_count = 1
                    i += 1
                    while i < len(args_str) and bracket_count > 0:
                        if args_str[i] == '[':
                            bracket_count += 1
                        elif args_str[i] == ']':
                            bracket_count -= 1
                        i += 1
                    value = args_str[start:i]
                    # Try to evaluate the list
                    try:
                        value = eval(value)
                    except:
                        pass
                elif args_str[i] == '{':
                    # Dict - find matching closing brace
                    start = i
                    brace_count = 1
                    i += 1
                    while i < len(args_str) and brace_count > 0:
                        if args_str[i] == '{':
                            brace_count += 1
                        elif args_str[i] == '}':
                            brace_count -= 1
                        i += 1
                    value = args_str[start:i]
                    # Try to evaluate the dict
                    try:
                        value = eval(value)
                    except:
                        pass
                else:
                    # Other value - read until comma or closing paren
                    start = i
                    paren_count = 0
                    while i < len(args_str):
                        if args_str[i] == '(':
                            paren_count += 1
                        elif args_str[i] == ')':
                            if paren_count == 0:
                                break
                            paren_count -= 1
                        elif args_str[i] == ',' and paren_count == 0:
                            break
                        i += 1
                    value = args_str[start:i].strip()
                    # Try to evaluate as Python literal
                    try:
                        if value == 'None':
                            value = None
                        elif value == 'True':
                            value = True
                        elif value == 'False':
                            value = False
                        else:
                            value = eval(value)
                    except:
                        pass

            attributes[key] = value

        result.append({class_name: attributes})

    dict_result = {}
    for step in result:
        for key, value in step.items():
            dict_result[key] = value
    return dict_result

def main():
    parser = argparse.ArgumentParser(
        description='Generate architecture summary from structured outputs'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        help='Name of the run to process'
    )
    parser.add_argument(
        '--iters-path',
        type=str,
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./architecture_summary.txt',
        help='Output path for the architecture summary file (default: ./architecture_summary.txt)'
    )

    args = parser.parse_args()

    # base_path = Path(args.base_path)
    run_name = args.run_name
    output_path = Path(args.output_path)

    # run_out_dir = base_path / run_name
    # iters_dir = run_out_dir / 'run_files'
    iters_dir = Path(args.iters_path)
    # if not any(d.is_dir() and d.name.startswith('iteration_') for d in iters_dir.iterdir()):
        # iters_dir = run_out_dir / 'run_files' / run_name 

    if not iters_dir.exists():
        with open(output_path, 'w') as f:
            f.write(f"Architecture Summary for Run: {run_name}\n\nError: iters path does not exist: {iters_dir}\n")
        print(f"Architecture summary written to {output_path} (iters path not found)")
        return

    iteration_dirs = sorted(
        [d for d in iters_dir.iterdir() if d.is_dir() and d.name.startswith('iteration_')],
        key=lambda x: int(x.name.split('_')[1])
    )

    report = ''
    report += f"Architecture Summary for Run: {run_name}\n\n"

    for iteration_dir in iteration_dirs:
        structured_outputs = iteration_dir / 'structured_outputs.txt'
        if structured_outputs.exists():
            with open(structured_outputs, 'r') as f:
                content = f.read()
                parsed_data = parse_structured_output(content)
                report += f"Iteration {iteration_dir.name}:\n"
                if 'DataRepresentation' in parsed_data and 'ModelArchitecture' in parsed_data:
                    representation_info = parsed_data['DataRepresentation']['representation']
                    architecture_info = parsed_data['ModelArchitecture']['architecture']
                    report += f"  Data Representation: {representation_info}\n"
                    report += f"  Model Architecture: {architecture_info}\n\n"
                else:
                    report += "  Missing DataRepresentation or ModelArchitecture information.\n\n"

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Architecture summary written to {output_path}")

if __name__ == '__main__':
    main()

# python ./src/eval/arch_summary.py --run-name ferocious_lustre_carved --iters-path /SCRATCH/biomlbench/runs/2025-12-18T22-19-59-GMT_run-group_agentomics-ml/polarishub/tdcommons-herg_80a3b004-8436-49ff-bb1b-58e32cbe85a5/code/run_files/ferocious_lustre_carved --output-path archsumms/ferocious_lustre_carved.txt
# /SCRATCH/biomlbench/runs/2025-12-18T22-19-59-GMT_run-group_agentomics-ml/polarishub/tdcommons-herg_80a3b004-8436-49ff-bb1b-58e32cbe85a5/code/run_files/ferocious_lustre_carved