import re

def get_target_col_from_description(description_path='/home/data/description.md'):
    with open(description_path, 'r') as file:
        for line in file.readlines():
            if 'Target column' in line:
                match = re.search(r'\{\s*["\'](.*?)["\']\s*\}', line)
                return match.group(1) if match else None

if __name__ == '__main__':
    print(get_target_col_from_description())