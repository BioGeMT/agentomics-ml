def collapse_repeated_lines(output: str, threshold: int = 4) -> str:
        lines = output.split('\n')
        result = []
        i = 0
        while i < len(lines):                                                                                                                                                  
            current_line = lines[i]
            count = 1
            while i + count < len(lines) and lines[i + count] == current_line:
                count += 1

            if count >= threshold:
                result.append(current_line)
                result.append(f"... [line repeated {count - 1} more times] ...")
                i += count #jump over next different line
            else: #not enough repeats, keep all lines
                for _ in range(count):
                    result.append(current_line)
                i += count

        return '\n'.join(result)

def concise_output(output: str, max_length: int = 5000):
    if len(output) > max_length:
        half_length = max_length // 2
        return f"output truncated, too long, showing first and last {half_length} characters. First {half_length}:\n{output[:half_length]}\n...\nLast {half_length}:\n{output[-half_length:]}"
    return output