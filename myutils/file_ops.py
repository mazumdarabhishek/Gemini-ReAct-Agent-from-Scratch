import os
def read_text_file(file_path: str):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print("File does not Exists")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

def write_to_file(file_path: str, content: str) -> None:
    """
    Writes content to a file or appends to the file if it already exists
    :param file_path:
    :param content:
    :return:
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error writing to file '{file_path}': {e}")
        raise