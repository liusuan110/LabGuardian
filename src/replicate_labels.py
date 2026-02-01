import os
import shutil

LABEL_DIR = r'd:\desktop\inter\LabGuardian\src\OneShot_Demo_Dataset\train\labels'
SOURCE_FILE = 'demo_clone_000.txt'

def replicate_labels():
    source_path = os.path.join(LABEL_DIR, SOURCE_FILE)
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found!")
        return

    # Read the content of the source file
    with open(source_path, 'r') as f:
        content = f.read()

    count = 0
    # Iterate over all files in the directory
    for filename in os.listdir(LABEL_DIR):
        if filename.endswith(".txt") and filename != SOURCE_FILE:
            target_path = os.path.join(LABEL_DIR, filename)
            with open(target_path, 'w') as f:
                f.write(content)
            count += 1
            
    print(f"Successfully replicated {SOURCE_FILE} to {count} other label files.")

if __name__ == "__main__":
    replicate_labels()
