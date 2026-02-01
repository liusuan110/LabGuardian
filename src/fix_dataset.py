import os
import glob

LABEL_DIR = r'd:\desktop\inter\LabGuardian\src\OneShot_Demo_Dataset\train\labels'

def xywh2xyxyxyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y - h / 2
    x3 = x + w / 2
    y3 = y + h / 2
    x4 = x - w / 2
    y4 = y + h / 2
    return [x1, y1, x2, y2, x3, y3, x4, y4]

def fix_labels():
    files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"Found {len(files)} label files.")
    
    fixed_count = 0
    for fpath in files:
        with open(fpath, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:
                # Convert xywh to xyxyxyxy
                cls = int(parts[0])
                coords = xywh2xyxyxyxy(*parts[1:])
                new_line = f"{cls} " + " ".join([f"{c:.6f}" for c in coords]) + "\n"
                new_lines.append(new_line)
                modified = True
            elif len(parts) == 9:
                new_lines.append(line)
            else:
                print(f"Warning: Unknown format in {fpath}: {line.strip()}")
                new_lines.append(line)
                
        if modified:
            with open(fpath, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1
            
    print(f"Fixed {fixed_count} files.")

if __name__ == "__main__":
    fix_labels()
