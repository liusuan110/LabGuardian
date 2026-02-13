"""
数据集标签格式修复工具
将 xywh 格式转换为 OBB 所需的 4 点格式

用法: python -m tools.fix_dataset --label-dir path/to/labels
"""

import argparse
import glob
from pathlib import Path


def xywh_to_obb(x, y, w, h):
    """Convert normalized xywh to obb 4-point format"""
    return [
        x - w / 2, y - h / 2,
        x + w / 2, y - h / 2,
        x + w / 2, y + h / 2,
        x - w / 2, y + h / 2,
    ]


def fix_labels(label_dir: Path):
    files = list(label_dir.glob("*.txt"))
    print(f"Found {len(files)} label files in {label_dir}")

    fixed = 0
    for fpath in files:
        lines = fpath.read_text().strip().splitlines()
        new_lines = []
        modified = False

        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:
                cls = int(parts[0])
                coords = xywh_to_obb(*parts[1:])
                new_lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in coords))
                modified = True
            elif len(parts) == 9:
                new_lines.append(line.strip())
            else:
                print(f"Warning: unknown format in {fpath.name}: {line.strip()}")
                new_lines.append(line.strip())

        if modified:
            fpath.write_text("\n".join(new_lines) + "\n")
            fixed += 1

    print(f"Fixed {fixed} files.")


def main():
    parser = argparse.ArgumentParser(description="Fix dataset label format")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="Path to the labels directory")
    args = parser.parse_args()

    fix_labels(Path(args.label_dir))


if __name__ == "__main__":
    main()
