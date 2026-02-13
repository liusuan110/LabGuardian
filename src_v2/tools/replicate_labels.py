"""
标签复制工具 — 将一个标签文件的内容复制到同目录下所有其他 .txt 文件
用于 OneShot 场景：只标注一张图，其余克隆同样标签

用法: python -m tools.replicate_labels --label-dir path/to/labels --source demo_clone_000.txt
"""

import argparse
from pathlib import Path


def replicate_labels(label_dir: Path, source_file: str):
    source_path = label_dir / source_file
    if not source_path.exists():
        print(f"Error: Source file {source_path} not found!")
        return

    content = source_path.read_text()
    count = 0

    for f in label_dir.glob("*.txt"):
        if f.name != source_file:
            f.write_text(content)
            count += 1

    print(f"Replicated {source_file} → {count} files in {label_dir}")


def main():
    parser = argparse.ArgumentParser(description="Replicate one label file to all others")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="Directory containing label .txt files")
    parser.add_argument("--source", type=str, default="demo_clone_000.txt",
                        help="Source label filename to replicate")
    args = parser.parse_args()

    replicate_labels(Path(args.label_dir), args.source)


if __name__ == "__main__":
    main()
