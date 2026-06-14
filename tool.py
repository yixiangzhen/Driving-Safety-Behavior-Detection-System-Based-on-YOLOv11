from pathlib import Path


# 改成你的手机数据集路径
# 这个目录下面应该有 train/labels 和 valid/labels 或 vaild/labels
DATASET_ROOT = Path(r"C:\Users\Yi\Desktop\生成\phone")

OLD_CLASS_ID = "0"
NEW_CLASS_ID = "4"


def convert_label_file(label_path):
    lines = label_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    changed = False

    for line in lines:
        line = line.strip()

        if not line:
            new_lines.append(line)
            continue

        parts = line.split()

        # 只改每一行开头的类别编号，不动后面的坐标
        if parts[0] == OLD_CLASS_ID:
            parts[0] = NEW_CLASS_ID
            changed = True

        new_lines.append(" ".join(parts))

    if changed:
        label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return changed


def main():
    label_dirs = [
        DATASET_ROOT / "train" / "labels",
        DATASET_ROOT / "valid" / "labels",
        DATASET_ROOT / "vaild" / "labels",
        DATASET_ROOT / "val" / "labels",
    ]

    total_files = 0
    changed_files = 0

    for label_dir in label_dirs:
        if not label_dir.exists():
            continue

        print(f"正在处理: {label_dir}")

        for label_path in label_dir.glob("*.txt"):
            total_files += 1
            if convert_label_file(label_path):
                changed_files += 1

    print("处理完成")
    print(f"扫描标签文件数量: {total_files}")
    print(f"修改标签文件数量: {changed_files}")
    print(f"已将类别 {OLD_CLASS_ID} 修改为类别 {NEW_CLASS_ID}")


if __name__ == "__main__":
    main()