#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选图片目录下「文件名（不含扩展名）少于 5 个字符」的图片，将路径全部记录到 test.txt。
用法: python filter_short_name_images.py [image_dir]
默认 image_dir 为当前目录下的 image。
"""

import sys
from pathlib import Path

# 视为图片的扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def main():
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("image")
    if not image_dir.is_dir():
        print(f"错误：目录不存在 - {image_dir}", file=sys.stderr)
        sys.exit(1)

    short_name_paths: list[str] = []
    for subdir in sorted(image_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = f.stem
            if len(stem) < 4:
                rel = f"{image_dir.name}/{subdir.name}/{f.name}".replace("\\", "/")
                short_name_paths.append(rel)

    out = Path("test.txt")
    out.write_text("\n".join(short_name_paths), encoding="utf-8")
    print(f"共 {len(short_name_paths)} 个短名图片，已写入 {out}")


if __name__ == "__main__":
    main()
