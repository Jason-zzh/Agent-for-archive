#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 batch_qa 生成的 answers.txt 中提取每题的序号、问题文本、耗时（秒）。
支持「耗时: 12.34 秒」与「耗时: 失败」。
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_answers_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict] = []
    q_num: int | None = None
    buf: list[str] = []

    pat_header = re.compile(r"^【问题\s*(\d+)】\s*$")
    pat_time = re.compile(r"^耗时:\s*(.+)\s*$")

    for line in lines:
        m = pat_header.match(line.strip())
        if m:
            q_num = int(m.group(1))
            buf = []
            continue
        if q_num is None:
            continue
        tm = pat_time.match(line.strip())
        if tm:
            raw = tm.group(1).strip()
            sec: float | None = None
            status = raw
            if raw.endswith("秒"):
                raw = raw[:-1].strip()
            try:
                sec = float(raw)
                status = "ok"
            except ValueError:
                sec = None
                status = raw if raw else "failed"
            question = "\n".join(buf).strip()
            rows.append(
                {
                    "index": q_num,
                    "question": question,
                    "seconds": sec,
                    "time_raw": tm.group(1).strip(),
                    "status": status,
                }
            )
            q_num = None
            buf = []
        else:
            buf.append(line)

    return rows


def main():
    parser = argparse.ArgumentParser(description="从 answers.txt 提取每题耗时")
    parser.add_argument(
        "answers_file",
        nargs="?",
        default="answers.txt",
        help="batch_qa 生成的结果文件，默认 answers.txt",
    )
    parser.add_argument(
        "-o",
        "--csv",
        metavar="FILE",
        default=None,
        help="可选：导出为 CSV（含 index,question,seconds,status）",
    )
    parser.add_argument("--stats", action="store_true", help="打印汇总统计（均值、最大、最小）")
    args = parser.parse_args()

    path = Path(args.answers_file)
    if not path.is_file():
        print(f"错误：文件不存在 {path}", file=sys.stderr)
        sys.exit(1)

    rows = parse_answers_file(path)
    if not rows:
        print("未解析到任何题目记录。", file=sys.stderr)
        sys.exit(1)

    # 控制台：序号、耗时、问题前 60 字
    for r in rows:
        sec = r["seconds"]
        tshow = f"{sec:.2f}s" if sec is not None else r["time_raw"]
        qprev = (r["question"][:60] + "…") if len(r["question"]) > 60 else r["question"]
        print(f"{r['index']:4d}  {tshow:>12}  {qprev}")

    nums = [r["seconds"] for r in rows if r["seconds"] is not None]
    if args.stats and nums:
        print()
        print(f"有效条数: {len(nums)} / {len(rows)}")
        print(f"平均耗时: {sum(nums) / len(nums):.2f} 秒")
        print(f"最短: {min(nums):.2f} 秒  最长: {max(nums):.2f} 秒")

    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["index", "question", "seconds", "status", "time_raw"],
                extrasaction="ignore",
            )
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "index": r["index"],
                        "question": r["question"],
                        "seconds": r["seconds"] if r["seconds"] is not None else "",
                        "status": r["status"],
                        "time_raw": r["time_raw"],
                    }
                )
        print(f"\n已写入 CSV: {outp.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
