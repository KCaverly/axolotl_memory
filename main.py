import argparse
from pathlib import Path
from typing import List
from axolotl_memory.utils import load_cfg
from axolotl_memory.modelling import (
    calculate_memory_base_model,
    calculate_memory_with_lora,
)
from axolotl_memory.optimizer import calculate_memory_for_optimizer


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def create_ascii_table(
    rows, title: str, num_columns: int, padding: int, align: List[str]
):
    column_widths = [0 for i in range(num_columns)]
    for row in rows:
        if row["type"] != "title":
            for column_idx, column in enumerate(row["columns"]):
                column_widths[column_idx] = max(
                    column_widths[column_idx], len(column) + padding
                )

    sep_char, in_between = "│", "─"
    formats = [f"%{column_widths[i]}s" for i in range(len(rows[0]))]

    pattern = f"{sep_char}{sep_char.join(formats)}{sep_char}"
    diff = 0

    def make_row(left_char, middle_char, right_char):
        return f"{left_char}{middle_char.join([in_between * n for n in column_widths])}{in_between * diff}{right_char}"

    separator = make_row("├", "┼", "┤")
    if len(title) > sum(column_widths):
        diff = abs(len(title) - len(separator))
        column_widths[-1] += diff

    table = [
        make_row("┌", in_between, "┐"),
        f"{sep_char}{title.center(len(separator) - 2)}{sep_char}",
    ]

    def left_align(text, line_len):
        line = text
        return f"{line}" + " " * (line_len - len(line))

    def center_align(text, line_len):
        return text.center(line_len)

    for row in rows:
        if row["type"] == "header":
            table.append(make_row("├", "┬", "┤"))

        column_widths[-1] += diff
        line = sep_char
        for column_idx, column in enumerate(row["columns"]):
            if align[column_idx] == "left":
                line += left_align(column, column_widths[column_idx]) + sep_char
            else:
                line += center_align(column, column_widths[column_idx]) + sep_char

        table.append(line)
        if row["type"] == "header":
            table.append(separator)

    bottom_line = "└" + "┴".join([in_between * n for n in column_widths]) + "┘"
    table.append(bottom_line)

    return "\n".join(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate the memory for an axolotl config"
    )
    parser.add_argument(
        "--config", metavar="path", required=True, help="the path to config"
    )

    args = parser.parse_args()
    path = Path(args.config)

    # Load in Config
    cfg = load_cfg(path)

    # Calculate Base Model Size
    base_model = cfg.get("base_model", None)
    if base_model is None:
        raise Exception("'base_model' not available in axolotl config")
    (model_size, empty_model) = calculate_memory_base_model(cfg)

    # Calculate Load Memory
    lora_model, lora_memory = calculate_memory_with_lora(empty_model, cfg)

    # Calculate Optimizer Memory
    optimizer = cfg.get("optimizer", None)
    if optimizer is None:
        raise Exception("'optimizer' not available in axolotl config")

    optimizer_memory = calculate_memory_for_optimizer(lora_model, cfg)

    rows = [
        {"type": "header", "columns": [" Modelling", ""]},
        {
            "type": "row",
            "columns": [
                f"  Base Model ({base_model})",
                f"{sizeof_fmt(model_size, 'B')}",
            ],
        },
        {"type": "row", "columns": ["  With LORA", f"{sizeof_fmt(model_size, 'B')}"]},
        {"type": "header", "columns": [" Optimization", ""]},
        {
            "type": "row",
            "columns": [
                f"  {optimizer}",
                f"{sizeof_fmt(optimizer_memory, 'B')}",
            ],
        },
    ]

    table = create_ascii_table(
        rows,
        title="Estimate Memory",
        num_columns=2,
        padding=3,
        align=["left", "center"],
    )

    print(table)
