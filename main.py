import argparse
from enum import Enum
from pathlib import Path
from typing import List
from axolotl_memory.memory import MemoryCategory
from axolotl_memory.utils import load_cfg
from axolotl_memory.modelling import (
    load_base_model,
    get_lora_model,
    calculate_lora_adapter_memory,
    calculate_base_model_memory,
)
from axolotl_memory.optimizer import (
    calculate_gradient_memory,
    calculate_optimizer_state_memory,
)


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def print_table(title, details):
    categories = {
        MemoryCategory.MODELLING: [],
        MemoryCategory.TRAINING: [],
        MemoryCategory.INFERENCE: [],
    }

    num_columns = 3

    for item in details:
        memory_str = sizeof_fmt(item.memory, "B")
        categories[item.category].append([item.title, str(item.precision), memory_str])

    rows = []
    for category, values in categories.items():
        if len(values) == 0:
            continue

        rows.append(
            {
                "type": "header",
                "columns": [f" {str(category)}", "Precision", "Memory"],
            }
        )

        for row in values:
            rows.append({"type": "data", "columns": [f"  {row[0]}"] + row[1:]})

    table = create_ascii_table(
        rows, title, num_columns=3, padding=2, align=["left", "center", "center"]
    )

    print(table)


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
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Estimate the memory for an axolotl config"
    )
    parser.add_argument(
        "--config", metavar="path", required=True, help="the path to config"
    )

    args = parser.parse_args()
    path = Path(args.config)

    # Initialize Details
    details = []

    # Load in Config
    cfg = load_cfg(path)

    # Load Empty Model
    base_model = load_base_model(cfg)

    # Calculate Base Model Memory
    base_memory = calculate_base_model_memory(base_model, cfg)
    details.append(base_memory)

    # Calculate Adapter Memory
    if cfg.adapter == "lora":
        base_model = get_lora_model(base_model, cfg)
        lora_memory = calculate_lora_adapter_memory(base_model, cfg)
        details.append(lora_memory)

    # Calculate Training Memory

    # Calculate Memory Needed for Gradients
    gradient_memory = calculate_gradient_memory(base_model, cfg)
    details.append(gradient_memory)

    # Calculate Memory Needed for Optimizer
    optimizer_memory = calculate_optimizer_state_memory(base_model, cfg)
    details.append(optimizer_memory)

    print_table("Memory Estimate", details)
