from pathlib import Path

import pandas as pd

from app.profiler import profile_dataframe
from app.render_html import render_report, save_report


# 这里就是“可选维度”
# 你以后想加/删，直接改这个列表
NULL_CONTEXT_DIMS = [
    "site",
    "plat",
    "entry_source",
    "is_in_view",
    "devi_brand",
]

# 每个维度最多展示前几档
NULL_CONTEXT_TOP_N = 8


def main():
    current_dir = Path.cwd()
    print(f"当前运行目录: {current_dir}")

    csv_path = Path("sample.csv")
    print(f"正在查找文件: {csv_path.resolve()}")

    if not csv_path.exists():
        raise FileNotFoundError("没有找到 sample.csv，请把数据文件放在项目根目录。")

    print("开始读取 CSV...")
    df = pd.read_csv(csv_path)

    print(f"读取完成，行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("本次空值上下文分析维度:", NULL_CONTEXT_DIMS)
    print("开始做字段统计...")

    report_data = profile_dataframe(
        df,
        top_n=10,
        null_context_dims=NULL_CONTEXT_DIMS,
        null_context_top_n=NULL_CONTEXT_TOP_N,
    )

    print("开始渲染 HTML...")
    html = render_report(report_data)

    save_report(html, "output/report.html")

    print("完成：报告已生成到 output/report.html")


if __name__ == "__main__":
    main()