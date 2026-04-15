from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_report(report_data: dict, template_dir: str = "templates") -> str:
    """使用 Jinja2 渲染 HTML 字符串。"""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    template = env.get_template("report.html")
    html = template.render(report=report_data)
    return html


def save_report(html: str, output_path: str = "output/report.html") -> None:
    """保存 HTML 到文件。"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")