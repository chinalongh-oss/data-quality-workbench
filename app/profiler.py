from typing import Any, List, Optional

import pandas as pd


def _safe_value(value: Any) -> Any:
    """把 pandas/numpy 的特殊值转成普通 Python 可序列化值。"""
    if pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def _display_value(value: Any) -> Any:
    """
    用于页面展示的值：
    - 空值统一显示为 __EMPTY__
    - 其他值转成可读文本
    """
    if value is None:
        return "__EMPTY__"
    return value


def _is_effectively_empty(series: pd.Series) -> pd.Series:
    """
    判断一列哪些记录算“空”：
    - NaN / None
    - 空字符串 ""
    - 只包含空格的字符串
    """
    mask = series.isna()

    # object/string 列进一步把空字符串、全空格视为“空”
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        text_series = series.fillna("").astype(str).str.strip()
        mask = mask | text_series.eq("")

    return mask


def _normalize_dimension_series(series: pd.Series) -> pd.Series:
    """
    把上下文维度列标准化用于统计：
    - 空值显示为 __EMPTY__
    - 字符串去掉首尾空格
    """
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        s = series.fillna("").astype(str).str.strip()
        s = s.replace("", "__EMPTY__")
        return s

    return series.where(~series.isna(), "__EMPTY__")


def _top_values(series: pd.Series, top_n: int = 10) -> list[dict]:
    """统计一列的 TopN 值分布。"""
    non_null_series = series.dropna()

    if non_null_series.empty:
        return []

    value_counts = non_null_series.value_counts(dropna=True).head(top_n)

    results = []
    total_non_null = int(non_null_series.shape[0])

    for value, count in value_counts.items():
        pct = round(count / total_non_null * 100, 2) if total_non_null > 0 else 0
        results.append(
            {
                "value": _safe_value(value),
                "count": int(count),
                "pct": pct,
            }
        )

    return results


def _sample_values(series: pd.Series, max_samples: int = 5) -> list[Any]:
    """抽取最多 5 个非空示例值。"""
    non_null_unique = series.dropna().drop_duplicates().head(max_samples)
    return [_safe_value(v) for v in non_null_unique.tolist()]


def _context_distribution(
    df_subset: pd.DataFrame,
    dim_name: str,
    top_n: int = 10,
) -> dict:
    """
    对某个上下文维度做分布统计。
    """
    if dim_name not in df_subset.columns:
        return {
            "dim": dim_name,
            "top_values": [],
        }

    series = _normalize_dimension_series(df_subset[dim_name])

    if series.empty:
        return {
            "dim": dim_name,
            "top_values": [],
        }

    value_counts = series.value_counts(dropna=False).head(top_n)

    total_rows = int(df_subset.shape[0])
    results = []

    for value, count in value_counts.items():
        pct = round(count / total_rows * 100, 2) if total_rows > 0 else 0
        results.append(
            {
                "value": _display_value(_safe_value(value)),
                "count": int(count),
                "pct": pct,
            }
        )

    return {
        "dim": dim_name,
        "top_values": results,
    }


def _build_null_context(
    df: pd.DataFrame,
    target_field: str,
    context_dims: Optional[List[str]],
    top_n: int = 10,
) -> dict:
    """
    统计“当 target_field 为空时”，在其它指定维度上的分布。
    """
    if not context_dims:
        return {
            "empty_count": 0,
            "dimensions": [],
        }

    target_series = df[target_field]
    empty_mask = _is_effectively_empty(target_series)
    empty_df = df[empty_mask]

    empty_count = int(empty_df.shape[0])

    if empty_count == 0:
        return {
            "empty_count": 0,
            "dimensions": [],
        }

    dimensions = []

    for dim in context_dims:
        # target 字段自己就不统计自己了
        if dim == target_field:
            continue

        if dim not in df.columns:
            continue

        dim_result = _context_distribution(empty_df, dim, top_n=top_n)
        dimensions.append(dim_result)

    return {
        "empty_count": empty_count,
        "dimensions": dimensions,
    }


def guess_field_type(series: pd.Series) -> str:
    """粗略推断字段类型。"""
    if pd.api.types.is_bool_dtype(series):
        return "bool"

    if pd.api.types.is_integer_dtype(series):
        return "int"

    if pd.api.types.is_float_dtype(series):
        return "float"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    return "string"


def profile_dataframe(
    df: pd.DataFrame,
    top_n: int = 10,
    null_context_dims: Optional[List[str]] = None,
    null_context_top_n: int = 10,
) -> dict:
    """
    对整个 DataFrame 做字段画像统计。
    """
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])

    fields = []

    for col in df.columns:
        series = df[col]

        # 这里仍保留“真正 NaN”的 null_count，方便和之前保持一致
        null_count = int(series.isna().sum())

        # 新增：有效空值，包含空字符串/空格
        effective_empty_count = int(_is_effectively_empty(series).sum())

        non_null_count = total_rows - null_count
        fill_rate = round(non_null_count / total_rows * 100, 4) if total_rows > 0 else 0.0
        distinct_count = int(series.dropna().nunique())

        is_high_cardinality = distinct_count > 20

        null_context = _build_null_context(
            df=df,
            target_field=col,
            context_dims=null_context_dims,
            top_n=null_context_top_n,
        )

        field_info = {
            "field_name": col,
            "field_type": guess_field_type(series),
            "total_rows": total_rows,
            "null_count": null_count,
            "effective_empty_count": effective_empty_count,
            "non_null_count": non_null_count,
            "fill_rate": fill_rate,
            "distinct_count": distinct_count,
            "is_high_cardinality": is_high_cardinality,
            "top_values": [] if is_high_cardinality else _top_values(series, top_n=top_n),
            "sample_values": _sample_values(series, max_samples=5),
            "null_context": null_context,
        }

        fields.append(field_info)

    fields.sort(key=lambda x: (x["fill_rate"], x["field_name"]))

    return {
        "summary": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "null_context_dims": null_context_dims or [],
        },
        "fields": fields,
    }