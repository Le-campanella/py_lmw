"""
Python skeleton for R's lmw::lmw() front function
=================================================
This file mirrors the control flow of the R package's `lmw()` (in lmw.r):
  1) capture call & normalize arguments
  2) process estimand/method/weights/treatment/contrast/focal
  3) build design matrix from a formula (placeholder)
  4) compute weights from X and treatment (placeholder)
  5) return a result object with a `.summary()` helper

Notes
-----
completed: lmw core function, processing functions, plot function, summary functions
partially completed:
to be implemented: IV port, est layer
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union, NamedTuple

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable
import re
import warnings

lalonde = pd.read_csv('https://raw.githubusercontent.com/LeonPan-Doukeshi/lalonde_dataset/refs/heads/main/lalonde.csv')

# ----------------------------
# Table backend adapters (kept here temporarily)
# ----------------------------
class TableBackend:
    """Minimal table abstraction so IO stays out of modeling."""
    def select(self, columns: Iterable[str]) -> "TableBackend":
        raise NotImplementedError
    def to_pandas(self, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        raise NotImplementedError
    def n_rows(self) -> int:
        raise NotImplementedError

class PandasBackend(TableBackend):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def select(self, columns: Iterable[str]) -> "PandasBackend":
        return PandasBackend(self.df.loc[:, list(columns)])
    def to_pandas(self, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self.df if columns is None else self.df.loc[:, list(columns)]
    def n_rows(self) -> int:
        return len(self.df)

class PolarsBackend(TableBackend):
    def __init__(self, pl_df):
        self.df = pl_df
    def select(self, columns: Iterable[str]) -> "PolarsBackend":
        return PolarsBackend(self.df.select(list(columns)))
    def to_pandas(self, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        df = self.df if columns is None else self.df.select(list(columns))
        return df.to_pandas()
    def n_rows(self) -> int:
        return self.df.height

# ----------------------------
# Exceptions & utilities
# ----------------------------
class LMWInputError(ValueError):
    pass


def _as_upper(x: Optional[str]) -> Optional[str]:
    return None if x is None else str(x).upper()


def _capture_call(locals_dict: Dict[str, Any]) -> str:
    parts = []
    for k, v in locals_dict.items():
        if k in {"__class__", "self"}:
            continue
        if k == "data" and isinstance(v, pd.DataFrame):
            parts.append("data=<DataFrame>")
        else:
            parts.append(f"{k}={repr(v)}")
    return f"lmw({', '.join(parts)})"

def _base_vars_from_patsy_columns(cols: Sequence[str]) -> list[str]:
    """
    从 patsy 展开后的列名还原底层原始变量名（去掉交互 ':'、水平 '[...]'、C() 包装、反引号）。
    例如：'C(race)[T.black]' → 'race'；'age' → 'age'；'educ:re74' → {'educ','re74'}。
    """
    base = []
    seen = set()
    for c in cols:
        if c == "Intercept":
            continue
        for part in str(c).split(":"):
            t = part
            if t.startswith("C(") and t.endswith(")"):
                t = t[2:-1]
            if t.startswith("`") and t.endswith("`"):
                t = t[1:-1]
            t = t.split("[", 1)[0]  # 去掉 [T.level] 或 [level]
            if t and t not in seen:
                seen.add(t); base.append(t)
    return base

# ----------------------------
# Enums (R: match_arg(...))
# ----------------------------
class Method(str, Enum):
    URI = "URI"
    MRI = "MRI"


class DRMethod(str, Enum):
    WLS = "WLS"
    AIPW = "AIPW"
    NONE = "NONE"


class EstimandType(str, Enum):
    ATE = "ATE"
    ATT = "ATT"
    ATC = "ATC"
    ATO = "ATO"
    CATE = "CATE"


# -----------------
# Typed containers
# -----------------
@dataclass
class Estimand:
    kind: EstimandType
    target: Optional[Any] = None


class XObject(NamedTuple):
    X: np.ndarray
    mf: pd.DataFrame
    target: Any


@dataclass
class LMWResult:
    treat: pd.Categorical
    weights: Optional[np.ndarray]
    covs: pd.DataFrame
    estimand: Estimand
    method: Method
    base_weights: Optional[np.ndarray] = None
    s_weights: Optional[np.ndarray] = None
    dr_method: Optional[DRMethod] = None
    call: str = ""
    fixef: Optional[Any] = None
    formula: Optional[Any] = None
    target: Optional[Any] = None
    contrast: Optional[Any] = None
    focal: Optional[Any] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "n": int(self.covs.shape[0]),
            "p": int(self.covs.shape[1]),
            "method": self.method.value,
            "estimand": self.estimand.kind.value,
            "has_s_weights": self.s_weights is not None,
            "has_base_weights": self.base_weights is not None,
            "dr_method": None if self.dr_method is None else self.dr_method.value,
            "treat_levels": list(self.treat.categories.astype(str)),
        }


# ---------------------
# Processing functions
# ---------------------

# make sure the method is aligned
def process_method(method: Union[str, Method]) -> Method:
    if isinstance(method, Method):
        return method
    m = _as_upper(str(method))
    try:
        return Method[m]
    except KeyError:
        raise LMWInputError(f"method must be one of {list(Method.__members__.keys())}, got {method!r}")

# make sure the estimand is aligned
def process_estimand(estimand: Union[str, Estimand, EstimandType], target: Any, obj: Any) -> Estimand:
    if isinstance(estimand, Estimand):
        return estimand
    if isinstance(estimand, EstimandType):
        return Estimand(kind=estimand, target=target)
    e = _as_upper(str(estimand))
    try:
        et = EstimandType[e]
    except KeyError:
        raise LMWInputError(f"estimand must be one of {list(EstimandType.__members__.keys())}, got {estimand!r}")
    return Estimand(kind=et, target=target)

# make sure the double robust method is aligned
def process_dr_method(dr_method: Union[str, DRMethod, None], base_weights: Optional[Sequence[float]],
                      method: Method, estimand: Estimand) -> Optional[DRMethod]:
    # if base_weight is NONE
    if base_weights is None:
        return None
    if dr_method is None:
        return None
    if isinstance(dr_method, DRMethod):
        dm = dr_method
    else:
        d = _as_upper(str(dr_method))
        if d == "IPWRA":
            d = "WLS"
        try:
            dm = DRMethod[d]
        except KeyError:
            raise LMWInputError(f"dr.method must be one of {{'WLS','AIPW','NONE', None}}, got {dr_method!r}")

    # Forbid the combination of CATE and AIPW
    if isinstance(estimand, Estimand) and estimand.kind == EstimandType.CATE and dm == DRMethod.AIPW:
        raise LMWInputError("the CATE cannot be used with AIPW")

    return dm

# preprocess the data to a table backend
def process_data(data: Any, obj: Any, *, backend: str = "auto",
                 dtypes: Optional[Dict[str, Any]] = None,
                 na_values: Optional[Sequence[str]] = None,
                 encoding: str = "utf-8",
                 small: bool = True,
                 streaming: bool = False) -> TableBackend:
    """Normalize `data` into a table backend adapter.

    Accepts:
      - pandas.DataFrame / polars.DataFrame
      - file path to CSV/TSV/TXT/Parquet/Feather
    backend:
      - "auto"/"pandas": read via pandas (prefers pyarrow engine)
      - "polars": read via polars
      - "duckdb": read via duckdb SQL (returns pandas)
    """
    # 1) Already a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if dtypes is not None:
            data = data.astype(dtypes)
        return PandasBackend(data)

    # 2) Polars DataFrame
    try:
        import polars as pl  # type: ignore
        if isinstance(data, pl.DataFrame):
            return PolarsBackend(data)
    except Exception:
        pass

    # 3) File path
    if isinstance(data, (str, Path)):
        p = Path(data).expanduser()
        if not p.exists():
            raise LMWInputError(f"Data file not found: {p}")
        ext = p.suffix.lower()

        if backend in ("auto", "pandas"):
            if ext in {".csv", ".tsv", ".txt"}:
                sep = "\t" if ext == ".tsv" else None
                try:
                    df = pd.read_csv(p, dtype=dtypes, na_values=na_values, encoding=encoding,
                                     sep=sep, engine="pyarrow")
                except Exception:
                    try:
                        df = pd.read_csv(p, dtype=dtypes, na_values=na_values, encoding=encoding, sep=sep)
                    except UnicodeDecodeError:
                        df = pd.read_csv(p, dtype=dtypes, na_values=na_values, encoding="gb18030", sep=sep)
                return PandasBackend(df)
            if ext in {".parquet"}:
                return PandasBackend(pd.read_parquet(p))
            if ext in {".feather"}:
                return PandasBackend(pd.read_feather(p))
            raise LMWInputError(f"Unsupported file extension: {ext}")

        if backend == "polars":
            try:
                import polars as pl  # type: ignore
            except Exception as e:
                raise LMWInputError("polars is not installed; try backend='pandas'") from e
            if ext in {".csv", ".tsv", ".txt"}:
                df = pl.read_csv(p, dtypes=dtypes, try_parse_dates=True, ignore_errors=False)
                return PolarsBackend(df)
            if ext in {".parquet"}:
                return PolarsBackend(pl.read_parquet(p))
            if ext in {".feather"}:
                return PolarsBackend(pl.read_ipc(p))
            raise LMWInputError(f"Unsupported file extension for polars: {ext}")

        if backend == "duckdb":
            try:
                import duckdb  # type: ignore
            except Exception as e:
                raise LMWInputError("duckdb is not installed; try backend='pandas'") from e
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM '{str(p)}'").df()
            return PandasBackend(df)

        raise LMWInputError(f"Unknown backend: {backend!r}")

    raise LMWInputError("Unsupported data type; pass a pandas/ polars DataFrame or a file path.")


def process_base_weights(base_weights: Optional[Union[str, Sequence[float]]], data: pd.DataFrame) -> Optional[np.ndarray]:
    if base_weights is None:
        return None
    if isinstance(base_weights, str):
        if base_weights not in data.columns:
            raise LMWInputError(f"base.weights column {base_weights!r} not in data")
        return np.asarray(data[base_weights], dtype=float)
    return np.asarray(base_weights, dtype=float)


def process_s_weights(s_weights: Optional[Union[str, Sequence[float]]], data: pd.DataFrame) -> Optional[np.ndarray]:
    if s_weights is None:
        return None
    if isinstance(s_weights, str):
        if s_weights not in data.columns:
            raise LMWInputError(f"s.weights column {s_weights!r} not in data")
        return np.asarray(data[s_weights], dtype=float)
    return np.asarray(s_weights, dtype=float)


def process_treat_name(treat: Optional[Union[str, Sequence[Any]]], formula: Any, data: pd.DataFrame,
                       method: Method, obj: Any) -> str:
    # Explicit strings
    if isinstance(treat, str):
        if treat in data.columns:
            return treat
        raise LMWInputError(f"treat column {treat!r} not found in data")

    # if not assigned, use the first term in the formula
    if treat is None:
        cand = _extract_first_rhs_variable(formula, data.columns)
        if cand is not None:
            msg = (
                f"`treat` not specified; using the first variable on the RHS of the formula as treatment: {cand!r}"
            )
            # if there is "treat" in the formula, but it is not the first term
            if cand != "treat" and "treat" in data.columns:
                msg += (
                    " (note: a column named 'treat' also exists; put it first in the formula to use it)."
                )
            warnings.warn(msg, RuntimeWarning)
            return cand
        raise LMWInputError("Could not auto-detect treatment from formula; please supply `treat`.")
    # if the treat input is an array, bind this array to the data
    name = "_treat"
    data[name] = np.asarray(treat)
    return name

def process_fixef(fixef: Any, formula: Any, data: pd.DataFrame, treat_name: str) -> Any:
    if fixef is None or (isinstance(fixef, (list, tuple)) and len(fixef) == 0):
        return None

    def _series_from_spec(spec: Any) -> tuple[pd.Series, str]:
        if isinstance(spec, str):
            s = spec.strip()
            if s.startswith("~"):
                rhs = s.split("~", 1)[1]
                toks = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs) if t in data.columns]
                if len(toks) != 1:
                    raise LMWInputError("`fixef` one-sided formula must reference exactly one variable present in `data`")
                name = toks[0]
            else:
                name = s
            if name not in data.columns:
                raise LMWInputError("all variables in `fixef` must be in `data`")
            return pd.Series(data[name], name=name), name
        if isinstance(spec, pd.Series):
            ser = spec.rename(spec.name or "fixef")
            return ser, ser.name
        arr = np.asarray(spec)
        if arr.ndim != 1 or arr.shape[0] != len(data):
            raise LMWInputError("`fixef` must be 1D and have the same length as the dataset")
        ser = pd.Series(arr, name="fixef")
        return ser, ser.name

    groups, name = _series_from_spec(fixef)

    if name == treat_name:
        raise LMWInputError("the fixed effect variable cannot be the same as the treatment variable")

    f_str = str(formula)
    rhs = f_str.split("~", 1)[1] if "~" in f_str else f_str
    vars_in_rhs = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs))
    if name in vars_in_rhs:
        raise LMWInputError("the fixed effect variable should not be present in the model formula")

    if pd.isna(groups).any():
        raise LMWInputError("missing values are not allowed in the fixed effect variable")

    groups = groups.astype("category")
    try:
        groups.attrs["fixef_name"] = name
    except Exception:
        pass
    return groups


def _as_series(x: Union[str, Sequence[Any], np.ndarray, pd.Series], data: pd.DataFrame, name: str) -> pd.Series:
    if isinstance(x, str):
        return pd.Series(data[x], name=x)
    if isinstance(x, pd.Series):
        return x.rename(name)
    return pd.Series(np.asarray(x), name=name)


def process_treat(treat_name: str, data: pd.DataFrame) -> pd.Categorical:
    s = _as_series(treat_name if treat_name in data.columns else data.get(treat_name), data, name=treat_name)
    if s is None or len(s) != len(data):
        raise LMWInputError("Treatment vector length must match data rows.")
    return pd.Categorical(s)


def _maybe_len(x: Optional[Union[Sequence[Any], np.ndarray, pd.Series]]) -> Optional[int]:
    if x is None:
        return None
    try:
        return len(x)
    except Exception:
        return None


def check_lengths(treat: pd.Categorical, data: pd.DataFrame, s_weights: Optional[Sequence[float]],
                  base_weights: Optional[Sequence[float]], fixef: Any) -> None:
    n = len(data)
    if len(treat) != n:
        raise LMWInputError(f"treat length {len(treat)} does not match data rows {n}")
    for nm, vec in {"s.weights": s_weights, "base.weights": base_weights}.items():
        m = _maybe_len(vec)
        if m is not None and m != n:
            raise LMWInputError(f"{nm} length {m} does not match data rows {n}")

def _vars_in_formula(formula_str: str, data_columns: Sequence[str]) -> list[str]:
    """从 Wilkinson 公式里启发式抽取变量名，并与 data 的列名求交集。"""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula_str))
    blacklist = {"C", "I", "bs", "scale", "np", "log", "exp", "sin", "cos", "Intercept"}
    return [c for c in data_columns if (c in tokens and c not in blacklist)]

def _extract_first_rhs_variable(formula: Any, data_columns: Sequence[str]) -> Optional[str]:
    """Return the *first* identifier on the RHS of `formula` that matches a data column.
    Mirrors R's behavior: if `treat` is unspecified, take the first variable in the formula.
    """
    # 归一化成“只看右侧”的字符串
    if isinstance(formula, (list, tuple)):
        rhs = " + ".join(map(str, formula))
    else:
        s = str(formula)
        rhs = s.split("~", 1)[1] if "~" in s else s

    # 按出现顺序扫描类似变量名的 token
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)
    blacklist = {"C", "I", "bs", "scale", "np", "log", "exp", "sin", "cos", "Intercept"}
    seen = set()
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok in data_columns and tok not in blacklist:
            return tok
    return None

def _process_mf_like(data: pd.DataFrame, vars_used: Sequence[str]) -> pd.DataFrame:
    """
    模拟 R 的 process_mf():
      - 把 object/string 列转为 pandas 'category'
      - 禁止协变量中出现缺失
      - 对数值列额外禁止非有限值（NaN/Inf）
    仅处理公式实际会用到的列；返回副本。
    """
    if not vars_used:
        return data
    df = data.copy()
    # 1) 协变量不允许缺失
    if df.loc[:, vars_used].isna().any().any():
        raise LMWInputError("missing values are not allowed in the covariates")
    # 2) 字符→类别；数值列做有限性检查
    for col in vars_used:
        s = df[col]
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            df[col] = s.astype("category")
        elif pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").to_numpy()
            if not np.isfinite(vals).all():
                raise LMWInputError("non-finite values are not allowed in the covariates")
        # 其他类型（category/bool/datetime）不强转；缺失已在上面统一禁止
    return df


# These functions are for CATE
def process_target(target: Any,
                   formula: Any,
                   mf: pd.DataFrame,
                   target_weights: Optional[Sequence[float]] = None,
                   *,
                   engine: str = "patsy",
                   treat_name: str = "treat",
                   ensure_full_rank: bool = True,
                   output: str = "numpy") -> Dict[str, Any]:
    """
    Python 实现的 process_target，对齐 R 版 input_processing.R::process_target。

    作用
    ----
    - 接收 target profile（dict）或 target 数据集（DataFrame）
    - 校验 target 与公式/mf 的变量一致性
    - 将字符列转为 category，并与 mf 的类别水平对齐
    - 用与协变量相同的公式引擎生成 target 的协变量设计矩阵（无截距、无 treat 列）
    - 若 target 是数据集：返回该设计矩阵的（加权）列均值；若是 profile：直接返回那一行

    返回
    ----
    dict:
      - 'vector': np.ndarray, 目标设计的均值向量（无截距）
      - 'names' : List[str], 对应的列名（与 covariate 设计术语一致）
      - 'target_original': pd.DataFrame, 处理后的 target 数据
      - 'target_weights' : Optional[np.ndarray]
    """
    # 0) 禁止在 formula 中出现子集操作符，与 R 保持一致
    f_str = str(formula)
    if "$" in f_str or "[" in f_str:
        raise LMWInputError("subsetting operations ($, [.], [[.]) are not allowed in the model formula when `target` is specified")

    # 1) 归一化 target 为 DataFrame
    if isinstance(target, pd.DataFrame):
        tgt_df = target.copy()
    elif isinstance(target, dict):
        row = {}
        for k, v in target.items():
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)) and not isinstance(v, (str, bytes)):
                if len(v) != 1:
                    raise LMWInputError("all entries in `target` must have lengths of 1 when supplied as a mapping")
                row[k] = v[0]
            else:
                row[k] = v
        tgt_df = pd.DataFrame([row])
    else:
        raise LMWInputError("`target` must be a list/dict of covariate-value pairs or a pandas DataFrame containing the target population")

    # 2) target.weights 校验
    tw: Optional[np.ndarray] = None
    if target_weights is not None:
        if len(tgt_df) == 1:
            warnings.warn("`target.weights` is ignored when `target` is a target profile", RuntimeWarning)
        else:
            tw = np.asarray(target_weights, dtype=float)
            if tw.shape[0] != len(tgt_df):
                raise LMWInputError("`target.weights` must be a numeric vector with length equal to the number of rows of the target dataset")

    # 3) 变量对齐：target 需要包含 mf 里的所有协变量；多余变量会被丢弃（profile 情况发出 warning）
    vars_in_formula = list(mf.columns)
    vars_in_target = list(tgt_df.columns)
    missing = [v for v in vars_in_formula if v not in vars_in_target]
    if missing:
        raise LMWInputError("All covariates in the model formula must be present in `target`; missing: " + ", ".join(missing))
    extras = [v for v in vars_in_target if v not in vars_in_formula]
    if extras:
        if not isinstance(target, pd.DataFrame):
            warnings.warn("The following value(s) in `target` will be ignored: " + ", ".join(extras), RuntimeWarning)
        tgt_df = tgt_df.loc[:, [c for c in vars_in_target if c in vars_in_formula]]

    # 4) 模拟 R 的 process_mf：字符→类别、禁止缺失、数值有限性
    tgt_df = _process_mf_like(tgt_df, vars_in_formula)

    # 5) 对齐 target 列的类型到 mf（尤其是类别型的水平）
    for col in vars_in_formula:
        src = mf[col]
        dst = tgt_df[col]
        if pd.api.types.is_categorical_dtype(src):
            src_levels = src.astype("category").cat.categories
            tgt_df[col] = pd.Categorical(dst.astype(str), categories=src_levels)
        elif pd.api.types.is_numeric_dtype(src):
            vals = pd.to_numeric(dst, errors="coerce").to_numpy()
            if not np.isfinite(vals).all():
                raise LMWInputError("non-finite values are not allowed in the covariates of `target`")
        # 其他类型保留；缺失已禁止

    # 6) 用与协变量一致的公式引擎生成 target 的设计矩阵，并去掉截距与 treat 列
    if "~" not in f_str:
        f_str = "~ " + f_str

    if engine.lower() == "patsy":
        try:
            import patsy
        except Exception as e:
            raise LMWInputError("Patsy is required for process_target(engine='patsy')") from e
        Xdf = patsy.dmatrix(f_str, tgt_df, return_type="dataframe", NA_action="raise")
        if "Intercept" in Xdf.columns:
            Xdf = Xdf.drop(columns=["Intercept"])
        keep_cols = [c for c in Xdf.columns if (treat_name not in c and c not in ("Intercept", "(Intercept)"))]
        Xdf = Xdf.loc[:, keep_cols]
        mm = np.ascontiguousarray(Xdf.to_numpy(dtype=np.float64, copy=False))
        mm_names = list(Xdf.columns)

    elif engine.lower() == "formulaic":
        try:
            from formulaic import model_matrix
        except Exception as e:
            raise LMWInputError("Formulaic is required for process_target(engine='formulaic')") from e
        mm_all = model_matrix(f_str, tgt_df, ensure_full_rank=ensure_full_rank,
                              output=("sparse" if output == "sparse" else "numpy"))
        Xmat = mm_all[1] if isinstance(mm_all, tuple) else mm_all
        col_names = getattr(getattr(Xmat, "model_spec", None), "column_names", None)
        if col_names is None:
            col_names = getattr(Xmat, "column_names", None)
        if col_names is None:
            try:
                col_names = list(Xmat.to_pandas().columns)
            except Exception:
                col_names = [f"x{i}" for i in range(getattr(Xmat, "shape", (0, 0))[1])]
        cols = list(col_names)
        keep_mask = []
        for c in cols:
            has_treat = (treat_name in c) or (f"{treat_name}:" in c) or (f":{treat_name}" in c)
            is_interc = c in ("Intercept", "(Intercept)")
            keep_mask.append(not has_treat and not is_interc)
        if hasattr(Xmat, "to_numpy"):
            Xnp_all = Xmat.to_numpy()
        else:
            try:
                Xnp_all = np.asarray(Xmat)
            except Exception:
                Xnp_all = Xmat.toarray()
        idx = np.where(np.asarray(keep_mask, dtype=bool))[0]
        mm = np.ascontiguousarray(Xnp_all[:, idx], dtype=np.float64)
        mm_names = [c for c, k in zip(cols, keep_mask) if k]
    else:
        raise LMWInputError(f"Unknown engine: {engine!r}")

    # 7) 汇总：单行 profile 直接取该行；多行数据集按（加权）列均值
    if mm.shape[0] == 1:
        vec = mm.reshape(-1)
    else:
        vec = _colmeans_w(mm, tw, subset=None)

    return {
        "vector": np.ascontiguousarray(vec, dtype=np.float64),
        "names": mm_names,
        "target_original": tgt_df,
        "target_weights": None if tw is None else np.asarray(tw, dtype=np.float64),
    }


def process_contrast(contrast: Any, treat: pd.Categorical, method: Method) -> Any:
    levels = list(pd.Categorical(treat).categories)
    K = len(levels)

    if contrast is None:
        if method == Method.URI and K > 2:
            raise LMWInputError("`contrast` must be specified when the treatment has more than two levels and `method = \"URI\"`")
        return None

    # 归一化为“名字列表”
    def _to_name_list(x: Any) -> list[str]:
        if isinstance(x, (str, bytes)):
            return [str(x)]
        if np.isscalar(x):
            try:
                idx = int(x)
                if 1 <= idx <= K:
                    return [levels[idx - 1]]
            except Exception:
                pass
            return [str(x)]
        out: list[str] = []
        for e in list(x):
            if isinstance(e, (int, np.integer)):
                if 1 <= int(e) <= K:
                    out.append(levels[int(e) - 1])
                else:
                    raise LMWInputError("`contrast` index out of range")
            else:
                out.append(str(e))
        return out

    out = _to_name_list(contrast)
    if not all(o in levels for o in out):
        raise LMWInputError("`contrast` must contain the names or 1-based indices of treatment levels to be contrasted")

    # 长度规范化（对齐 R）
    if len(out) == 1:
        if K == 2:
            other = levels[0] if levels[0] != out[0] else levels[1]
            out = [out[0], other]
        else:
            if out[0] != levels[0]:
                out = [out[0], levels[0]]
            else:
                raise LMWInputError("if `contrast` is a single value, it cannot be the reference value of the treatment")
    elif len(out) != 2:
        raise LMWInputError("`contrast` cannot have length greater than 2")

    return out

def apply_contrast_to_treat(treat: pd.Categorical, contrast: Any) -> pd.Categorical:
    if contrast is None:
        return treat
    levels = list(treat.categories)
    others = [lvl for lvl in levels if lvl not in contrast]
    new_levels = list(reversed(list(contrast))) + others  # R: c(rev(contrast), others)
    vals = pd.Series(treat).astype(object).values
    return pd.Categorical(vals, categories=new_levels)

def process_focal(focal: Any, treat_contrast: pd.Categorical, estimand: Estimand, obj: Any) -> Any:
    est = estimand.kind if isinstance(estimand, Estimand) else EstimandType(str(estimand))
    levels = list(pd.Categorical(treat_contrast).categories)
    K = len(levels)

    if est in (EstimandType.ATE, EstimandType.CATE):
        if focal is not None:
            warnings.warn(f"`focal` is ignored when `estimand = \"{est.value}\"`", RuntimeWarning)
        return None

    if focal is not None:
        if isinstance(focal, (list, tuple, np.ndarray, pd.Series)):
            if len(focal) != 1:
                raise LMWInputError("`focal` must be of length 1")
            focal = focal[0]
        f = str(focal)
        if f not in [str(l) for l in levels]:
            raise LMWInputError("`focal` must be the name of a value of the treatment variable")
        return f

    if K < 2:
        raise LMWInputError("ATT/ATC require at least two treatment levels")

    if est == EstimandType.ATT:
        default = levels[1]
        warnings.warn(f"using {default!r} as the focal (treated) group. If this is incorrect, please supply `focal`.", RuntimeWarning)
    else:  # ATC
        default = levels[0]
        warnings.warn(f"using {default!r} as the focal (control) group. If this is incorrect, please supply `focal`.", RuntimeWarning)
    return default



# ----------------------------
# Helpers for design-matrix reconstruction
# ----------------------------
def _colmeans_w(M: np.ndarray, w: Optional[Sequence[float]] = None, subset: Optional[np.ndarray] = None) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if subset is not None:
        M = M[subset]
        w = None if w is None else np.asarray(w, dtype=float)[subset]
    if w is None:
        return M.mean(axis=0)
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    s = float(w.sum())
    if s <= 0:
        raise LMWInputError("sum of sampling weights must be positive for centering")
    return (M * w).sum(axis=0) / s

def center_covs(covs: np.ndarray,
                treat: pd.Categorical,
                target: Optional[Sequence[float]] = None,
                s_weights: Optional[Sequence[float]] = None,
                focal: Any = None) -> np.ndarray:
    """
    Center at reference mean, mirroring R's center_covs():
      - focal  : center at that group's (weighted) mean
      - target : center at provided target mean vector
      - else   : center at overall (weighted) mean
    """
    covs = np.asarray(covs, dtype=float)
    if focal is not None:
        mask = (pd.Series(treat).astype(object).values == focal)
        mu = _colmeans_w(covs, s_weights, subset=mask)
    elif target is not None:
        mu = np.asarray(target, dtype=float)
    else:
        mu = _colmeans_w(covs, s_weights, subset=None)
    if mu.shape[0] != covs.shape[1]:
        raise LMWInputError("target mean length does not match number of covariate columns")
    return covs - mu

def _one_hot_treatment(
    treat: pd.Categorical,
    treat_name: str,
    treat_fixed: Optional[Any] = None,
) -> tuple[np.ndarray, list[str]]:
    """(n x (K-1)) dummy matrix excluding baseline level; names like treat_name + level.

    If `treat_fixed` is provided (R's get_X.R behavior), return a row-constant matrix where
    each non-baseline column equals 1.0 if that level equals `treat_fixed`, else 0.0.
    """
    levels = list(treat.categories)
    if len(levels) < 2:
        raise LMWInputError("treatment must have at least 2 levels")
    n = len(treat)
    K = len(levels)
    colnames = [f"{treat_name}{lvl}" for lvl in levels[1:]]

    # treat_fixed branch: build constant columns comparing levels[-baseline] to treat_fixed
    if treat_fixed is not None:
        # Accept either the raw category value or its string representation; compare by value
        if treat_fixed not in levels and str(treat_fixed) not in [str(l) for l in levels]:
            raise LMWInputError(
                f"treat_fixed {treat_fixed!r} is not among treatment levels {levels!r}"
            )
        cmp = np.array([1.0 if (lvl == treat_fixed or str(lvl) == str(treat_fixed)) else 0.0
                        for lvl in levels[1:]], dtype=float)
        t_mat = np.tile(cmp, (n, 1))
        return t_mat, colnames

    # Default: per-row one-hot for each non-baseline level
    t_mat = np.zeros((n, K - 1), dtype=float)
    vals = pd.Series(treat).astype(object).values
    for j, lvl in enumerate(levels[1:]):
        t_mat[:, j] = (vals == lvl)
    return t_mat, colnames

def _mask_cols_without_treat(col_names: Sequence[str], treat_name: str) -> np.ndarray:
    """Return a boolean mask keeping columns that are neither treatment-related nor intercepts.
    Robust to names like 'T', 'T:x1', 'C(T)[T.1]' (patsy), and '(Intercept)'.
    """
    pat = re.compile(rf"(^|:){re.escape(treat_name)}(\[|:|$)")
    mask: list[bool] = []
    for c in col_names:
        is_intercept = c in ("Intercept", "(Intercept)")
        has_treat = bool(pat.search(c))
        mask.append((not has_treat) and (not is_intercept))
    return np.asarray(mask, dtype=bool)

def _build_covs_with_engine(formula_str: str,
                            data: pd.DataFrame,
                            treat_name: str,
                            engine: str,
                            ensure_full_rank: bool,
                            output: str) -> tuple[np.ndarray, list[str]]:
    """Return (covs_np, cov_col_names) after removing any columns involving the treatment."""

    if engine.lower() == "formulaic":
        try:
            from formulaic import model_matrix
        except Exception as e:
            raise LMWInputError(
                "Formulaic is not installed; install with `pip install formulaic` or use engine='patsy'."
            ) from e
        mm = model_matrix(formula_str, data, ensure_full_rank=ensure_full_rank,
                          output=("sparse" if output == "sparse" else "numpy"))
        Xmat = mm[1] if isinstance(mm, tuple) else mm
        col_names = getattr(getattr(Xmat, "model_spec", None), "column_names", None)
        if col_names is None:
            col_names = getattr(Xmat, "column_names", None)
        if col_names is None:
            try:
                col_names = list(Xmat.to_pandas().columns)
            except Exception:
                col_names = [f"x{i}" for i in range(getattr(Xmat, "shape", (0, 0))[1])]
        keep = _mask_cols_without_treat(col_names, treat_name)
        if hasattr(Xmat, "to_numpy"):
            Xnp_all = Xmat.to_numpy()
        else:
            try:
                Xnp_all = np.asarray(Xmat)
            except:
                Xnp_all = Xmat.toarray()
        X_np = np.ascontiguousarray(Xnp_all[:, np.where(keep)[0]], dtype=np.float64)
        kept_cols = [c for c, k in zip(col_names, keep) if k]
        return X_np, kept_cols

    elif engine.lower() == "patsy":
        try:
            import patsy
        except Exception as e:
            raise LMWInputError(
                "Patsy is not installed; install with `pip install patsy` or use engine='formulaic'."
            ) from e
        Xdf = patsy.dmatrix(formula_str, data, return_type="dataframe", NA_action="raise")
        col_names = list(Xdf.columns)
        keep_mask = _mask_cols_without_treat(col_names, treat_name)
        keep_cols = [c for c, k in zip(col_names, keep_mask) if k]
        X_np = np.ascontiguousarray(Xdf.loc[:, keep_cols].to_numpy(dtype=np.float64, copy=False))
        return X_np, keep_cols

    else:
        raise LMWInputError(f"Unknown formula engine: {engine!r}")

def _remove_treat_terms_patsy(formula_str: str, data: pd.DataFrame, treat_name: str) -> tuple[str, list[str], set[str], bool]:
    """
    Expand formula with patsy and remove treatment-containing terms.

    Returns
    -------
    formula_no_treat : str
        RHS-only formula string rebuilt without treat terms (+ interactions with treat stripped of it).
    new_terms : list[str]
        Term labels of the no-treat formula (expanded; excludes Intercept).
    interacted_terms : set[str]
        Subset of `new_terms` that originated from a treat-containing term.
    has_intercept : bool
        Whether the original formula had an intercept term.
    """
    import patsy
    Xfull = patsy.dmatrix(formula_str, data, return_type="dataframe", NA_action="raise")
    di = Xfull.design_info
    term_names = [t for t in di.term_names if t != "Intercept"]
    new_terms: list[str] = []
    interacted_terms: set[str] = set()
    for t in term_names:
        parts = t.split(":")
        if treat_name in parts:
            reduced = ":".join([p for p in parts if p != treat_name])
            if reduced:
                interacted_terms.add(reduced)
                new_terms.append(reduced)
        else:
            new_terms.append(t)
    # de-duplicate keeping order
    seen = set()
    uniq_terms = []
    for t in new_terms:
        if t not in seen:
            uniq_terms.append(t); seen.add(t)
    has_intercept = ("Intercept" in di.term_names)
    rhs = ("1 + " if has_intercept else "0 + ") + " + ".join(uniq_terms) if uniq_terms else ("1" if has_intercept else "0")
    formula_no_treat = f"~ {rhs}"
    return formula_no_treat, uniq_terms, interacted_terms, has_intercept

def _demean_by_group(vec: np.ndarray, groups: Union[pd.Series, np.ndarray], w: np.ndarray) -> np.ndarray:
    """Weighted within-group demeaning: x_i - \bar{x}_g (权重为 w)."""
    if isinstance(groups, pd.DataFrame):
        if groups.shape[1] != 1:
            raise LMWInputError("multi-way fixed effects are not yet supported in Python get_w_from_X; pass a single grouping vector")
        groups = groups.iloc[:, 0]
    g = pd.Series(groups).reset_index(drop=True)
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.shape[0] != g.shape[0] or x.shape[0] != w.shape[0]:
        raise LMWInputError("dimensions of x, groups, and weights must agree in demean")
    df = pd.DataFrame({"x": x, "w": w, "g": g})
    sums = df["w"].groupby(df["g"], sort=False).sum()
    sums = sums.replace(0.0, 1.0)
    num = (df["x"] * df["w"]).groupby(df["g"], sort=False).sum()
    mu = (num / sums).reindex(df["g"]).to_numpy()
    return x - mu

# ----------------------------
# Core computational functions
# ----------------------------
def get_X_from_formula(
    formula: Any,
    table: TableBackend,
    treat_contrast: pd.Categorical,
    method: Method,
    estimand: Estimand,
    target: Any,
    s_weights: Optional[Sequence[float]],
    target_weights: Optional[Sequence[float]],
    focal: Any,
    *,
    engine: str = "formulaic",
    ensure_full_rank: bool = True,
    output: str = "numpy",
    treat_name: Optional[str] = None,
    treat_fixed: Optional[Any] = None,
) -> XObject:
    """
    Build final X matching R's get_X_from_formula (URI/MRI):

    1) Remove treatment at *term* level; build covariate matrix.
    2) Center covariates per estimand/target/focal.
    3) Build treatment dummies (K-1 columns).
    4) URI: keep non-interacting covs; replace interacting covs by (t_mat * cov).
       MRI: keep all covs; and add (t_mat * cov) for all.
    5) Prepend intercept and t_mat.
    """
    # convert the table to pandas backend and get treat variable name
    data = table.to_pandas()
    if treat_name is None:
        treat_name = "treat"

    # Normalize formula to a RHS-present string
    # 1. remove the response term
    if isinstance(formula, (list, tuple)):
        rhs = " + ".join(map(str, formula)) if len(formula) else "1"
        formula_str = f"~ {rhs}"
    else:
        formula_str = str(formula)
        if "~" not in formula_str:
            formula_str = f"~ {formula_str}"

    # Treatment dummies (exclude baseline)
    t_mat, t_colnames = _one_hot_treatment(treat_contrast, treat_name, treat_fixed=treat_fixed)
    n = t_mat.shape[0]

    # —— 新增：对公式使用到的原始列做建模前预处理（模拟 R 的 process_mf） ——
    vars_used = _vars_in_formula(formula_str, data.columns)
    vars_used = [c for c in vars_used if c != treat_name]
    data_proc = _process_mf_like(data, vars_used)

    # 2. extract and remove the response variable
    # Decide if we can/should use patsy for term-level bookkeeping
    use_patsy = (method == Method.URI) or (engine.lower() == "patsy")

    covs_np: np.ndarray
    cov_colnames: list[str]
    interacted_terms: set[str] = set()
    term_to_indices: Dict[str, list[int]] = {}

    if use_patsy:
        try:
            import patsy  # noqa: F401
        except Exception:
            if method == Method.URI:
                raise LMWInputError("URI requires term-level parsing; install 'patsy' or set engine='patsy'.")
            # Fallback for MRI: just build covs without term bookkeeping
            covs_np, cov_colnames = _build_covs_with_engine(formula_str, data_proc, treat_name, engine,
                                                            ensure_full_rank, output)
        else:
            # Remove treat at *term* level and rebuild a no-treat formula
            f_no_treat, new_terms, interacted_terms, has_intercept = _remove_treat_terms_patsy(
                formula_str, data_proc, treat_name
            )
            # Build covariate design with patsy (drop Intercept)
            Xcov_df = patsy.dmatrix(f_no_treat, data_proc, return_type="dataframe", NA_action="raise")
            cov_cols = [c for c in Xcov_df.columns if c != "Intercept"]
            covs_np = np.ascontiguousarray(Xcov_df.loc[:, cov_cols].to_numpy(dtype=np.float64, copy=False))
            cov_colnames = list(cov_cols)
            # Map terms to cov columns using patsy term slices when available; fallback to prefix matching
            di2 = Xcov_df.design_info
            orig_cols = list(Xcov_df.columns)  # includes possible 'Intercept'
            name_to_new = {name: i for i, name in enumerate(cov_colnames)}  # after dropping Intercept
            for term in new_terms:
                idxs: list[int] = []
                term_slices = getattr(di2, "term_name_slices", {})
                sl = term_slices.get(term) if isinstance(term_slices, dict) else None
                if sl is not None:
                    term_names = orig_cols[sl]
                    # Ensure iterable
                    if not isinstance(term_names, (list, tuple)):
                        term_names = [term_names]
                    for nm in term_names:
                        if nm == "Intercept":
                            continue
                        j = name_to_new.get(nm)
                        if j is not None:
                            idxs.append(j)
                else:
                    # Fallback: prefix-based grouping (robust for common contrast codings)
                    idxs = [i for i, c in enumerate(cov_colnames)
                            if c == term or c.startswith(term + ":") or c.startswith(term + "[")]
                term_to_indices[term] = idxs
    else:
        covs_np, cov_colnames = _build_covs_with_engine(formula_str, data_proc, treat_name, engine,
                                                        ensure_full_rank, output)

    # Enforce no non-finite values in covariate design (aligns with process_mf)
    if covs_np.size and not np.isfinite(covs_np).all():
        raise LMWInputError("non-finite values are not allowed in the covariates")

    # —— 在构造 covs_np / cov_colnames 之后、center_covs 之前 ——
    target_info = None
    target_vec = target  # 默认沿用传入
    if hasattr(estimand, "kind") and getattr(estimand.kind, "name", None) == "CATE":
        # 构造“无 treat”的 RHS 公式：patsy 分支已有 f_no_treat；否则退化为 vars_used 的线性 RHS
        if 'use_patsy' in locals() and use_patsy and 'f_no_treat' in locals():
            f_rhs = f_no_treat
            engine_for_target = "patsy"
        else:
            f_rhs = f"~ {' + '.join(vars_used) if vars_used else '1'}"
            engine_for_target = engine
        target_info = process_target(
            target=target,
            formula=f_rhs,
            mf=data_proc.loc[:, vars_used] if vars_used else data_proc,
            target_weights=target_weights,
            engine=engine_for_target,
            treat_name=treat_name,
            ensure_full_rank=ensure_full_rank,
            output=output,
        )
        # 按 cov_colnames 顺序对齐 target 向量
        name_to_pos = {n: i for i, n in enumerate(target_info["names"])}
        try:
            target_vec = np.asarray([target_info["vector"][name_to_pos[n]] for n in cov_colnames], dtype=np.float64)
        except KeyError as e:
            raise LMWInputError(f"target profile design columns do not align with covariate design: missing {e}")

    # Center covariates
    covs_centered = center_covs(covs_np, treat_contrast, target_vec, s_weights, focal)

    # Determine which cov columns interact, following R's URI: keep ALL main-effect covariates,
    # and add interactions only for those terms that originally included the treatment.
    if method == Method.URI and term_to_indices:
        # R's URI: keep ALL main-effect covariates, and add interactions only for
        # those terms that originally included the treatment.
        all_idx = list(range(covs_centered.shape[1]))
        interact_indices = sorted({j for term, idxs in term_to_indices.items() if term in interacted_terms for j in idxs})
        keep_indices = all_idx
    elif method == Method.URI and not term_to_indices:
        # Fallback when term mapping is unavailable: keep all covariates with no interactions
        interact_indices = []
        keep_indices = list(range(covs_centered.shape[1]))
    else:
        # MRI: all covs both kept and interacted
        interact_indices = list(range(covs_centered.shape[1]))
        keep_indices = list(range(covs_centered.shape[1]))

    # Assemble X pieces
    # Note: For URI, main effects for all covariates are kept, and interactions are added only for those
    # covariates whose original terms included the treatment.
    pieces: list[np.ndarray] = []
    colnames: list[str] = []

    # Intercept + T dummies
    pieces.append(np.ones((n, 1), dtype=np.float64)); colnames.append("(Intercept)")
    pieces.append(t_mat); colnames.extend(t_colnames)

    # Main-effect covariates (all for URI and MRI)
    if keep_indices:
        pieces.append(covs_centered[:, keep_indices])
        colnames.extend([cov_colnames[j] for j in keep_indices])

    # Interactions t_mat * covs (only for selected covariates in URI)
    if interact_indices:
        C = covs_centered[:, interact_indices]            # (n, pI)
        Xint = (t_mat[:, :, None] * C[:, None, :]).reshape(n, -1)  # (n, (K-1)*pI)
        pieces.append(Xint)
        for tcn in t_colnames:
            for j in interact_indices:
                colnames.append(f"{tcn}:{cov_colnames[j]}")

    X = np.ascontiguousarray(np.hstack(pieces), dtype=np.float64)

    # Modeling frame: original covariates (drop treatment column), as in R
    if 'cov_cols' in locals() and cov_cols:
        base_vars = _base_vars_from_patsy_columns(cov_cols)
        mf_covs = data_proc.loc[:, [v for v in base_vars if v in data_proc.columns]]
    else:
        mf_covs = (data_proc.loc[:, vars_used]
                   if vars_used else data_proc.drop(columns=[treat_name], errors="ignore"))
    target_payload = target_info if target_info is not None else target
    return XObject(X=X, mf=mf_covs, target=target_payload)



def get_w_from_X(X: np.ndarray, treat_contrast: pd.Categorical, method: Method,
                  base_weights: Optional[Sequence[float]], s_weights: Optional[Sequence[float]],
                  dr_method: Optional[DRMethod], fixef: Any) -> np.ndarray:
    """
    Python port of R/get_w.R::get_w_from_X
    - 采样权重 s.weights: 默认为 1
    - 除 AIPW 外，base.weights 与 s.weights 相乘并进入主权重
    - 可选 FE：对每一列做组内带权去均值（AIPW+FE 禁止）
    - 对 rw*X 做列主元 QR → 取独立列 (pivot)，构造 (R^T R)^{-1}
    - URI：使用“处理组哑变量列”（原列索引=1）对应的列向量
    - MRI：对每个处理水平使用其对应列；若共线缺失则回退到截距列并取负号
    - AIPW：对 IPW 残差做增广（按 R 里 .lm.fit 的等效实现）
    - 最后按组缩放，使得每组权重均值=1
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape

    # 1) s.weights
    if s_weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(s_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise LMWInputError("s.weights length does not match number of rows in X")

    # 2) base.weights（除 AIPW 外，直接乘入）
    use_base_in_main = (base_weights is not None) and (dr_method != DRMethod.AIPW)
    if use_base_in_main:
        bw = np.asarray(base_weights, dtype=np.float64).reshape(-1)
        if bw.shape[0] != n:
            raise LMWInputError("base.weights length does not match number of rows in X")
        w = w * bw

    # 3) URI 下取 treated 哑变量列（原始列索引=1；0 是截距）
    if method == Method.URI:
        if d < 2:
            raise LMWInputError("X must have at least 2 columns (Intercept and treated dummy) for URI")
        t = X[:, 1].copy()
    else:
        t = None

    # 4) FE 约束 + 吸收
    if fixef is not None and dr_method == DRMethod.AIPW and base_weights is not None:
        raise LMWInputError("fixed effects cannot be used with AIPW")
    if fixef is not None:
        # 跳过截距（第 0 列）。常数列去均值后会变 0，会让 MRI 的“回退到截距”失效。
        for j in range(1, d):
            X[:, j] = _demean_by_group(X[:, j], fixef, w)

    # 5) 列主元 QR（优先 SciPy；否则 pinv 回退）
    rw = np.sqrt(w)
    A = rw[:, None] * X
    try:
        from scipy.linalg import qr as scipy_qr, solve_triangular
        Q, R, piv = scipy_qr(A, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        tol = diag[0] * max(n, d) * np.finfo(float).eps if diag.size else 0.0
        p = int(np.sum(diag > tol))
        R11 = R[:p, :p]
        invR = solve_triangular(R11, np.eye(p), lower=False)
        XtX1_p = invR @ invR.T            # (R^T R)^{-1} in pivoted subspace
        piv_ind = np.asarray(piv[:p], dtype=int)  # 被保留的独立列（原列索引，0 基）
    except Exception:
        XtWX = (X.T * w) @ X
        XtX1_full = np.linalg.pinv(XtWX)
        p = int(np.linalg.matrix_rank(A))
        XtX1_p = XtX1_full
        piv_ind = np.arange(d, dtype=int)

    X_p = X[:, piv_ind]  # 把 X 也按独立列顺序重排，和 R 一致

    # 6) 主权重（URI/MRI）
    if method == Method.URI:
        # 在独立列里找到“原始第 2 列（treated 哑变量）”的位置
        try:
            col_pos = int(np.where(piv_ind == 1)[0][0])
        except Exception:
            raise LMWInputError("treated dummy column (2nd) is not in the independent set; check X construction")
        v = XtX1_p[:, col_pos].reshape(-1, 1)         # p x 1
        weights = (w * (X_p @ v).reshape(-1)).astype(np.float64)
        weights[t == 0.0] *= -1.0                     # 控制组翻符号
    else:  # MRI
        levels = list(treat_contrast.categories)
        pos_map = {int(orig): int(k) for k, orig in enumerate(piv_ind)}  # 原列索引 → 在独立列中的位置
        weights = np.zeros(n, dtype=np.float64)
        codes = pd.Categorical(treat_contrast).codes  # 0..K-1
        for i, _ in enumerate(levels, start=1):       # i: 1..K
            in_t = (codes == (i - 1))
            if (i - 1) in pos_map:                    # 该组对应的列存在（i=1 → 截距 0）
                v = XtX1_p[:, pos_map[i - 1]].reshape(-1, 1)
                weights[in_t] = w[in_t] * (X_p[in_t, :] @ v).reshape(-1)
            else:
                # 缺列 → 回退到截距列（原索引 0）且取负号（与 R 一致）
                if 0 not in pos_map:
                    raise LMWInputError("intercept column not in independent set; design matrix may be singular")
                v0 = XtX1_p[:, pos_map[0]].reshape(-1, 1)
                weights[in_t] = -w[in_t] * (X_p[in_t, :] @ v0).reshape(-1)

    # 7) AIPW extension
    if base_weights is not None and dr_method == DRMethod.AIPW:
        ipw = np.asarray(base_weights, dtype=np.float64) * np.asarray(s_weights if s_weights is not None else np.ones(n), dtype=np.float64)
        # 各组内归一化使得 sum=1
        ipw_norm = ipw.copy()
        codes = pd.Categorical(treat_contrast).codes
        for k in np.unique(codes):
            mask = (codes == k)
            s = ipw_norm[mask].sum()
            if s != 0:
                ipw_norm[mask] /= s
        ipw_rw = np.divide(ipw_norm, rw, out=np.zeros_like(ipw_norm), where=(rw != 0))

        if method == Method.URI:
            # 多分类 URI：只保留前两个对比组的权重，其它组置 0
            if len(treat_contrast.categories) > 2:
                first_two = set(treat_contrast.categories[:2])
                keep_mask = pd.Series(treat_contrast).isin(first_two).to_numpy()
                ipw_rw = np.where(keep_mask, ipw_rw, 0.0)
            # 与 R 的“先翻符号再回归再翻回”的写法一致
            ipw_rw = ipw_rw.copy()
            ipw_rw[t == 0.0] *= -1.0
            y = ipw_rw.reshape(-1, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = (y - A @ beta).reshape(-1)
            aug = rw * resid
            aug[t == 0.0] *= -1.0
        else:  # MRI
            y = ipw_rw.reshape(-1, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = (y - A @ beta).reshape(-1)
            aug = rw * resid

        weights = weights + aug

    # 8) scale each group to make the mean in each group equal to 1
    if method == Method.URI and len(treat_contrast.categories) > 2 and t is not None:
        for val in (0.0, 1.0):
            mask = (t == val)
            m = weights[mask].mean()
            if m != 0:
                weights[mask] /= m
    else:
        codes = pd.Categorical(treat_contrast).codes
        for k in np.unique(codes):
            mask = (codes == k)
            m = weights[mask].mean()
            if m != 0:
                weights[mask] /= m

    return np.ascontiguousarray(weights, dtype=np.float64)


# ----------------------------
# Summary helpers (weighted stats & matrix conversion)
# ----------------------------

def ESS(w: Sequence[float]) -> float:
    w = np.asarray(w, dtype=float)
    s1 = float(w.sum())
    s2 = float(np.sum(w * w))
    return 0.0 if s2 == 0.0 else (s1 * s1) / s2


def mean_w(x: Sequence[float], w: Optional[Sequence[float]] = None, mask: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    if mask is not None:
        x = x[mask]
        if w is not None:
            w = np.asarray(w, dtype=float)[mask]
    if w is None:
        return float(x.mean()) if x.size else 0.0
    w = np.asarray(w, dtype=float)
    sw = float(w.sum())
    return float((x * w).sum() / sw) if sw != 0.0 else 0.0


def var_w(x: Sequence[float], is_binary: bool, w: Optional[Sequence[float]] = None, mask: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    if mask is not None:
        x = x[mask]
        if w is not None:
            w = np.asarray(w, dtype=float)[mask]
    if is_binary:
        if w is None:
            p = float(x.mean()) if x.size else 0.0
        else:
            w = np.asarray(w, dtype=float)
            sw = float(w.sum())
            p = float((x * w).sum() / sw) if sw != 0.0 else 0.0
        return p * (1.0 - p)
    if w is None:
        return float(np.var(x, ddof=0)) if x.size else 0.0
    w = np.asarray(w, dtype=float)
    sw = float(w.sum())
    if sw == 0.0:
        return 0.0
    m = float((x * w).sum() / sw)
    return float(((x - m) ** 2 * w).sum() / sw)


def covs_df_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    模拟 R 的 covs_df_to_matrix():
      - numeric/bool → 一列
      - categorical/object → 对各水平做 one-hot（全水平）
    返回 (矩阵, 列名)，不含截距。
    """
    from pandas.api.types import CategoricalDtype
    if df is None or df.shape[1] == 0:
        return np.zeros((len(df) if isinstance(df, pd.DataFrame) else 0, 0), dtype=float), []
    mats: list[np.ndarray] = []
    names: list[str] = []
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, CategoricalDtype) or s.dtype == object:
            cat = pd.Categorical(s)
            for lvl in cat.categories:
                arr = (cat == lvl)
                vals = np.asarray(arr, dtype=float).reshape(-1, 1)
                mats.append(vals)
                names.append(f"{col}[{lvl}]")
        else:
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            if not np.isfinite(vals).all():
                raise LMWInputError("non-finite values are not allowed in covariates")
            mats.append(vals)
            names.append(str(col))
    X = np.hstack(mats) if mats else np.zeros((len(df), 0), dtype=float)
    return np.ascontiguousarray(X, dtype=np.float64), names


def _strip_tics(names: list[str]) -> list[str]:
    out = []
    for n in names:
        out.append(n[1:-1] if (isinstance(n, str) and n.startswith("`") and n.endswith("`")) else n)
    return out

# ---- KS / TKS statistics (weighted) ----

def ks_w(x: np.ndarray, treat: pd.Categorical, weights: np.ndarray) -> float:
    levels = list(treat.categories)
    if len(levels) < 2:
        return 0.0
    t1 = levels[1]  # treated level
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float).copy()
    tr = pd.Series(treat).astype(object).to_numpy()

    # normalize within group to sum 1
    m1 = (tr == t1)
    m0 = ~m1
    s1 = w[m1].sum();  s0 = w[m0].sum()
    if s1 > 0: w[m1] /= s1
    if s0 > 0: w[m0] /= s0

    ord_idx = np.argsort(x, kind="mergesort")  # stable like R's order()
    x_ord = x[ord_idx]
    w_ord = w[ord_idx]
    t_ord = tr[ord_idx]

    w_signed = w_ord.copy()
    w_signed[t_ord == t1] *= -1.0

    diffs = np.abs(np.cumsum(w_signed))
    jumps = np.concatenate(([True], np.diff(x_ord) != 0))
    return float(diffs[jumps].max(initial=0.0))


def tks_w(x: np.ndarray, x_target: np.ndarray, weights: np.ndarray, target_weights: np.ndarray) -> float:
    tr = pd.Categorical(np.r_[np.ones_like(x, dtype=int), np.zeros_like(x_target, dtype=int)], categories=[0, 1])
    xx = np.r_[x, x_target]
    ww = np.r_[weights, target_weights]
    return ks_w(xx, tr, ww)

# ---- Per-variable balance & distribution ----

def balance_one_var(x: np.ndarray, treat: pd.Categorical, weights: np.ndarray, s_weights: np.ndarray,
                    standardize: bool = True, focal: Optional[str] = None,
                    x_target: Optional[np.ndarray] = None, target_weights: Optional[np.ndarray] = None) -> dict:
    levels = list(treat.categories)
    t1 = levels[1]
    x = np.asarray(x, dtype=float)
    s_weights = np.asarray(s_weights, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if x.size and np.all(np.abs(x - x[0]) < np.sqrt(np.finfo(float).eps)):
        keys = ("SMD", f"TSMD {levels[0]}", f"TSMD {levels[1]}", "KS", f"TKS {levels[0]}", f"TKS {levels[1]}") if standardize \
               else ("MD", f"TMD {levels[0]}", f"TMD {levels[1]}", "KS", f"TKS {levels[0]}", f"TKS {levels[1]}")
        return {k: 0.0 for k in keys}

    bin_var = np.all((x == 0) | (x == 1))
    tr_np = pd.Series(treat).astype(object).to_numpy()
    too_small = any([(s_weights[tr_np == lv] != 0).sum() < 2 for lv in levels[:2]])

    if focal is not None:
        M = mean_w(x, s_weights, mask=(tr_np == focal))
    elif x_target is not None:
        M = mean_w(x_target, target_weights)
    else:
        M = mean_w(x, s_weights)

    m0 = mean_w(x, weights, mask=(tr_np != t1))
    m1 = mean_w(x, weights, mask=(tr_np == t1))

    mdiff = m1 - m0
    mdiff0 = m0 - M
    mdiff1 = m1 - M

    out = {}
    if standardize:
        if not too_small:
            if focal is not None:
                std = np.sqrt(var_w(x, bin_var, s_weights, mask=(tr_np == focal)))
            elif x_target is not None and (x_target.size > 1):
                std = np.sqrt(var_w(x_target, bin_var, target_weights))
            else:
                std = np.sqrt(np.mean([var_w(x, bin_var, s_weights, mask=(tr_np == lv)) for lv in levels]))
            if std < np.sqrt(np.finfo(float).eps):
                std = np.sqrt(var_w(x, bin_var, s_weights))
            out["SMD"] = mdiff / std
            out[f"TSMD {levels[0]}"] = mdiff0 / std
            out[f"TSMD {levels[1]}"] = mdiff1 / std
    else:
        out["MD"] = mdiff
        out[f"TMD {levels[0]}"] = mdiff0
        out[f"TMD {levels[1]}"] = mdiff1

    # KS / TKS
    if bin_var:
        out["KS"] = abs(mdiff)
        out[f"TKS {levels[0]}"] = abs(mdiff0)
        out[f"TKS {levels[1]}"] = abs(mdiff1)
    elif not too_small:
        out["KS"] = ks_w(x, treat, weights)
        mask0 = (tr_np != t1); mask1 = ~mask0
        if focal is not None:
            mask_f = (tr_np == focal)
            out[f"TKS {levels[0]}"] = tks_w(x[mask0], x[mask_f], weights[mask0], s_weights[mask_f])
            out[f"TKS {levels[1]}"] = tks_w(x[mask1], x[mask_f], weights[mask1], s_weights[mask_f])
        elif x_target is not None:
            tw = np.ones_like(x_target, dtype=float) if target_weights is None else target_weights
            out[f"TKS {levels[0]}"] = tks_w(x[mask0], x_target, weights[mask0], tw)
            out[f"TKS {levels[1]}"] = tks_w(x[mask1], x_target, weights[mask1], tw)
        else:
            out[f"TKS {levels[0]}"] = tks_w(x[mask0], x, weights[mask0], s_weights)
            out[f"TKS {levels[1]}"] = tks_w(x[mask1], x, weights[mask1], s_weights)

    return out


def balance_one_var_multi(x: np.ndarray, treat: pd.Categorical, weights: np.ndarray, s_weights: np.ndarray,
                          standardize: bool = True, focal: Optional[str] = None,
                          x_target: Optional[np.ndarray] = None, target_weights: Optional[np.ndarray] = None) -> dict:
    levels = list(treat.categories)
    x = np.asarray(x, dtype=float)
    s_weights = np.asarray(s_weights, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if x.size and np.all(np.abs(x - x[0]) < np.sqrt(np.finfo(float).eps)):
        keys = [("TSMD" if standardize else "TMD") + f" {lv}" for lv in levels] + [f"TKS {lv}" for lv in levels]
        return {k: 0.0 for k in keys}

    bin_var = np.all((x == 0) | (x == 1))
    tr_np = pd.Series(treat).astype(object).to_numpy()
    too_small = any([(s_weights[tr_np == lv] != 0).sum() < 2 for lv in levels])

    if focal is not None:
        M = mean_w(x, s_weights, mask=(tr_np == focal))
    elif x_target is not None:
        M = mean_w(x_target, target_weights)
    else:
        M = mean_w(x, s_weights)

    means = {lv: mean_w(x, weights, mask=(tr_np == lv)) for lv in levels}
    mdifft = {lv: means[lv] - M for lv in levels}

    out = {}
    if standardize:
        if not too_small:
            if focal is not None:
                std = np.sqrt(var_w(x, bin_var, s_weights, mask=(tr_np == focal)))
            elif x_target is not None and (x_target.size > 1):
                std = np.sqrt(var_w(x_target, bin_var, target_weights))
            else:
                std = np.sqrt(np.mean([var_w(x, bin_var, s_weights, mask=(tr_np == lv)) for lv in levels]))
            if std < np.sqrt(np.finfo(float).eps):
                std = np.sqrt(var_w(x, bin_var, s_weights))
            for lv in levels:
                out[f"TSMD {lv}"] = mdifft[lv] / std
    else:
        for lv in levels:
            out[f"TMD {lv}"] = mdifft[lv]

    if bin_var:
        for lv in levels:
            out[f"TKS {lv}"] = abs(mdifft[lv])
    elif not too_small:
        for lv in levels:
            mask_lv = (tr_np == lv)
            if focal is not None:
                mask_f = (tr_np == focal)
                out[f"TKS {lv}"] = tks_w(x[mask_lv], x[mask_f], weights[mask_lv], s_weights[mask_f])
            elif x_target is not None:
                tw = np.ones_like(x_target, dtype=float) if target_weights is None else target_weights
                out[f"TKS {lv}"] = tks_w(x[mask_lv], x_target, weights[mask_lv], tw)
            else:
                out[f"TKS {lv}"] = tks_w(x[mask_lv], x, weights[mask_lv], s_weights)

    return out


def distribution_one_var(x: np.ndarray, treat: pd.Categorical, weights: np.ndarray, s_weights: np.ndarray,
                         focal: Optional[str] = None, x_target: Optional[np.ndarray] = None,
                         target_weights: Optional[np.ndarray] = None, contrast: Optional[list[str]] = None) -> dict:
    levels = list(treat.categories)
    t1 = levels[1]
    x = np.asarray(x, dtype=float)
    s_weights = np.asarray(s_weights, dtype=float)
    weights = np.asarray(weights, dtype=float)

    out = {}
    if focal is not None:
        tr_np = pd.Series(treat).astype(object).to_numpy()
        out["Mean Target"] = mean_w(x, s_weights, mask=(tr_np == focal))
        out["SD Target"] = np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), s_weights, mask=(tr_np == focal)))
    elif x_target is not None:
        out["Mean Target"] = mean_w(x_target, target_weights)
        out["SD Target"] = (np.sqrt(var_w(x_target, np.all((x_target == 0) | (x_target == 1)), target_weights)) if (x_target.size > 1) else np.nan)
    else:
        out["Mean Target"] = mean_w(x, s_weights)
        out["SD Target"] = np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), s_weights))

    tr_np = pd.Series(treat).astype(object).to_numpy()
    out[f"Mean {levels[0]}"] = mean_w(x, weights, mask=(tr_np != t1))
    out[f"SD {levels[0]}"] = np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), weights, mask=(tr_np != t1)))
    out[f"Mean {levels[1]}"] = mean_w(x, weights, mask=(tr_np == t1))
    out[f"SD {levels[1]}"] = np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), weights, mask=(tr_np == t1)))

    return out


def distribution_one_var_multi(x: np.ndarray, treat: pd.Categorical, weights: np.ndarray, s_weights: np.ndarray,
                               focal: Optional[str] = None, x_target: Optional[np.ndarray] = None,
                               target_weights: Optional[np.ndarray] = None, contrast: Optional[list[str]] = None) -> dict:
    levels = list(treat.categories)
    x = np.asarray(x, dtype=float)
    s_weights = np.asarray(s_weights, dtype=float)
    weights = np.asarray(weights, dtype=float)

    out = {
        "Mean Target": mean_w(x_target, target_weights) if x_target is not None else (mean_w(x, s_weights) if focal is None else mean_w(x, s_weights, mask=(pd.Series(treat).astype(object).to_numpy() == focal))),
        "SD Target": (
            np.sqrt(var_w(x_target, np.all((x_target == 0) | (x_target == 1)), target_weights)) if (x_target is not None and x_target.size > 1)
            else (np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), s_weights, mask=(pd.Series(treat).astype(object).to_numpy() == focal))) if focal is not None
                  else np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), s_weights)))
        )
    }
    for lv in levels:
        mask_lv = (pd.Series(treat).astype(object).to_numpy() == lv)
        out[f"Mean {lv}"] = mean_w(x, weights, mask=mask_lv)
        out[f"SD {lv}"] = np.sqrt(var_w(x, np.all((x == 0) | (x == 1)), weights, mask=mask_lv))
    return out

# ---- Group sizes (ESS) ----

def _nn_binary(treat: pd.Categorical, weights: np.ndarray, base_weights: Optional[np.ndarray], s_weights: np.ndarray) -> pd.DataFrame:
    levels = list(treat.categories)[:2]
    t1 = levels[1]; t0 = levels[0]
    tr_np = pd.Series(treat).astype(object).to_numpy()
    if base_weights is None:
        rows = ["All", "Weighted"]
        data = np.zeros((2, 2), dtype=float)
        data[0, 0] = ESS(s_weights[tr_np == t0]); data[0, 1] = ESS(s_weights[tr_np == t1])
        data[1, 0] = ESS(weights[tr_np != t1]);  data[1, 1] = ESS(weights[tr_np == t1])
        return pd.DataFrame(data, index=rows, columns=levels[:2])
    else:
        bw = np.asarray(base_weights, dtype=float) * s_weights
        rows = ["All", "Base weighted", "Weighted"]
        data = np.zeros((3, 2), dtype=float)
        data[0, 0] = ESS(s_weights[tr_np == t0]);     data[0, 1] = ESS(s_weights[tr_np == t1])
        data[1, 0] = ESS(bw[tr_np == t0]);            data[1, 1] = ESS(bw[tr_np == t1])
        data[2, 0] = ESS(weights[tr_np != t1]);       data[2, 1] = ESS(weights[tr_np == t1])
        return pd.DataFrame(data, index=rows, columns=levels[:2])


def _nn_multi(treat: pd.Categorical, weights: np.ndarray, base_weights: Optional[np.ndarray], s_weights: np.ndarray) -> pd.DataFrame:
    levels = list(treat.categories)
    tr_np = pd.Series(treat).astype(object).to_numpy()
    if base_weights is None:
        rows = ["All", "Weighted"]
        data = np.vstack([
            [ESS(s_weights[tr_np == lv]) for lv in levels],
            [ESS(weights[tr_np == lv]) for lv in levels],
        ])
        return pd.DataFrame(data, index=rows, columns=levels)
    else:
        bw = np.asarray(base_weights, dtype=float) * s_weights
        rows = ["All", "Base weighted", "Weighted"]
        data = np.vstack([
            [ESS(s_weights[tr_np == lv]) for lv in levels],
            [ESS(bw[tr_np == lv]) for lv in levels],
            [ESS(weights[tr_np == lv]) for lv in levels],
        ])
        return pd.DataFrame(data, index=rows, columns=levels)


# ---- summary entry point ----

def summary_lmw(object: 'LMWResult', un: bool = True, addlvariables: Optional[Union[pd.DataFrame, list[str]]] = None,
                standardize: bool = True, data: Optional[pd.DataFrame] = None,
                stat: str = "balance", contrast: Optional[Union[list[str], list[int]]] = None) -> dict:
    stat = str(stat).lower()
    if stat not in ("balance", "distribution"):
        raise LMWInputError("stat must be 'balance' or 'distribution'")

    # Base covariate matrix (from modeling frame)
    if object.covs is None or (isinstance(object.covs, pd.DataFrame) and object.covs.shape[1] == 0):
        X = np.zeros((len(object.treat), 0), dtype=float); X_names = []
    else:
        X, X_names = covs_df_to_matrix(object.covs)

    # Add additional variables
    if addlvariables is not None:
        if isinstance(addlvariables, (list, tuple)) and all(isinstance(x, str) for x in addlvariables):
            if data is None:
                raise LMWInputError("if `addlvariables` is a list of names, you must supply `data`")
            extra = data[addlvariables]
            Xadd, Nadd = covs_df_to_matrix(extra)
        elif isinstance(addlvariables, pd.DataFrame):
            Xadd, Nadd = covs_df_to_matrix(addlvariables)
        else:
            raise LMWInputError("`addlvariables` must be a DataFrame or a list of column names")
        keep_idx = [j for j, n in enumerate(Nadd) if n not in set(X_names)]
        if keep_idx:
            X = np.hstack([X, Xadd[:, keep_idx]]) if X.size else Xadd[:, keep_idx]
            X_names.extend([Nadd[j] for j in keep_idx])

    # Target handling
    X_target = None; target_weights = None
    if object.target is not None:
        if isinstance(object.target, dict) and isinstance(object.target.get("target_original"), pd.DataFrame):
            X_target, X_target_names = covs_df_to_matrix(object.target["target_original"])
            target_weights = object.target.get("target_weights", None)
            miss = [n for n in X_names if n not in X_target_names]
            if miss:
                pad = np.full((X_target.shape[0], len(miss)), np.nan, dtype=float)
                X_target = np.hstack([X_target, pad])
        elif isinstance(object.target, pd.DataFrame):
            X_target, _ = covs_df_to_matrix(object.target)
        else:
            pass  # profile: 不需要逐列矩阵

    # Treatment & contrast
    treat = object.treat
    if contrast is not None:
        contrast_p = process_contrast(contrast, treat, object.method)
    else:
        contrast_p = object.contrast
    treat = apply_contrast_to_treat(treat, contrast_p)
    levels = list(treat.categories)
    K = len(levels)

    # Which per-variable function to use
    use_multi = (str(object.method).endswith("MRI")) and (contrast_p is None) and (K > 2)
    if stat == "balance":
        balance_fun = balance_one_var_multi if use_multi else balance_one_var
    else:
        distribution_fun = distribution_one_var_multi if use_multi else distribution_one_var

    focal = object.focal  # ATT/ATC 情形

    weights = np.asarray(object.weights, dtype=float)
    s_weights = np.ones_like(weights) if object.s_weights is None else np.asarray(object.s_weights, dtype=float)

    kk = X.shape[1]
    bal_un = bal_base = bal_w = None
    dist_un = dist_base = dist_w = None

    if kk > 0:
        names_clean = _strip_tics(list(X_names))
        if stat == "balance":
            if un:
                rows = []
                for j in range(kk):
                    x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                    if use_multi:
                        out = balance_fun(x, treat, s_weights, s_weights, True, focal, xt, target_weights)
                    else:
                        out = balance_fun(x, treat, s_weights, s_weights, standardize, focal, xt, target_weights)
                    rows.append(out)
                bal_un = pd.DataFrame(rows, index=names_clean)
                if object.base_weights is not None:
                    rows = []
                    bw = s_weights * np.asarray(object.base_weights, dtype=float)
                    for j in range(kk):
                        x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                        if use_multi:
                            out = balance_fun(x, treat, bw, s_weights, True, focal, xt, target_weights)
                        else:
                            out = balance_fun(x, treat, bw, s_weights, standardize, focal, xt, target_weights)
                        rows.append(out)
                    bal_base = pd.DataFrame(rows, index=names_clean)
            rows = []
            for j in range(kk):
                x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                out = balance_fun(x, treat, weights, s_weights, standardize, focal, xt, target_weights) if not use_multi \
                      else balance_fun(x, treat, weights, s_weights, True, focal, xt, target_weights)
                rows.append(out)
            bal_w = pd.DataFrame(rows, index=names_clean)
        else:
            if un:
                rows = []
                for j in range(kk):
                    x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                    out = distribution_fun(x, treat, s_weights, s_weights, focal, xt, target_weights, contrast=None)
                    rows.append(out)
                dist_un = pd.DataFrame(rows, index=names_clean)
                if object.base_weights is not None:
                    rows = []
                    bw = s_weights * np.asarray(object.base_weights, dtype=float)
                    for j in range(kk):
                        x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                        out = distribution_fun(x, treat, bw, s_weights, focal, xt, target_weights, contrast=None)
                        rows.append(out)
                    dist_base = pd.DataFrame(rows, index=names_clean)
            rows = []
            for j in range(kk):
                x = X[:, j]; xt = None if X_target is None else X_target[:, j]
                out = distribution_fun(x, treat, weights, s_weights, focal, xt, target_weights, contrast=None)
                rows.append(out)
            dist_w = pd.DataFrame(rows, index=names_clean)

    # Sample sizes
    if use_multi:
        nn_df = _nn_multi(treat, weights, object.base_weights, s_weights)
    else:
        nn_df = _nn_binary(treat, weights, object.base_weights, s_weights)

    return {
        "call": object.call,
        "nn": nn_df,
        "bal.un": bal_un,
        "bal.base.weighted": bal_base,
        "bal.weighted": bal_w,
        "dist.un": dist_un,
        "dist.base.weighted": dist_base,
        "dist.weighted": dist_w,
        "method": object.method.name if hasattr(object.method, "name") else str(object.method),
        "base.weights.origin": None,
    }

# Attach a convenience method if not present
if 'LMWResult' in globals() and not hasattr(LMWResult, 'summary'):
    def _lmw_summary(self, *args, **kwargs):
        return summary_lmw(self, *args, **kwargs)
    setattr(LMWResult, 'summary', _lmw_summary)

def print_summary(obj: dict, digits: int = 3):
    """模仿 R 的 print.summary.lmw：四舍五入并美化输出。"""
    from math import isnan

    def _fmt(df):
        if df is None:
            return None
        out = df.copy()
        for c in out.columns:
            out[c] = out[c].apply(lambda v: (f"{round(v, digits):.{digits}f}"
                                             if isinstance(v, (int, float)) and not isnan(v) else v))
        return out

    if obj.get("call"):
        print("\nCall:\n", obj["call"])

    if obj.get("bal.un") is not None:
        print("\nSummary of Balance for Unweighted Data:")
        print(_fmt(obj["bal.un"]))

    if obj.get("bal.base.weighted") is not None:
        print("\nSummary of Balance for Base Weighted Data:")
        print(_fmt(obj["bal.base.weighted"]))

    if obj.get("bal.weighted") is not None:
        print("\nSummary of Balance for Weighted Data:")
        print(_fmt(obj["bal.weighted"]))

    if obj.get("dist.un") is not None:
        print("\nDistribution Summary for Unweighted Data:")
        print(_fmt(obj["dist.un"]))

    if obj.get("dist.base.weighted") is not None:
        print("\nDistribution Summary for Base Weighted Data:")
        print(_fmt(obj["dist.base.weighted"]))

    if obj.get("dist.weighted") is not None:
        print("\nDistribution Summary for Weighted Data:")
        print(_fmt(obj["dist.weighted"]))

    if obj.get("nn") is not None:
        print("\nEffective Sample Sizes:")
        print(_fmt(obj["nn"]))


# ----------------------------
# Estimation layer – Python port of R/lmw_est.R (non-IV)
# ----------------------------

from dataclasses import dataclass

@dataclass
class LMWEstResult:
    coefficients: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    weights: np.ndarray            # weights used in the regression fit (w_fit)
    rank: int
    df_residual: int
    model_matrix: np.ndarray
    vcov: np.ndarray
    lmw_weights: np.ndarray        # implied regression weights from lmw()
    call: str
    estimand: Any
    focal: Any
    method: Any
    robust: str
    outcome: str
    treat_levels: list
    fixef: Any | None = None


def _parse_outcome_arg(outcome: Any | None, data: pd.DataFrame | None, formula: Any,
                       covs: pd.DataFrame | None, n: int) -> tuple[np.ndarray, str]:
    if outcome is None:
        y_name = None
        if isinstance(formula, str) and "~" in formula:
            lhs = formula.split("~", 1)[0].strip()
            y_name = lhs if lhs else None
        if y_name is None:
            raise LMWInputError("`outcome` 必须提供（列名或数组），因为公式没有左侧响应变量。")
        outcome = y_name

    if isinstance(outcome, str):
        df_src = data if (isinstance(data, pd.DataFrame)) else (covs if isinstance(covs, pd.DataFrame) else None)
        if df_src is None or outcome not in df_src.columns:
            raise LMWInputError(f"Outcome '{outcome}' 不在提供的数据中。")
        y = pd.to_numeric(df_src[outcome], errors="coerce").to_numpy(dtype=float)
        outcome_name = outcome
    else:
        y = np.asarray(outcome, dtype=float)
        outcome_name = "<array>"

    if y.shape[0] != n:
        raise LMWInputError("`outcome` 的长度与样本量不一致。")
    if not np.isfinite(y).all():
        raise LMWInputError("`outcome` 中存在非有限值。")
    return y, outcome_name


def _demean_matrix_by_fixef(X: np.ndarray, fixef: Any, w: np.ndarray) -> None:
    # 跳过截距列 0；与 R 保持一致
    if fixef is None:
        return
    for j in range(1, X.shape[1]):
        X[:, j] = _demean_by_group(X[:, j], fixef, w)


def _compute_wls_beta(A: np.ndarray, y_w: np.ndarray) -> tuple[np.ndarray, int]:
    beta, _, rank, _ = np.linalg.lstsq(A, y_w, rcond=None)
    return beta, int(rank)


def _subset_pos_w(y: np.ndarray, X: np.ndarray, resid: np.ndarray, w_fit: np.ndarray,
                  fixef: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any | None, np.ndarray]:
    pos = np.asarray(w_fit, dtype=float) > 0
    return y[pos], X[pos, :], resid[pos], w_fit[pos], (fixef[pos] if fixef is not None else None), pos


def _hat_diag_from_A(A: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import qr as scipy_qr
        Q, R = scipy_qr(A, mode="economic", pivoting=False)
        return np.sum(Q * Q, axis=1)
    except Exception:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        return np.sum(U * U, axis=1)


def _vcov_const(A: np.ndarray, resid_star: np.ndarray, df_resid: int) -> np.ndarray:
    # resid_star = sqrt(w) * residuals
    B = np.linalg.inv(A.T @ A)
    sigma2 = float(np.dot(resid_star, resid_star) / max(df_resid, 1))
    return B * sigma2


def _vcov_HC(A: np.ndarray, resid_star: np.ndarray, kind: str = "HC3") -> np.ndarray:
    n, p = A.shape
    B = np.linalg.inv(A.T @ A)
    e = resid_star.copy()
    if kind.upper() in ("HC2", "HC3"):
        h = _hat_diag_from_A(A)
        adj = 1.0 - np.clip(h, 0.0, 1.0 - 1e-12)
        if kind.upper() == "HC2":
            e = e / np.sqrt(adj)
        else:
            e = e / adj
    meat = (A.T * (e * e)) @ A
    if kind.upper() == "HC1":
        meat *= (n / max(n - p, 1))
    return B @ meat @ B


def _vcov_cluster(A: np.ndarray, resid_star: np.ndarray, clusters: np.ndarray,
                  small_sample: bool = True, base_kind: str = "HC1") -> np.ndarray:
    # CR1 型：V = B (sum_g Z_g Z_g') B，Z_g = A_g' e*_g
    n, p = A.shape
    B = np.linalg.inv(A.T @ A)
    e = resid_star.copy()
    if base_kind.upper() in ("HC2", "HC3"):
        h = _hat_diag_from_A(A)
        adj = 1.0 - np.clip(h, 0.0, 1.0 - 1e-12)
        e = e / (np.sqrt(adj) if base_kind.upper() == "HC2" else adj)

    meat = np.zeros((p, p), dtype=float)
    _, cl_ids = np.unique(clusters, return_inverse=True)
    G = int(cl_ids.max() + 1)
    for g in range(G):
        idx = (cl_ids == g)
        Ag = A[idx, :]
        eg = e[idx]
        Zg = Ag.T @ eg
        meat += np.outer(Zg, Zg)

    if small_sample and G > 1:
        meat *= (G / (G - 1)) * ((n - 1) / max(n - p, 1))

    return B @ meat @ B


def lmw_est(x: 'LMWResult',
            outcome: Any | None = None,
            data: pd.DataFrame | None = None,
            robust: bool | str = True,
            cluster: Any | None = None,
            **kwargs) -> LMWEstResult:
    """Estimate the outcome regression consistent with the lmw design.

    - WLS on the centered/expanded design used by `x`.
    - Covariance: 'const', HC0/HC1/HC2/HC3, or cluster-robust (CR1).
    - If AIPW, fit excludes base weights; variance uses HC0（与 R 注释一致的近似）.
    """
    # 数据来源
    data0 = data if (isinstance(data, pd.DataFrame)) else (x.covs if isinstance(x.covs, pd.DataFrame) else None)
    if data0 is None:
        raise LMWInputError("`data` 必须提供或在 `x.covs` 中可用。")

    n = len(x.treat)

    # 1) 与 lmw 一致的设计矩阵（居中/交互）
    treat_contrast = apply_contrast_to_treat(x.treat, x.contrast)
    tname = (getattr(x.treat, 'name', None) or 'treat')
    table_backend = PandasBackend(data0.assign(**{tname: pd.Series(x.treat).astype(object)}))
    X_obj = get_X_from_formula(
        x.formula, table_backend, treat_contrast, x.method, x.estimand,
        x.target, x.s_weights, None, x.focal,
        engine="formulaic", ensure_full_rank=True, output="numpy",
        treat_name=tname,
    )
    X = np.asarray(X_obj.X, dtype=float)

    # 2) 结局向量
    y, outcome_name = _parse_outcome_arg(outcome, data0, x.formula, x.covs, n)

    # 3) 拟合权重
    s_w = np.ones(n, dtype=float) if x.s_weights is None else np.asarray(x.s_weights, dtype=float).reshape(-1)
    b_w = np.ones(n, dtype=float) if x.base_weights is None else np.asarray(x.base_weights, dtype=float).reshape(-1)
    if s_w.shape[0] != n or b_w.shape[0] != n:
        raise LMWInputError("s.weights 或 base.weights 长度不匹配。")

    if hasattr(x, 'dr_method') and str(x.dr_method).upper() == 'AIPW':
        w_fit = s_w.copy()      # AIPW：拟合不乘 base.weights
        robust_eff = 'HC0'      # R：AIPW 用 M-估计；这里按 HC0 近似
    else:
        w_fit = s_w * b_w
        robust_eff = None

    # 4) 固定效应的带权去均值（R 在有 fixef 时会对 X 和 y 去均值）
    if x.fixef is not None:
        _demean_matrix_by_fixef(X, x.fixef, w_fit)
        y = _demean_by_group(y, x.fixef, w_fit)

    # 5) WLS（对 A=√w X，y*=√w y 做 OLS）
    rw = np.sqrt(w_fit)
    A = rw[:, None] * X
    y_w = rw * y
    beta, rank = _compute_wls_beta(A, y_w)
    fitted = X @ beta
    resid = y - fitted

    # 6) 仅对正权重样本做协方差（与 R 的 subset_fit 一致）
    y_pos, X_pos, resid_pos, w_pos, fixef_pos, pos_mask = _subset_pos_w(y, X, resid, w_fit, x.fixef)
    A_pos = np.sqrt(w_pos)[:, None] * X_pos
    p = X_pos.shape[1]
    df_resid = int(max(y_pos.shape[0] - rank, 0))

    # FE 自由度修正（R: df.resid 减去组数-1）
    if x.fixef is not None:
        try:
            fe_groups = pd.Series(x.fixef).to_numpy()[pos_mask]
            df_resid -= (len(np.unique(fe_groups)) - 1)
        except Exception:
            pass

    # 7) 选择稳健类型 / 聚类
    if robust_eff is not None:
        robust_type = robust_eff
        cluster_arr = None
    else:
        if robust is True:
            robust_type = 'HC3' if cluster is None else 'HC1'
        elif robust is False:
            robust_type = 'const' if cluster is None else 'HC1'
        elif isinstance(robust, str):
            robust_type = robust.upper()
        else:
            raise LMWInputError("`robust` 必须是 True/False 或 'const'/'HC0'/'HC1'/'HC2'/'HC3'。")

        cluster_arr = None
        if cluster is not None:
            # 允许右侧公式字符串（~subclass）、Series/DataFrame/array
            if isinstance(cluster, str) and cluster.strip().startswith('~'):
                try:
                    import patsy
                    cl_df = patsy.dmatrix(cluster, data0, return_type='dataframe')
                except Exception:
                    raise LMWInputError("无法解析 `cluster` 公式；请直接传入列名或数组。")
                if 'Intercept' in cl_df.columns:
                    cl_df = cl_df.drop(columns=['Intercept'])
                cluster_arr = cl_df.to_numpy()
            elif isinstance(cluster, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
                cluster_arr = np.asarray(cluster)
                if cluster_arr.ndim == 1:
                    cluster_arr = cluster_arr.reshape(-1, 1)
            else:
                raise LMWInputError("`cluster` 必须是数组/Series/DataFrame 或右侧公式字符串。")

            # 与 pos_mask 对齐
            if cluster_arr.shape[0] == data0.shape[0]:
                cluster_arr = cluster_arr[pos_mask, ...]
            elif cluster_arr.shape[0] != np.sum(pos_mask):
                raise LMWInputError("`cluster` 的行数必须与原始数据相同或与正权重样本数相同。")

    # 8) 协方差矩阵
    if cluster_arr is None:
        if robust_type == 'CONST':
            vc = _vcov_const(A_pos, np.sqrt(w_pos) * resid_pos, df_resid)
        elif robust_type in {'HC0', 'HC1', 'HC2', 'HC3'}:
            vc = _vcov_HC(A_pos, np.sqrt(w_pos) * resid_pos, kind=robust_type)
        else:
            raise LMWInputError(f"未知的 `robust` 类型: {robust_type}")
    else:
        if cluster_arr.ndim == 2 and cluster_arr.shape[1] > 1:
            combined = pd.util.hash_pandas_object(pd.DataFrame(cluster_arr)).to_numpy()
        else:
            combined = cluster_arr.reshape(-1)
        vc = _vcov_cluster(A_pos, np.sqrt(w_pos) * resid_pos, combined,
                           small_sample=True, base_kind=robust_type)
        # 聚类情形：df.residual 取“最少簇数-1”
        try:
            df_resid = int(len(np.unique(combined)) - 1)
        except Exception:
            pass

    # FE + 常规/HC1 的 df 缩放（R 的修正）
    if (x.fixef is not None) and (robust_type in {'CONST', 'HC1'}):
        n_pos = int(np.sum(pos_mask))
        vc *= ((n_pos - p) / max(df_resid, 1))

    return LMWEstResult(
        coefficients=beta.reshape(-1),
        residuals=resid,
        fitted_values=fitted,
        weights=w_fit,
        rank=rank,
        df_residual=int(df_resid),
        model_matrix=X,
        vcov=vc,
        lmw_weights=np.asarray(x.weights, dtype=float),
        call="lmw_est(x, outcome=..., data=..., robust=..., cluster=...)",
        estimand=x.estimand,
        focal=x.focal,
        method=x.method,
        robust=robust_type,
        outcome=outcome_name,
        treat_levels=list(pd.Categorical(x.treat).categories),
        fixef=(x.fixef if x.fixef is not None else None),
    )

def print_lmw_est(obj: LMWEstResult) -> None:
    print(f"An LMWEstResult object")
    print(f" - outcome: {obj.outcome}")
    print(f" - standard errors: {obj.robust}")
    print(f" - estimand: {obj.estimand}")
    print(f" - method: {obj.method}")
    if obj.fixef is not None:
        print(" - fixed effects: provided")

    se = np.sqrt(np.diag(obj.vcov))
    z = obj.coefficients / np.where(se > 0, se, np.nan)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1 + erf(np.abs(z) / sqrt(2))))  # 正态近似双侧 p

    import pandas as _pd
    df = _pd.DataFrame({'coef': obj.coefficients, 'se': se, 'z': z, 'p': p})
    print("\nCoefficients:\n", df)

# ----------------------------
# Influence – Python port of R/influence.lmw.R
# ----------------------------

def influence_lmw(x: 'LMWResult',
                  outcome: Any | None = None,
                  data: pd.DataFrame | None = None) -> dict:
    """
    Python port of R's influence.lmw():
    Returns SIC (raw & scaled), leverage (h), residuals (r), and metadata used.
    SIC = (N - 1) * w_impl * r / (1 - h)
    """
    n = len(x.treat)

    # --- outcome vector y ---
    if outcome is None:
        y_name = None
        if isinstance(x.formula, str) and "~" in x.formula:
            lhs = x.formula.split("~", 1)[0].strip()
            y_name = lhs if lhs else None
        if y_name is None:
            raise LMWInputError("`outcome` 必须提供（列名或数组），因为公式没有左侧响应变量。")
        outcome = y_name

    if isinstance(outcome, str):
        df_src = data if (isinstance(data, pd.DataFrame)) \
            else (x.covs if isinstance(x.covs, pd.DataFrame) else None)
        if df_src is None or outcome not in df_src.columns:
            raise LMWInputError(f"Outcome '{outcome}' 不在提供的数据中。")
        y = pd.to_numeric(df_src[outcome], errors="coerce").to_numpy(dtype=float)
    else:
        y = np.asarray(outcome, dtype=float)

    if y.shape[0] != n:
        raise LMWInputError("`outcome` 的长度与样本量不一致。")
    if not np.isfinite(y).all():
        raise LMWInputError("`outcome` 中存在非有限值。")

    # --- rebuild design X consistent with lmw ---
    treat = x.treat
    treat_contrast = apply_contrast_to_treat(treat, x.contrast)

    if isinstance(x.covs, pd.DataFrame):
        df_table = x.covs.copy()
    else:
        if not isinstance(data, pd.DataFrame):
            raise LMWInputError("需要 `data` (DataFrame) 来重建 design。")
        df_table = data.copy()

    treat_name = getattr(treat, "name", None) or "treat"
    df_table[treat_name] = pd.Series(treat).astype(object)
    table_backend = PandasBackend(df_table)

    X_obj = get_X_from_formula(
        x.formula, table_backend, treat_contrast, x.method, x.estimand,
        x.target, x.s_weights, None, x.focal,
        engine="formulaic", ensure_full_rank=True, output="numpy",
        treat_name=treat_name,
    )
    X = np.asarray(X_obj.X, dtype=float)

    # --- WLS residuals & leverage on A = sqrt(s_w)*X ---
    s_w = np.ones(n, dtype=float) if x.s_weights is None \
        else np.asarray(x.s_weights, dtype=float).reshape(-1)
    if s_w.shape[0] != n:
        raise LMWInputError("s.weights 长度与样本量不一致。")

    rw = np.sqrt(s_w)
    A = rw[:, None] * X
    y_w = rw * y

    beta, *_ = np.linalg.lstsq(A, y_w, rcond=None)
    resid = y - (X @ beta)

    try:
        from scipy.linalg import qr as scipy_qr
        Q, R = scipy_qr(A, mode="economic", pivoting=False)
        h = np.sum(Q * Q, axis=1)
    except Exception:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        h = np.sum(U * U, axis=1)

    # --- SIC ---
    w_impl = np.asarray(x.weights, dtype=float).reshape(-1)
    if w_impl.shape[0] != n:
        raise LMWInputError("`x.weights` 长度与样本量不一致。")

    denom = 1.0 - np.clip(h, 0.0, 1.0 - 1e-12)
    sic_raw = (n - 1.0) * w_impl * resid / denom
    sic_abs = np.abs(sic_raw)
    max_sic = float(np.max(sic_abs)) if sic_abs.size else 1.0
    sic_std = sic_abs / (max_sic if max_sic > 0 else 1.0)

    return {
        "sic": sic_std,
        "sic_raw": sic_raw,
        "leverage": h,
        "resid": resid,
        "beta": beta,
        "X": X,
        "y": y,
        "s_w": s_w,
    }



# ----------------------------
# Plotting (matplotlib) – Python port of R/plot.lmw.R
# ----------------------------

from typing import Iterable



def _label_for_level(lv: Any, tlevs: Sequence[str]) -> str:
    s = str(lv)
    if len(tlevs) == 2 and set(map(str, tlevs)).issubset({"0", "1"}):
        return {"0": "Control", "1": "Treated"}.get(s, s)
    return s

def _kde1d(x: np.ndarray, grid_n: int = 512, bw: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight Gaussian KDE (no SciPy requirement).
    Returns (grid_x, density_y). Handles constant vectors by using a tiny bandwidth around the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(0.0, 1.0, 2), np.zeros(2)
    mu = float(x.mean()); sd = float(x.std(ddof=0))
    if sd < np.sqrt(np.finfo(float).eps):
        h = (abs(mu) * 1e-4) + 1e-6 if bw is None else float(bw)
        xs = np.linspace(max(0.0, mu - 2*h), mu + 2*h, grid_n)
        ys = (1.0/(np.sqrt(2*np.pi)*h)) * np.exp(-0.5*((xs - mu)/h)**2)
        ys /= np.trapz(ys, xs) if np.trapz(ys, xs) != 0 else 1.0
        return xs, ys
    if bw is None:
        h = 1.06 * sd * (x.size ** (-1/5))
    else:
        h = float(bw)
    xs = np.linspace(x.min() - 0.1*sd, x.max() + 0.1*sd, grid_n)
    diffs = (xs[:, None] - x[None, :]) / h
    ys = np.exp(-0.5 * diffs * diffs).sum(axis=1) / (x.size * h * np.sqrt(2*np.pi))
    return xs, ys

def _variables_to_matrix_for_plot(variables: Any, data: pd.DataFrame, fallback_covs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mimic R in plot(extrapolation):
    - `variables`: right-sided formula string (e.g., "~ age + married") or list[str]
    - `data`: preferred df to pull raw columns; else fallback to `fallback_covs`
    Returns DataFrame with expanded columns (categoricals one-hot) and their names.
    """
    if variables is None:
        raise LMWInputError("`variables` must be provided for type='extrapolation'")

    # list of names
    if isinstance(variables, (list, tuple)) and all(isinstance(v, str) for v in variables):
        if data is None or not isinstance(data, pd.DataFrame):
            raise LMWInputError("When `variables` is a list of names, `data` must be supplied as a DataFrame")
        for v in variables:
            if v not in data.columns:
                raise LMWInputError(f"variable {v!r} not found in `data`")
        X, names = covs_df_to_matrix(data[variables])
        return pd.DataFrame(X, columns=names), names

    # formula-like
    if isinstance(variables, str):
        f = variables.strip()
        if not f.startswith("~"):
            f = "~ " + f
        try:
            import patsy
            mf = patsy.dmatrix(f, (data if data is not None else fallback_covs), return_type="dataframe", NA_action="raise")
            if "Intercept" in mf.columns:
                mf = mf.drop(columns=["Intercept"])
            X, names = covs_df_to_matrix(mf)
            return pd.DataFrame(X, columns=names), names
        except Exception:
            tokens = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f)))
            pool = data if data is not None else fallback_covs
            missing = [t for t in tokens if t not in pool.columns]
            if missing:
                raise LMWInputError("variables not found in data: " + ", ".join(missing))
            X, names = covs_df_to_matrix(pool[tokens])
            return pd.DataFrame(X, columns=names), names

    raise LMWInputError("`variables` must be a right-sided formula string or a list of column names")

def plot_lmw(x: 'LMWResult', type: str = "weights", **kwargs) -> 'LMWResult':
    """Matplotlib port of R's plot.lmw().
    type in {'weights','extrapolation','influence'}.
      - weights: rug=True, mean=True, ess=True, bw=None
      - extrapolation: variables=..., data=None
      - influence: outcome=None, data=None, id_n=3  (currently NotImplemented)
    """
    t = str(type).lower()
    if t not in {"weights", "extrapolation", "influence"}:
        raise LMWInputError("type must be one of 'weights', 'extrapolation', 'influence'")
    if t == "weights":
        return _plot_weights_lmw(x, **kwargs)
    elif t == "extrapolation":
        return _plot_extrapolation_lmw(x, **kwargs)
    else:
        return _plot_influence_lmw(x, **kwargs)

def _plot_weights_lmw(x: 'LMWResult', rug: bool = True, mean: bool = True, ess: bool = True, bw: float | None = None, **kwargs) -> 'LMWResult':
    import matplotlib.pyplot as plt

    t = x.treat
    w = np.asarray(x.weights, dtype=float)
    sw = np.ones_like(w) if x.s_weights is None else np.asarray(x.s_weights, dtype=float)

    tlevs = list(x.contrast) if x.contrast is not None else list(t.categories)

    fig, axes = plt.subplots(nrows=len(tlevs), ncols=1, figsize=(6, 2.5*len(tlevs)), squeeze=False)

    t_np = pd.Series(t).astype(object).to_numpy()
    for row, i in enumerate(tlevs):
        if len(tlevs) == 2 and i != tlevs[1] and str(x.method) == str(Method.URI):
            idx = (t_np != tlevs[1])
        else:
            idx = (t_np == i)
        wi = w[idx]

        xs, ys = _kde1d(wi, bw=bw)
        xlim = (xs.min(), xs.max() + (0.1 * (xs.max() - xs.min()) if ess else 0.0))

        ax = axes[row, 0]
        ax.plot(xs, ys)
        ax.set_ylabel("Density")
        ax.set_xlabel("Weight")
        ax.set_xlim(xlim)
        ax.set_title(f"Distribution of Weights ({_label_for_level(i, tlevs)})", fontsize=10)
        ax.set_ylim(0, max(ys) * 1.05 if ys.size else 1)

        if rug and wi.size:
            ax.plot(wi, np.zeros_like(wi), "|", markersize=6)
        if mean and wi.size:
            ax.axvline(wi.mean(), color="red", linewidth=1)
        if ess:
            e = ESS(wi)
            e_un = ESS(sw[t_np == i])
            ax.text(0.98, 0.95, f"N = {e_un:.1f}\nESS = {e:.1f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9)

    fig.tight_layout()
    plt.show()
    return x

def _plot_extrapolation_lmw(x: 'LMWResult', variables: Any, data: pd.DataFrame | None = None, **kwargs) -> 'LMWResult':
    import matplotlib.pyplot as plt

    t = x.treat
    w = np.asarray(x.weights, dtype=float)
    tlevs = list(x.contrast) if x.contrast is not None else list(t.categories)

    data0 = data if (data is not None and isinstance(data, pd.DataFrame)) else x.covs
    if data0 is None:
        raise LMWInputError("`data` is required or `x.covs` must be available for extrapolation plots")

    V_df, V_names = _variables_to_matrix_for_plot(variables, data0, x.covs if isinstance(x.covs, pd.DataFrame) else data0)

    # Target mean selection
    X_target_means: dict[str, float] | None = None
    if x.focal is not None:
        tr_np = pd.Series(t).astype(object).to_numpy()
        mask_f = (tr_np == x.focal)
        sw = np.ones_like(w) if x.s_weights is None else np.asarray(x.s_weights, dtype=float)
        X_target_means = {nm: float(mean_w(V_df[nm].to_numpy(), sw, mask=mask_f)) for nm in V_names}
    elif isinstance(x.target, dict) and isinstance(x.target.get("target_original"), pd.DataFrame):
        Xt_df = x.target["target_original"]
        Xt_mat, Xt_names = covs_df_to_matrix(Xt_df)
        Xt_means = Xt_mat.mean(axis=0) if x.target.get("target_weights") is None else _colmeans_w(Xt_mat, x.target["target_weights"])
        X_target_means = {n: float(Xt_means[list(Xt_names).index(n)]) if n in Xt_names else np.nan for n in V_names}
    else:
        sw = np.ones_like(w) if x.s_weights is None else np.asarray(x.s_weights, dtype=float)
        X_target_means = {nm: float(mean_w(V_df[nm].to_numpy(), sw)) for nm in V_names}

    K = len(tlevs)
    fig, axes = plt.subplots(nrows=1, ncols=len(V_names), figsize=(4*len(V_names), 2.5*K), squeeze=False)
    codes = pd.Categorical(t).codes  # 0..K-1

    # point sizes
    t_np = pd.Series(t).astype(object).to_numpy()
    cex = np.zeros_like(w)
    for lv in tlevs:
        if len(tlevs) == 2 and lv != tlevs[1] and str(x.method) == str(Method.URI):
            in_i = (t_np != tlevs[1])
        else:
            in_i = (t_np == lv)
        denom = np.sum(w[in_i]) if np.sum(in_i) else 1.0
        cex[in_i] = 30.0 * np.sqrt(np.abs(w[in_i]) / max(denom, np.finfo(float).eps))

    # transparency by group
    alpha = {}
    for lv in tlevs:
        in_i = (t_np == lv)
        nz = np.sum(np.abs(w[in_i]) > 1e-8)
        a = min(1.0, 0.6 / (np.log10(nz) if nz > 1 else 1.0))
        alpha[str(lv)] = a

    # 用 (n,4) 的 RGBA 浮点数组，彻底避免布尔索引标量/元组广播问题
    cols = np.zeros((len(t), 4), dtype=float)
    for lv in tlevs:
        mask = (t_np == lv)
        a = alpha[str(lv)]
        cols[mask & (w >= 0), :] = (0.0, 0.0, 0.0, a)  # black with alpha
        cols[mask & (w < 0), :] = (1.0, 0.0, 0.0, a)  # red with alpha

    for j, name in enumerate(V_names):
        # 1) 用原始变量计算均值
        vj_raw = V_df[name].to_numpy(dtype=float)

        means_vj = []
        for lv in tlevs:
            if len(tlevs) == 2 and lv != tlevs[1] and str(x.method) == str(Method.URI):
                in_i = (t_np != tlevs[1])
            else:
                in_i = (t_np == lv)
            means_vj.append(mean_w(vj_raw, w, mask=in_i))

        mu_t = X_target_means.get(name, np.nan)

        # 2) 仅用于画点：二值变量加轻微抖动
        vj = vj_raw.copy()
        if np.all((vj == 0) | (vj == 1)):
            vj += np.random.uniform(-0.02, 0.02, size=vj.shape[0])


        # 3) 画散点（用 vj），竖线和 X（用 means_vj / mu_t）
        jitter = np.random.uniform(-0.1, 0.1, size=len(t))
        y = 0.5 + 2 * (K - (codes + 1)) + 1 * (w >= 0) + jitter
        ax = axes[0, j]
        ax.scatter(vj, y, s=(cex ** 2), c=cols, marker='o')




        # separators & vertical weighted means
        for g in range(1, K):
            ax.axhline(y=2*g, color=(0,0,0,0.3), linewidth=1)


        for i, mu in enumerate(means_vj, start=1):
            ax.vlines(mu, 2*(K - i + 1), 2*(K - i), colors='k', linewidth=1)

        # x 轴范围最好也基于原始变量
        rng = (vj_raw.min(), vj_raw.max())
        pad = 0.025 * (rng[1] - rng[0] if rng[1] > rng[0] else 1.0)
        ax.set_xlim(min(rng[0], mu_t, *means_vj) - pad,
                    max(rng[1], mu_t, *means_vj) + pad)

        ax.plot([mu_t]*K, 2*np.arange(1, K+1) - 1, marker='x', linestyle='None', markersize=6, color='k')

        ax.set_xlabel(name)
        ax.set_ylim(0, 2*K)
        ax.set_yticks(2*np.arange(K, 0, -1) - 1)
        ax.set_yticklabels([_label_for_level(lv, tlevs) for lv in tlevs])
        ax.set_title("Extrapolation / Representativeness", fontsize=10)
        ax.grid(False)

    fig.tight_layout()
    plt.show()
    return x


def _plot_influence_lmw(x: 'LMWResult',
                        outcome: Any | None = None,
                        data: pd.DataFrame | None = None,
                        id_n: int = 3,
                        **kwargs) -> 'LMWResult':
    """Plot the Sample Influence Curve (SIC) using influence_lmw()."""
    import matplotlib.pyplot as plt

    res = influence_lmw(x, outcome=outcome, data=data)
    sic_std = np.asarray(res["sic"], dtype=float)
    n = sic_std.shape[0]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(1, n + 1)
    ax.vlines(xs, 0.0, sic_std, colors=(0, 0, 0), linewidth=1)

    ax.set_xlabel("Obs. number")
    ax.set_ylabel("Scaled SIC")
    ax.set_ylim(0.0, 1.075)

    if n <= 10:
        xticks = list(range(1, n + 1))
    else:
        grid = np.linspace(1, n, num=5)
        xticks = sorted(set([1, n] + [int(round(v)) for v in grid[1:-1]]))
    ax.set_xticks(xticks)

    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Sample Influence Curve")

    if id_n and id_n > 0:
        idx_top = np.argsort(sic_std)[-int(id_n):]
        for i in idx_top:
            x_i, y_i = xs[i], sic_std[i]
            ha = 'left' if x_i > (n / 2) else 'right'
            dx = 0.25 if ha == 'left' else -0.25
            ax.text(x_i + dx, y_i, str(x_i), fontsize=8, ha=ha, va='bottom')

    fig.tight_layout()
    plt.show()
    return x


# ----------------------------
# Love plot for summary objects – Python port of R/plot.summary.lmw.R
# ----------------------------

def _rename_summary_stat_py(col_name: str, abs_value: bool = False) -> str:
    """Mimic R's rename_summary_stat() for axis titles.
    - If endswith KS → "KS statistic"
    - If endswith SMD and abs → replace "SMD" with "ASMD"
    - If has a group token after space, render as "<stat> (<group>)" and map treated/control wording.
    """
    name = str(col_name)
    parts = name.split(" ")
    stat = parts[0]
    if stat.endswith("KS"):
        stat = f"{stat} statistic"
    elif stat.endswith("SMD") and abs_value:
        stat = stat.replace("SMD", "ASMD")
    if len(parts) == 1:
        return stat
    group = " ".join(parts[1:])
    if group.lower() in ("treated", "control"):
        group = f"{group.lower()} group"
    return f"{stat} ({group})"


def plot_summary_lmw(x: dict,
                     stats: list[str] | None = None,
                     abs: bool = True,
                     var_order: str = "data",
                     threshold: list[float] | float | None = None,
                     layout: str = "vertical",
                     **kwargs) -> dict:
    """Love plot for a `summary_lmw()` result.

    Parameters
    ----------
    x : dict
        The object returned by `summary_lmw(...)`.
    stats : list[str] | None
        Column names (or their prefixes) to plot. Defaults to all columns starting with "TSMD".
    abs : bool
        Plot absolute values (KS unaffected). Default True.
    var_order : {"data","alphabetical","unadjusted"}
        Order variables as in summary, alphabetically, or by the first selected stat in the unadjusted table.
    threshold : list[float] | float | None
        One or more x-location(s) to draw vertical threshold line(s). If `abs=False`, lines are drawn at ±value.
    layout : {"vertical","horizontal"}
        Arrange multiple stats vertically (rows) or horizontally (columns).

    Returns
    -------
    dict
        Returns `x` invisibly (for chaining), matching R's behavior.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Presence flags and pull tables
    un_tbl   = x.get("bal.un", None)
    base_tbl = x.get("bal.base.weighted", None)
    w_tbl    = x.get("bal.weighted", None)

    if un_tbl is None and base_tbl is None and w_tbl is None:
        raise LMWInputError("plot_summary_lmw() requires a summary object with balance tables; run summary_lmw(stat='balance').")

    # Determine candidate columns from any available table
    def _cols(df):
        return list(df.columns) if df is not None else []

    all_cols = list(dict.fromkeys(_cols(un_tbl) + _cols(base_tbl) + _cols(w_tbl)))
    if not all_cols:
        raise LMWInputError("summary object has no balance columns to plot.")

    # --- Friendly alias mapping for stats like "TSMD Treated" / "TSMD Control" ---
    levels_from_nn = None
    try:
        nn_tbl = x.get("nn", None)
        if nn_tbl is not None and hasattr(nn_tbl, "columns"):
            levels_from_nn = [str(c) for c in nn_tbl.columns]  # e.g., ["0","1"] or ["Control","Treated"]
    except Exception:
        levels_from_nn = None

    def _normalize_stat_request(s_req: str) -> str:
        """Map friendly tokens (e.g., 'TSMD Treated') to actual column names like 'TSMD 1'.
        Works case-insensitively; leaves input unchanged if mapping info is unavailable.
        """
        import re as _re
        s0 = _re.sub(r"\s+", " ", str(s_req).strip())
        s_l = s0.lower()
        if levels_from_nn is not None and len(levels_from_nn) >= 2:
            treated_aliases = {"treated", "treat", "t1"}
            control_aliases = {"control", "ctrl", "t0"}
            tokens = s_l.split(" ")
            if any(tok in tokens for tok in treated_aliases):
                parts = s0.split(" ")
                for idx in range(len(parts) - 1, -1, -1):
                    if parts[idx].lower() in treated_aliases:
                        parts[idx] = levels_from_nn[1]
                        break
                else:
                    parts.append(levels_from_nn[1])
                return " ".join(parts)
            if any(tok in tokens for tok in control_aliases):
                parts = s0.split(" ")
                for idx in range(len(parts) - 1, -1, -1):
                    if parts[idx].lower() in control_aliases:
                        parts[idx] = levels_from_nn[0]
                        break
                else:
                    parts.append(levels_from_nn[0])
                return " ".join(parts)
        return s0

    # Default stats: columns starting with TSMD
    if stats is None:
        chosen = [c for c in all_cols if str(c).upper().startswith("TSMD")] or all_cols
    else:
        # Normalize requests (e.g., 'TSMD Treated' → 'TSMD <treated_level>' if possible)
        req_list = stats if isinstance(stats, (list, tuple)) else [stats]
        norm_list = [_normalize_stat_request(s) for s in req_list]
        # Build a lowercase lookup for available columns
        all_cols_lc = {str(c).lower(): str(c) for c in all_cols}
        chosen = []
        for s in norm_list:
            s_lc = str(s).lower()
            # exact match first
            if s_lc in all_cols_lc:
                chosen.append(all_cols_lc[s_lc])
                continue
            # otherwise prefix match
            matches = [c for c in all_cols if str(c).lower().startswith(s_lc)]
            if not matches:
                raise LMWInputError(
                    f"stat column not found (or prefix didn't match): {s_lc!r}. "
                    f"Available: {', '.join(map(str, all_cols))}"
                )
            chosen.extend(matches)
        # de-duplicate keep order
        chosen = list(dict.fromkeys(chosen))

    # Subset tables to chosen stats
    stats_un   = un_tbl[chosen]   if un_tbl   is not None else None
    stats_base = base_tbl[chosen] if base_tbl is not None else None
    stats_w    = w_tbl[chosen]    if w_tbl    is not None else None

    # Absolute value option (KS unaffected by sign, but we follow R: apply abs to all selected columns)
    if abs:
        if stats_un   is not None: stats_un   = stats_un.abs()
        if stats_base is not None: stats_base = stats_base.abs()
        if stats_w    is not None: stats_w    = stats_w.abs()

    # Variable names and ordering
    # Prefer the index from the weighted table; fall back to others
    base_index = None
    for df in (stats_w, stats_base, stats_un):
        if df is not None:
            base_index = list(df.index)
            break
    if base_index is None:
        raise LMWInputError("could not infer variable names (empty balance tables)")

    var_order = str(var_order).lower()
    if var_order not in {"data", "alphabetical", "unadjusted"}:
        raise LMWInputError("var_order must be one of 'data', 'alphabetical', 'unadjusted'")

    if var_order == "data":
        ord_idx = list(range(len(base_index)))[::-1]  # reverse like R's rev()
    elif var_order == "alphabetical":
        ord_idx = list(np.argsort(np.asarray(base_index))[::-1])
    else:  # "unadjusted"
        if stats_un is None:
            raise LMWInputError("var_order='unadjusted' requires `un=True` in summary_lmw (bal.un present)")
        first_stat = chosen[0]
        ord_idx = list(np.argsort(stats_un[first_stat].to_numpy()))

    # X-limits computed across selected stats and available tables (imitate R)
    def _collect_vals(df_list):
        vals = []
        for df in df_list:
            if df is not None and df.size:
                vals.append(df.to_numpy().ravel())
        return np.concatenate(vals) if vals else np.array([0.0])

    vals_all = _collect_vals([stats_un, stats_base, stats_w])
    minx = float(np.nanmin(vals_all)) if vals_all.size else (0.0 if abs else -0.01)
    maxx = float(np.nanmax(vals_all)) if vals_all.size else 0.01
    if abs:
        minx = min(minx, 0.0)
    else:
        minx = min(minx, -0.01)
        maxx = max(maxx, 0.01)
    xlim = (minx, 1.75*maxx - minx)

    # Layout
    chosen_labels = [_rename_summary_stat_py(c, abs_value=abs) for c in chosen]
    layout = str(layout).lower()
    if layout not in {"vertical", "horizontal"}:
        raise LMWInputError("layout must be 'vertical' or 'horizontal'")

    n_panels = len(chosen)
    if layout == "vertical":
        fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(7, 2.2*n_panels), squeeze=False)
        axes = [axes[i, 0] for i in range(n_panels)]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=n_panels, figsize=(7*n_panels, 3.2), squeeze=False)
        axes = [axes[0, j] for j in range(n_panels)]

    # Legend text
    method = x.get("method", "")
    origin = x.get("base.weights.origin", None)
    if stats_base is not None:
        if origin == "MatchIt":
            legend_text = [
                f"Before matching" if stats_un is not None else None,
                f"After matching",
                f"After matching + \n{method} regression" if stats_w is not None else None,
            ]
        elif origin == "WeightIt":
            legend_text = [
                f"Before weighting" if stats_un is not None else None,
                f"After weighting",
                f"After weighting + \n{method} regression" if stats_w is not None else None,
            ]
        else:
            legend_text = [
                f"Before base weighting" if stats_un is not None else None,
                f"After base weighting",
                f"After base weighting + \n{method} regression" if stats_w is not None else None,
            ]
    else:
        legend_text = [
            f"Before regression" if stats_un is not None else None,
            f"After {method} regression" if stats_w is not None else None,
        ]
    legend_text = [s for s in legend_text if s is not None]

    # Draw per-panel dot charts
    for ax, stat_name, xlabel in zip(axes, chosen, chosen_labels):
        y_labels = [base_index[i] for i in ord_idx]
        y_pos = np.arange(1, len(y_labels) + 1)

        ax.set_xlim(xlim)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(xlabel)
        ax.axvline(0.0, linewidth=1)

        # Plot points: unadjusted (x), base-weighted (o, empty), weighted (o, filled)
        if stats_un is not None:
            ax.plot(stats_un[stat_name].to_numpy()[ord_idx], y_pos, linestyle='None', marker='x', markersize=5)
        if stats_base is not None:
            ax.plot(stats_base[stat_name].to_numpy()[ord_idx], y_pos, linestyle='None', marker='o', markersize=6, fillstyle='none')
        if stats_w is not None:
            ax.plot(stats_w[stat_name].to_numpy()[ord_idx], y_pos, linestyle='None', marker='o', markersize=6)

        # Threshold lines
        if threshold is not None:
            thr_list = threshold if isinstance(threshold, (list, tuple, np.ndarray)) else [threshold]
            if abs:
                for t0 in thr_list:
                    if t0 is None or not np.isfinite(t0):
                        continue
                    ax.axvline(float(t0), linestyle='--', linewidth=1)
            else:
                for t0 in thr_list:
                    if t0 is None or not np.isfinite(t0):
                        continue
                    ax.axvline(float(+t0), linestyle='--', linewidth=1)
                    ax.axvline(float(-t0), linestyle='--', linewidth=1)

        # Legend (place to the right with some padding)
        ax.legend(legend_text, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        ax.set_ylim(0.5, len(y_labels) + 0.5)

    fig.tight_layout()
    plt.show()
    return x

# -------------
# Public API
# -------------

def lmw(
    formula: Any,
    data: Optional[pd.DataFrame] = None,
    estimand: Union[str, Estimand, EstimandType] = "ATE",
    method: Union[str, Method] = "URI",
    treat: Optional[Union[str, Sequence[Any]]] = None,
    base_weights: Optional[Union[str, Sequence[float]]] = None,
    s_weights: Optional[Union[str, Sequence[float]]] = None,
    dr_method: Union[str, DRMethod, None] = "WLS",
    obj: Any = None,
    fixef: Any = None,
    target: Any = None,
    target_weights: Optional[Sequence[float]] = None,
    contrast: Any = None,
    focal: Any = None,
    *,
    data_backend: str = "auto",
    io_options: Optional[Dict[str, Any]] = None,
    engine: str = "formulaic",
    ensure_full_rank: bool = True,
    output: str = "numpy",
) -> LMWResult:
    """Top-level API mirroring R's `lmw()`.

    This function handles the *input contract* and returns an `LMWResult`.
    The heavy lifting (design matrix; weights) is delegated to two functions
    that we will implement next.
    """

    # store the function call
    call_str = _capture_call(locals())

    # ensure the method and he estimand
    method_p = process_method(method)
    estimand_p = process_estimand(estimand, target, obj)

    # data processing
    table = process_data(data, obj, backend=data_backend, **(io_options or {}))
    data_p = table.to_pandas()

    # set the weights
    base_w = process_base_weights(base_weights, data_p)
    s_w = process_s_weights(s_weights, data_p)


    dr_method_p = process_dr_method(dr_method, base_w, method_p, estimand_p)

    treat_name = process_treat_name(treat, formula, data_p, method_p, obj)
    fixef_p = process_fixef(fixef, formula, data_p, treat_name)

    treat_vec = process_treat(treat_name, data_p)

    check_lengths(treat_vec, data_p, s_w, base_w, fixef_p)

    contrast_p = process_contrast(contrast, treat_vec, method_p)
    treat_contrast = apply_contrast_to_treat(treat_vec, contrast_p)

    focal_p = process_focal(focal, treat_contrast, estimand_p, obj)

    # Core steps
    X_obj = get_X_from_formula(
        formula,
        table,
        treat_contrast,
        method_p,
        estimand_p,
        target,
        s_w,
        target_weights,
        focal_p,
        engine=engine,
        ensure_full_rank=ensure_full_rank,
        output=output,
        treat_name=treat_name,
    )

    weights = get_w_from_X(X_obj.X, treat_contrast, method_p, base_w, s_w, dr_method_p, fixef_p)

    result = LMWResult(
        treat=treat_vec,
        weights=weights,
        covs=X_obj.mf,
        estimand=estimand_p,
        method=method_p,
        base_weights=base_w,
        s_weights=s_w,
        dr_method=dr_method_p,
        call=call_str,
        fixef=fixef_p,
        formula=formula,
        target=getattr(X_obj, "target", None),
        contrast=contrast_p,
        focal=focal_p,
    )

    return result

#
# __all__ = [
#     "LMWResult",
#     "Estimand",
#     "EstimandType",
#     "Method",
#     "DRMethod",
#     "lmw",
#     # processing helpers (for tests / future refactor)
#     "process_method",
#     "process_estimand",
#     "process_dr_method",
#     "process_data",
#     "process_base_weights",
#     "process_s_weights",
#     "process_treat_name",
#     "process_fixef",
#     "process_treat",
#     "check_lengths",
#     "process_contrast",
#     "apply_contrast_to_treat",
#     "process_focal",
#     "get_X_from_formula",
#     "get_w_from_X",
#     "TableBackend",
#     "PandasBackend",
#     "PolarsBackend",
#     "center_covs",
#     "_one_hot_treatment",
# ]



def main():
    # 以 MRI / ATT 为例
    res = lmw(
        "~ treat + age + education + race + married + nodegree + re74 + re75 + treat:re74+ treat:re75",
        data=lalonde, estimand="ATT", method="MRI", treat="treat"
    )

    res_sum = summary_lmw(res)

    print_summary(res_sum)


    return

if __name__ == "__main__":
    main()
