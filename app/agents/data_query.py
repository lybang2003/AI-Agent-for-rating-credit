from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import re
from difflib import SequenceMatcher


class CSVDataQuery:
    def __init__(self, data_paths: List[str] | None = None) -> None:
        self.data_paths = data_paths or [
            "corporateCreditRatingWithFinancialRatios.csv",
            "corporate_rating.csv",
        ]
        self._last_debug_paths: List[str] = []

    def _search_path(self, filename: str) -> List[str]:
        """Tìm kiếm đệ quy theo tên file bắt đầu từ thư mục làm việc hiện tại."""
        matches: List[str] = []
        cwd = os.getcwd()
        for root, _, files in os.walk(cwd):
            if filename in files:
                matches.append(os.path.join(root, filename))
        return matches

    def _load(self) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        checked: List[str] = []
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        for p in self.data_paths:
            candidates: List[str] = []
            if os.path.isabs(p):
                candidates = [p]
            else:
                # thử đường dẫn tương đối trước
                candidates = [p]
                # nếu không có, tìm theo tên file trong toàn project
                if not os.path.exists(p):
                    candidates.extend(self._search_path(os.path.basename(p)))
                candidates.append(os.path.join(project_root, os.path.basename(p)))
            for c in candidates:
                checked.append(c)
                if os.path.exists(c):
                    try:
                        df = pd.read_csv(c, encoding="utf-8-sig")
                    except Exception:
                        try:
                            df = pd.read_csv(c, engine="python")
                        except Exception:
                            continue
                    # gắn nguồn để debug
                    setattr(df, "_source", c)
                    frames.append(df)
        self._last_debug_paths = checked
        return frames

    def _normalize(self, s: Any) -> str:
        return str(s).strip().lower()

    def _column_similarity(self, col: str, target: str) -> float:
        a = re.sub(r"[^a-z0-9]", "", self._normalize(col))
        b = re.sub(r"[^a-z0-9]", "", self._normalize(target))
        return SequenceMatcher(None, a, b).ratio()

    def _pick_company_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = [
            "company", "companyname", "name", "issuer", "firm", "entity",
            "organization", "org", "ticker", "symbol",
        ]
        cols_norm = {self._normalize(c): c for c in df.columns}
        for key in candidates:
            if key in cols_norm:
                return cols_norm[key]
        # fallback: choose first text-like column
        for c in df.columns:
            if df[c].dtype == object:
                return c
        return None

    def _pick_year_column(self, df: pd.DataFrame) -> Optional[str]:
        for cand in [
            "year", "fiscalyear", "asof", "date", "period", "reportdate",
        ]:
            for c in df.columns:
                if self._normalize(c) == cand:
                    return c
        # fallback: any column that looks like date/year
        for c in df.columns:
            if any(k in self._normalize(c) for k in ["year", "date", "asof", "period"]):
                return c
        return None

    def _pick_metric_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        # exact/normalized match first
        cols = {re.sub(r"[^a-z0-9]", "", self._normalize(c)): c for c in df.columns}
        key = re.sub(r"[^a-z0-9]", "", self._normalize(metric))
        if key in cols:
            return cols[key]
        # try alias
        aliases: Dict[str, List[str]] = {
            "currentratio": ["current ratio", "currentratio", "CurrentRatio"],
            "debtequityratio": ["debt to equity", "debttoequity", "DebtToEquity", "debtEquityRatio"],
        }
        for k, al in aliases.items():
            if key == k:
                for a in al:
                    norm_a = re.sub(r"[^a-z0-9]", "", self._normalize(a))
                    if norm_a in cols:
                        return cols[norm_a]
        # similarity match
        best_col, best_score = None, 0.0
        for c in df.columns:
            score = self._column_similarity(c, metric)
            if score > best_score:
                best_col, best_score = c, score
        return best_col if best_score >= 0.55 else None

    def _extract_year(self, val: Any) -> Optional[int]:
        s = str(val)
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None

    def query_metric_timeseries(self, company: str, metric: str) -> Dict[str, Any]:
        frames = self._load()
        if not frames:
            return {"company": company, "metric": metric, "years": [], "values": [], "debug": {"reason": "no_frames"}}

        years: List[int] = []
        values: List[float] = []
        debug_used: List[Dict[str, Any]] = []
        for df in frames:
            metric_col = self._pick_metric_column(df, metric)
            year_col = self._pick_year_column(df)
            company_col = self._pick_company_column(df)

            if metric_col is None or year_col is None:
                debug_used.append({"file": getattr(df, "_source", "csv"), "metric_col": metric_col, "year_col": year_col, "available_cols": list(df.columns)[:20]})
                continue

            if company:
                if company_col:
                    try:
                        mask = df[company_col].astype(str).str.contains(company, case=False, na=False)
                        df_company = df[mask]
                    except Exception:
                        df_company = df
                else:
                    # Fallback: tìm công ty trên toàn hàng nếu không xác định được cột công ty
                    try:
                        df_company = df[df.apply(lambda r: company.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
                    except Exception:
                        df_company = df
                # Nếu vẫn rỗng, dùng toàn bộ để không mất kết quả
                if df_company is None or len(df_company) == 0:
                    df_company = df
            else:
                df_company = df

            for _, row in df_company.iterrows():
                y = row[year_col]
                yi = None
                try:
                    yi = int(str(y)[:4])
                except Exception:
                    yi = self._extract_year(y)
                if yi is None:
                    continue
                try:
                    v = float(row[metric_col])
                except Exception:
                    continue
                years.append(yi)
                values.append(v)

        # Gộp theo năm, chọn giá trị mới nhất
        series: Dict[int, float] = {}
        for y, v in zip(years, values):
            series[y] = v
        ys = sorted(series.keys())
        return {
            "company": company,
            "metric": metric,
            "years": ys,
            "values": [series[y] for y in ys],
            "debug": {
                "used": debug_used,
                "paths_checked": getattr(self, "_last_debug_paths", []),
                "cwd": os.getcwd(),
                "project_root": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
            },
        }

    def list_companies(self, max_items: int = 2000) -> List[str]:
        frames = self._load()
        companies: List[str] = []
        for df in frames:
            col = None
            for c in df.columns:
                lc = str(c).lower()
                if "company" in lc or "name" in lc:
                    col = c
                    break
            if col is None:
                continue
            try:
                vals = [str(v).strip() for v in df[col].dropna().tolist()]
            except Exception:
                continue
            companies.extend([v for v in vals if v])
        uniq = sorted(list(dict.fromkeys(companies)))
        return uniq[:max_items]


