from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from app.storage.postgres_client import pg


def load_csv_to_storage(csv_path: str, company_column: str | None = None) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    count = 0
    for r in df.to_dict(orient="records"):
        company_id = None
        if company_column and company_column in r:
            company_id = r[company_column]
        else:
            for col in ["CompanyName", "company", "name"]:
                if col in r:
                    company_id = r[col]
                    break
        if company_id is None:
            continue
        features = {}
        for k, v in r.items():
            if isinstance(v, (int, float)):
                features[k] = float(v)
        try:
            pg.upsert_features(company_id, features)
            count += 1
        except Exception:
            pass
    return {"loaded": count, "source": csv_path}

