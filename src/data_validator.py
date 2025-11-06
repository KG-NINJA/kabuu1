"""データ品質を検証しレポートを生成する。"""



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineer

__all__ = ["DataValidator", "validate_data"]


@dataclass
class DataValidator:
    """欠損値や外れ値をチェックするユーティリティ。"""


    data_path: Path = Path("data/processed/features.csv")
    report_path: Path = Path("logs/validation_report.txt")

    def __post_init__(self) -> None:


        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """対象データを読み込む。無い場合は特徴量生成を実行する。"""
        if not self.data_path.exists():
            FeatureEngineer().engineer()
        dataframe = pd.read_csv(self.data_path)
        if dataframe.empty:
            raise ValueError("検証対象のデータが空です")
        return dataframe

    def check_missing(self, dataframe: pd.DataFrame) -> Dict[str, int]:
        """列ごとの欠損値数を返す。"""
        return dataframe.isna().sum().to_dict()

    def check_outliers(
        self, dataframe: pd.DataFrame, threshold: float = 3.0
    ) -> Dict[str, int]:

        """単純な Z スコアを利用して外れ値を検出する。"""

        numeric = dataframe.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}
        z_scores = (numeric - numeric.mean()) / numeric.std(ddof=0)
        outlier_counts = (np.abs(z_scores) > threshold).sum().to_dict()
        return outlier_counts

    def validate(self) -> Dict[str, Dict[str, int]]:
        """検証を実施しレポートを生成する。"""
        dataframe = self.load_data()
        missing = self.check_missing(dataframe)
        outliers = self.check_outliers(dataframe)
        report_lines = [
            "Data Validation Report",
            "======================",
            "Missing Values:",
        ]
        for column, count in missing.items():
            report_lines.append(f"  {column}: {count}")
        report_lines.append("")
        report_lines.append("Outlier Counts:")
        for column, count in outliers.items():
            report_lines.append(f"  {column}: {count}")

        self.report_path.write_text("\n".join(report_lines), encoding="utf-8")

        return {"missing": missing, "outliers": outliers}


def validate_data() -> Dict[str, Dict[str, int]]:

    """DataValidator を実行して結果を返す。"""

    validator = DataValidator()
    return validator.validate()


def _format_summary(result: Dict[str, Dict[str, int]]) -> str:
    """検証結果を読みやすく整形する。"""
    lines = ["Validation Summary"]
    for section, values in result.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)



if __name__ == "__main__":
    SUMMARY_PATH = Path("logs") / "validation_report.txt"
    validator = DataValidator(report_path=SUMMARY_PATH)
    RESULT = validator.validate()
    print(_format_summary(RESULT))
    print(f"Report generated at: {SUMMARY_PATH}")
