"""Raw-data validation tests (lrd.core.data.validation) and the loader's guard."""

import pytest

from lrd.core.data.loader import load_fama_french
from lrd.core.data.validation import DataValidationError, validate_fama_french

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"
HEADER = "Generated test file.\nSecond note line.\nThird note line.\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
FOOTER = "\n Annual Factors: January-December \n,Mkt-RF,SMB,HML,RMW,CMA,RF\n  2000,   1.00,   2.00,   3.00,   4.00,   5.00,   6.00\n"


def make_csv(tmp_path, rows, name="ff.csv"):
    """Write a file in the Kenneth French layout; rows are (yyyymm, [6 values as str])."""
    body = "".join(f"{d}," + ",".join(f"{v:>8}" for v in vals) + "\n" for d, vals in rows)
    path = tmp_path / name
    path.write_text(HEADER + body + FOOTER)
    return path


def clean_rows(n=24, start_year=2000):
    rows = []
    for i in range(n):
        year, month = start_year + i // 12, i % 12 + 1
        rows.append((f"{year}{month:02d}", ["0.50", "-0.20", "0.10", "0.30", "-0.10", "0.20"]))
    return rows


class TestRealData:
    def test_repository_data_passes(self):
        report = validate_fama_french(DATA_PATH)
        assert report.ok, report.errors
        assert report.n_monthly_rows == len(load_fama_french(DATA_PATH)[1]) + 1
        assert report.date_range[0] == "1963-07"
        assert len(report.source_hash) == 8

    def test_crash_months_are_flagged_not_errors(self):
        report = validate_fama_french(DATA_PATH)
        flagged = set(zip(report.outliers["month"], report.outliers["column"]))
        assert ("1987-10", "Mkt-RF") in flagged
        assert report.ok


class TestSyntheticProblems:
    def test_clean_file(self, tmp_path):
        report = validate_fama_french(make_csv(tmp_path, clean_rows()))
        assert report.ok
        assert report.n_monthly_rows == 24
        assert report.n_dropped_rows >= 1  # annual section and notes

    def test_missing_value(self, tmp_path):
        rows = clean_rows()
        rows[5][1][2] = ""
        report = validate_fama_french(make_csv(tmp_path, rows))
        assert report.missing["HML"] == 1
        assert not report.ok

    def test_missing_value_code(self, tmp_path):
        rows = clean_rows()
        rows[3][1][0] = "-99.99"  # French library missing-data code
        assert validate_fama_french(make_csv(tmp_path, rows)).missing["Mkt-RF"] == 1

    def test_garbage_value(self, tmp_path):
        rows = clean_rows()
        rows[7][1][1] = "abc"
        report = validate_fama_french(make_csv(tmp_path, rows))
        assert report.non_numeric["SMB"] == 1
        assert any("Non-numeric" in e for e in report.errors)

    def test_duplicate_month(self, tmp_path):
        rows = clean_rows()
        rows.insert(10, rows[3])
        report = validate_fama_french(make_csv(tmp_path, rows))
        assert report.duplicate_months == ["2000-04"]
        assert not report.is_sorted
        assert not report.ok

    def test_missing_month(self, tmp_path):
        rows = clean_rows()
        del rows[6]
        report = validate_fama_french(make_csv(tmp_path, rows))
        assert report.missing_months == ["2000-07"]

    def test_raise_if_invalid(self, tmp_path):
        rows = clean_rows()
        del rows[6]
        with pytest.raises(DataValidationError, match="missing month"):
            validate_fama_french(make_csv(tmp_path, rows)).raise_if_invalid()

    def test_outlier_flagged(self, tmp_path):
        rows = clean_rows(60)
        for i, (_, vals) in enumerate(rows):  # some spread so MAD > 0
            vals[0] = f"{(i % 7) * 0.1:.2f}"
        rows[30][1][0] = "-40.00"
        report = validate_fama_french(make_csv(tmp_path, rows))
        assert report.ok
        assert list(report.outliers["month"]) == ["2002-07"]

    def test_markdown_report(self, tmp_path):
        rows = clean_rows()
        del rows[6]
        text = validate_fama_french(make_csv(tmp_path, rows)).to_markdown()
        assert "FAILED" in text and "2000-07" in text and "| column |" in text

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_fama_french(tmp_path / "nope.csv")


class TestLoaderGuard:
    def test_loader_rejects_duplicated_months(self, tmp_path):
        rows = clean_rows()
        rows.insert(10, rows[3])
        with pytest.raises(ValueError, match="duplicated or out-of-order"):
            load_fama_french(make_csv(tmp_path, rows))

    def test_loader_accepts_clean_file(self, tmp_path):
        X, y, dates = load_fama_french(make_csv(tmp_path, clean_rows()))
        assert X.shape == (23, 5)
        assert dates.is_monotonic_increasing
