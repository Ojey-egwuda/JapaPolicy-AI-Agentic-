"""
Tests for src/tools.py — pure-logic tools with no external calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.tools import calculate_visa_dates, check_basic_eligibility


class TestCalculateVisaDates:
    def test_basic_calculation(self):
        result = calculate_visa_dates.invoke({
            "visa_start_date": "2020-01-01",
            "visa_length_years": 5,
            "days_outside_uk": 0,
            "route": "skilled_worker",
        })
        # 2020-01-01 + 1825 days (5*365) = 2024-12-30 (crosses two leap years)
        assert result["visa_expiry"] == "2024-12-30"
        assert result["ilr_eligible_from"] == "2024-12-30"
        assert result["absence_compliant"] is True
        assert result["remaining_absence_days"] == 900  # 180 * 5

    def test_exceeded_absence(self):
        result = calculate_visa_dates.invoke({
            "visa_start_date": "2020-01-01",
            "visa_length_years": 5,
            "days_outside_uk": 1000,
            "route": "skilled_worker",
        })
        assert result["absence_compliant"] is False
        assert "Exceeded" in result["absence_status"]

    def test_invalid_date_returns_error(self):
        result = calculate_visa_dates.invoke({
            "visa_start_date": "not-a-date",
            "visa_length_years": 5,
            "days_outside_uk": 0,
            "route": "skilled_worker",
        })
        assert "error" in result

    def test_global_talent_ilr_three_years(self):
        result = calculate_visa_dates.invoke({
            "visa_start_date": "2020-01-01",
            "visa_length_years": 5,
            "days_outside_uk": 0,
            "route": "global_talent",
        })
        assert result["years_until_ilr"] == 3

    def test_compliant_status_label(self):
        result = calculate_visa_dates.invoke({
            "visa_start_date": "2022-01-01",
            "visa_length_years": 5,
            "days_outside_uk": 50,
            "route": "family",
        })
        assert "Compliant" in result["absence_status"]


class TestCheckBasicEligibility:
    def test_skilled_worker_meets_all(self):
        result = check_basic_eligibility.invoke({
            "visa_type": "skilled_worker",
            "salary": 40000,
            "has_job_offer": True,
            "has_sponsor": True,
            "english_level": None,
        })
        assert result["preliminary_status"] == "Likely eligible"

    def test_skilled_worker_low_salary(self):
        result = check_basic_eligibility.invoke({
            "visa_type": "skilled_worker",
            "salary": 25000,
            "has_job_offer": True,
            "has_sponsor": True,
            "english_level": None,
        })
        assert result["preliminary_status"] == "May not meet requirements"
        salary_check = next(c for c in result["checks"] if "salary" in c["requirement"].lower())
        assert salary_check["status"] == "Not met"

    def test_unknown_visa_type(self):
        result = check_basic_eligibility.invoke({
            "visa_type": "mystery_visa",
            "salary": None,
            "has_job_offer": None,
            "has_sponsor": None,
            "english_level": None,
        })
        assert result["status"] == "unknown"

    def test_graduate_no_salary_check(self):
        result = check_basic_eligibility.invoke({
            "visa_type": "graduate",
            "salary": None,
            "has_job_offer": None,
            "has_sponsor": None,
            "english_level": None,
        })
        # Graduate visa has no salary/sponsor/job-offer requirement → checks should be empty
        assert result["checks"] == []
        assert result["preliminary_status"] == "Likely eligible"

    def test_missing_sponsor_fails(self):
        result = check_basic_eligibility.invoke({
            "visa_type": "skilled_worker",
            "salary": 50000,
            "has_job_offer": True,
            "has_sponsor": False,
            "english_level": None,
        })
        assert result["preliminary_status"] == "May not meet requirements"


class TestSearchImmigrationDocs:
    def test_no_results_returns_message(self):
        """When VectorDB returns empty, tool should return the standard no-results string."""
        mock_db = MagicMock()
        mock_db.search.return_value = {
            "documents": [],
            "metadatas": [],
            "similarities": [],
            "search_type": "none",
            "top_semantic_sim": 0.0,
        }
        with patch("src.tools.get_vector_db", return_value=mock_db):
            from src.tools import search_immigration_docs
            result = search_immigration_docs.invoke({"query": "some query", "n_results": 5})
        assert result == "No relevant documents found for this query."
