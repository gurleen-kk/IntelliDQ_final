# test_rules.py
import pandas as pd
import numpy as np
import pytest
from app import run_dq_checks


# Test 1: Completeness detects missing values 
def test_completeness_score_drops_with_nulls():
    df = pd.DataFrame({
        "name": ["Alice", None, "Bob"],
        "age":  [25, 30, None]
    })
    result = run_dq_checks(df)
    assert result["completeness"]["score"] < 100
    assert result["completeness"]["total_missing_cells"] == 2


# Test 2: Uniqueness detects duplicate rows 
def test_uniqueness_flags_duplicate_rows():
    df = pd.DataFrame({
        "id":   [1, 1, 2],
        "name": ["Alice", "Alice", "Bob"]
    })
    result = run_dq_checks(df)
    assert result["uniqueness"]["duplicate_rows"] == 1
    assert result["uniqueness"]["score"] < 100


# Test 3: Validity flags negative values in an 'age' column 
def test_validity_flags_negative_age():
    df = pd.DataFrame({
        "age": [25, -5, 40, 30, 22]
    })
    result = run_dq_checks(df)
    # 'age' triggers the negative-value check in your code
    assert "age" in result["validity"]["issues"]


# Test 4: Consistency detects case-variant strings 
def test_consistency_detects_case_variants():
    df = pd.DataFrame({
        "status": ["Active", "active", "Inactive", "ACTIVE", "Inactive"]
    })
    result = run_dq_checks(df)
    assert "status" in result["consistency"]["issues"]


# Test 5: Overall score is a valid percentage 
def test_overall_score_is_valid_percentage():
    df = pd.DataFrame({
        "id":    [1, 2, 3, 4, 5],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    result = run_dq_checks(df)
    assert 0 <= result["overall_score"] <= 100