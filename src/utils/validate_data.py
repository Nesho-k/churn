import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Data validation for Telco Customer Churn dataset using pandas checks.
    """
    print("🔍 Starting data validation...")

    failed_expectations = []

    def check(name: str, condition: bool):
        if not condition:
            failed_expectations.append(name)

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")
    required_cols = ["customerID", "gender", "Partner", "Dependents",
                     "PhoneService", "InternetService", "Contract",
                     "tenure", "MonthlyCharges", "TotalCharges"]
    for col in required_cols:
        check(f"column_exists:{col}", col in df.columns)

    if "customerID" in df.columns:
        check("customerID_not_null", df["customerID"].notna().all())

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    if "gender" in df.columns:
        check("gender_valid_values", df["gender"].isin(["Male", "Female"]).all())
    if "Partner" in df.columns:
        check("Partner_valid_values", df["Partner"].isin(["Yes", "No"]).all())
    if "Dependents" in df.columns:
        check("Dependents_valid_values", df["Dependents"].isin(["Yes", "No"]).all())
    if "PhoneService" in df.columns:
        check("PhoneService_valid_values", df["PhoneService"].isin(["Yes", "No"]).all())
    if "Contract" in df.columns:
        check("Contract_valid_values",
              df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all())
    if "InternetService" in df.columns:
        check("InternetService_valid_values",
              df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all())

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges...")
    if "tenure" in df.columns:
        tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
        check("tenure_not_null", tenure_num.notna().all())
        check("tenure_non_negative", (tenure_num >= 0).all())
        check("tenure_max_120", (tenure_num <= 120).all())
    if "MonthlyCharges" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        check("MonthlyCharges_not_null", mc.notna().all())
        check("MonthlyCharges_non_negative", (mc >= 0).all())
        check("MonthlyCharges_max_200", (mc <= 200).all())
    if "TotalCharges" in df.columns:
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        check("TotalCharges_non_negative", (tc.dropna() >= 0).all())

    # === SUMMARY ===
    total_checks = (len(required_cols) + 7 + 6)  # schema + business + numeric
    passed_checks = total_checks - len(failed_expectations)
    is_valid = len(failed_expectations) == 0

    if is_valid:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)} checks failed")
        print(f"   Failed: {failed_expectations}")

    return is_valid, failed_expectations
