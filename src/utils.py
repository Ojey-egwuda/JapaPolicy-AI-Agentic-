"""
Utility functions for the UK Immigration Assistant.
"""

from datetime import datetime, timedelta

# Key policy change dates - update these when major changes happen
POLICY_CHANGE_DATES = {
    "student_dependants_restriction": "2024-01-01",
    "skilled_worker_salary_increase": "2024-04-04",
    "care_worker_dependants_ban": "2024-03-11",
    "graduate_visa_launch": "2021-07-01",
}

def get_freshness_warning(query: str) -> str:
    """
    Return a warning if the query relates to recently changed policy.
    """
    query_lower = query.lower()
    
    warnings = []
    
    if any(word in query_lower for word in ["student", "dependant", "dependent"]):
        warnings.append(
            "⚠️ Student dependant rules changed significantly in January 2024. "
            "Verify current rules on gov.uk."
        )
    
    if any(word in query_lower for word in ["salary", "threshold", "skilled worker"]):
        warnings.append(
            "⚠️ Skilled Worker salary thresholds changed on 4 April 2024. "
            "Verify current figures on gov.uk."
        )
    
    if any(word in query_lower for word in ["care worker", "care home", "social care"]):
        warnings.append(
            "⚠️ Care worker dependant rules changed on 11 March 2024. "
            "Verify current rules on gov.uk."
        )
    
    if any(word in query_lower for word in ["maintenance", "funds", "bank statement"]):
        warnings.append(
            "⚠️ Maintenance fund requirements may have changed. "
            "Always verify current figures on gov.uk before applying."
        )
    
    return "\n".join(warnings) if warnings else ""
