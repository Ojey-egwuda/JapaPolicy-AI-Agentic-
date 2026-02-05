import os
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults



# TOOL 1: Vector Search uses VectorDB
_vector_db_instance = None

def get_vector_db():
    """Lazy load VectorDB singleton."""
    global _vector_db_instance
    if _vector_db_instance is None:
        from src.vectordb import VectorDB  
        _vector_db_instance = VectorDB()
    return _vector_db_instance


@tool
def search_immigration_docs(query: str, n_results: int = 5) -> str:
    """
    Search the UK immigration policy document database for relevant information.
    Use this tool to find official gov.uk guidance on visas, eligibility, and requirements.
    
    Args:
        query: The search query about UK immigration
        n_results: Number of results to return (default 5)
    
    Returns:
        Formatted string of relevant document excerpts with sources
    """
    vector_db = get_vector_db()
    
    results = vector_db.search(
        query=query,
        n_results=n_results,
        min_similarity=0.65,
        use_hybrid=True
    )
    
    if not results.get("documents"):
        return "No relevant documents found for this query."
    
    formatted_results = []
    for i, (doc, meta, sim) in enumerate(zip(
        results["documents"],
        results["metadatas"],
        results["similarities"]
    ), 1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        formatted_results.append(
            f"[Result {i}] Source: {source}, Page {page} (Relevance: {sim:.1%})\n{doc}"
        )
    
    return "\n\n---\n\n".join(formatted_results)



# TOOL 2: Web Search for recent policy updates
@tool
def search_govuk_updates(query: str) -> str:
    """
    Search for recent UK immigration policy updates and news from gov.uk.
    Use this for questions about recent changes, new rules, or current processing times.
    
    Args:
        query: Search query about recent UK immigration updates
    
    Returns:
        Recent news and updates from gov.uk
    """
    search = TavilySearchResults(
        max_results=3,
        include_domains=["gov.uk", "freemovement.org.uk"],
        search_depth="advanced"
    )
    
    enhanced_query = f"UK immigration {query} site:gov.uk"
    
    try:
        results = search.invoke(enhanced_query)
        if not results:
            return "No recent updates found."
        
        formatted = []
        for r in results:
            formatted.append(f"**{r.get('title', 'No title')}**\n{r.get('content', 'No content')}\nURL: {r.get('url', 'N/A')}")
        
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search error: {str(e)}"



# TOOL 3: Visa Date Calculator
@tool
def calculate_visa_dates(
    visa_start_date: str,
    visa_length_years: int,
    days_outside_uk: int = 0,
    route: str = "skilled_worker"
) -> dict:
    """
    Calculate important visa dates including expiry, ILR eligibility, and absence compliance.
    
    Args:
        visa_start_date: Start date in YYYY-MM-DD format
        visa_length_years: Length of visa in years
        days_outside_uk: Total days spent outside UK during visa
        route: Visa route (skilled_worker, student, family, etc.)
    
    Returns:
        Dictionary with calculated dates and compliance status
    """
    try:
        start = datetime.strptime(visa_start_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}
    
    # Calculate key dates
    expiry = start + timedelta(days=visa_length_years * 365)
    
    # ILR eligibility varies by route
    ilr_years = {
        "skilled_worker": 5,
        "health_care": 5,
        "global_talent": 3,  # Can be 3 years for exceptional talent
        "innovator": 3,
        "student": 10,  # Long residence route
        "family": 5,
        "default": 5
    }
    
    years_for_ilr = ilr_years.get(route, ilr_years["default"])
    ilr_eligible = start + timedelta(days=years_for_ilr * 365)
    
    # Absence limits ie normally 180 days per year
    max_absence_per_year = 180
    max_total_absence = max_absence_per_year * visa_length_years
    absence_compliant = days_outside_uk <= max_total_absence
    
    # Calculate remaining allowed absences
    remaining_absence_days = max(0, max_total_absence - days_outside_uk)
    
    return {
        "visa_start": visa_start_date,
        "visa_expiry": expiry.strftime("%Y-%m-%d"),
        "ilr_eligible_from": ilr_eligible.strftime("%Y-%m-%d"),
        "years_until_ilr": years_for_ilr,
        "route": route,
        "max_allowed_absence_days": max_total_absence,
        "days_outside_uk": days_outside_uk,
        "remaining_absence_days": remaining_absence_days,
        "absence_compliant": absence_compliant,
        "absence_status": "✅ Compliant" if absence_compliant else "⚠️ Exceeded limit"
    }



# TOOL 4: Eligibility Checker 
@tool
def check_basic_eligibility(
    visa_type: str,
    salary: Optional[float] = None,
    has_job_offer: Optional[bool] = None,
    english_level: Optional[str] = None,
    has_sponsor: Optional[bool] = None
) -> dict:
    """
    Perform a basic eligibility pre-check for common visa types.
    This is a preliminary check - always verify with official sources.
    
    Args:
        visa_type: Type of visa (skilled_worker, student, family, graduate)
        salary: Annual salary in GBP (for work visas)
        has_job_offer: Whether applicant has a job offer
        english_level: English level (A1, A2, B1, B2, C1, C2)
        has_sponsor: Whether applicant has a licensed sponsor
    
    Returns:
        Eligibility assessment with requirements met/not met
    """
    requirements = {
        "skilled_worker": {
            "min_salary": 38700,  # General threshold as of 2024
            "needs_sponsor": True,
            "needs_job_offer": True,
            "english_required": "B1"
        },
        "health_care": {
            "min_salary": 29000,  # Lower threshold for health/care
            "needs_sponsor": True,
            "needs_job_offer": True,
            "english_required": "B1"
        },
        "graduate": {
            "min_salary": None,
            "needs_sponsor": False,
            "needs_job_offer": False,
            "english_required": None  # Already proven via student visa
        },
        "student": {
            "min_salary": None,
            "needs_sponsor": True,  # CAS from institution
            "needs_job_offer": False,
            "english_required": "B2"  # Varies by course level
        }
    }
    
    if visa_type not in requirements:
        return {
            "visa_type": visa_type,
            "status": "unknown",
            "message": f"Visa type '{visa_type}' not in basic checker. Consult official guidance."
        }
    
    reqs = requirements[visa_type]
    checks = []
    all_passed = True
    
    # Salary check
    if reqs["min_salary"] and salary is not None:
        passed = salary >= reqs["min_salary"]
        checks.append({
            "requirement": f"Minimum salary £{reqs['min_salary']:,}",
            "your_value": f"£{salary:,}",
            "status": "Met" if passed else "Not met"
        })
        if not passed:
            all_passed = False
    
    # Sponsor check
    if reqs["needs_sponsor"] and has_sponsor is not None:
        passed = has_sponsor
        checks.append({
            "requirement": "Licensed sponsor required",
            "your_value": "Yes" if has_sponsor else "No",
            "status": "Met" if passed else "Not met"
        })
        if not passed:
            all_passed = False
    
    # Job offer check
    if reqs["needs_job_offer"] and has_job_offer is not None:
        passed = has_job_offer
        checks.append({
            "requirement": "Job offer required",
            "your_value": "Yes" if has_job_offer else "No",
            "status": "Met" if passed else "Not met"
        })
        if not passed:
            all_passed = False
    
    return {
        "visa_type": visa_type,
        "checks": checks,
        "preliminary_status": "Likely eligible" if all_passed else "May not meet requirements",
        "disclaimer": "This is a preliminary check only. Always verify with official gov.uk guidance."
    }


# Export all tools
ALL_TOOLS = [
    search_immigration_docs,
    search_govuk_updates,
    calculate_visa_dates,
    check_basic_eligibility
]