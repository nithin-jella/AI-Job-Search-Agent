import ast
import re
import pandas as pd
from typing import Dict, Tuple


def _parse_skills(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        # Most rows store skills as a Python-style list string.
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [s.strip().lower() for s in text.split(",") if s.strip()]


def _location_match(job_location: str, preferred_locations: list[str]) -> bool:
    loc = (job_location or "").lower()
    if "remote" in loc:
        return True
    return any(p.lower() in loc for p in preferred_locations)


def _parse_years_required(value) -> float:
    """
    Parses values like:
    - '5+ years' -> 5
    - '2-4 years' -> 2
    - '3 years' -> 3
    Unknown/missing -> NaN
    """
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return float("nan")
    return float(nums[0])


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Supports both old and new CSV schemas by normalizing to internal names.
    """
    # This lets the tool work with either the assignment column names or our internal ones.
    colmap = {
        "Job Title": "title",
        "Company": "company",
        "Location": "location",
        "Required Skills": "skills",
        "Shortened Job Description": "description",
        "URL": "job_url",
        "Date Posted": "job_posted_date",
        "Years of Experience Required": "years_experience_required",
    }
    rename_map = {c: colmap[c] for c in data.columns if c in colmap}
    return data.rename(columns=rename_map)


def filter_jobs(df: pd.DataFrame, candidate_profile: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Required filters:
    1) location match or remote
    2) min skill matches >= K
    3) years_experience_required <= candidate_years (if years column exists)
    """
    data = _normalize_columns(df.copy())

    data["title"] = data["title"].fillna("")
    data["company"] = data["company"].fillna("")
    data["location"] = data["location"].fillna("")
    data["skills"] = data["skills"].fillna("")
    data["description"] = data["description"].fillna("")
    data["skills_list"] = data["skills"].apply(_parse_skills)
    if "years_experience_required" in data.columns:
        data["years_experience_required"] = data["years_experience_required"].apply(
            _parse_years_required
        )
    else:
        data["years_experience_required"] = float("nan")

    preferred_locations = candidate_profile.get("preferred_locations", [])
    candidate_skills = {s.lower() for s in candidate_profile.get("skills", [])}
    min_skill_matches_k = int(candidate_profile.get("min_skill_matches_k", 2))
    candidate_years = float(candidate_profile.get("years_experience", 0))

    total_in = len(data)

    # 1) Location filter
    loc_mask = data["location"].apply(lambda loc: _location_match(loc, preferred_locations))
    data = data[loc_mask].copy()
    after_location = len(data)

    # 2) Skills filter
    data["skill_overlap_count"] = data["skills_list"].apply(
        lambda job_skills: len(candidate_skills.intersection(set(job_skills)))
    )
    data = data[data["skill_overlap_count"] >= min_skill_matches_k].copy()
    after_skills = len(data)

    # 3) Experience filter
    exp_mask = data["years_experience_required"].isna() | (
        data["years_experience_required"] <= candidate_years
    )
    data = data[exp_mask].copy()
    after_experience = len(data)

    trace = {
        "input_rows": total_in,
        "after_location_filter": after_location,
        "after_skill_filter": after_skills,
        "after_experience_filter": after_experience,
        "filters_applied": [
            "location_match_or_remote",
            f"skill_overlap_count >= {min_skill_matches_k}",
            "years_experience_required <= candidate_years",
        ],
    }

    return data.reset_index(drop=True), trace
