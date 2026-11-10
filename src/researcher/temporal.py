from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class TemporalKind(str, Enum):
    """Distinguishes instantaneous real-world occurrences from durative conditions."""
    EVENT = "event"  # Point-in-time occurrence (e.g. facility opened, funding raised, leadership departed)
    STATE = "state"  # Durative condition (e.g. headquartered in, operating in, employee count, hiring planned)


class DatePrecision(str, Enum):
    """Preserves date granularity without false precision inflation."""
    YEAR = "year"              # e.g. "2024"
    MONTH = "month"            # e.g. "2024-03"
    DAY = "day"                # e.g. "2024-03-15"
    TIMESTAMP = "timestamp"    # e.g. "2024-03-15T10:30:00Z"
    UNKNOWN = "unknown"        # Missing or unparseable


def parse_partial_date(val: Optional[Union[str, date, datetime]]) -> Tuple[Optional[str], DatePrecision]:
    """Parse a date value preserving partial date precision.
    
    Invariant: An unknown date must remain unknown. Never substitutes the current time.
    A year-only date must never silently become an exact day.
    """
    if val is None:
        return None, DatePrecision.UNKNOWN

    if isinstance(val, datetime):
        return val.isoformat(), DatePrecision.TIMESTAMP

    if isinstance(val, date):
        return val.isoformat(), DatePrecision.DAY

    text = str(val).strip()
    if not text or text.lower() in ("unknown", "none", "n/a", "null"):
        return None, DatePrecision.UNKNOWN

    # Match full ISO timestamp (e.g. 2024-03-15T10:00:00Z or with offset)
    if "T" in text:
        try:
            # Validate parseable
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return text, DatePrecision.TIMESTAMP
        except ValueError:
            pass

    # Match YYYY-MM-DD
    match_day = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if match_day:
        year, month, day = int(match_day.group(1)), int(match_day.group(2)), int(match_day.group(3))
        try:
            date(year, month, day)
            return text, DatePrecision.DAY
        except ValueError:
            return None, DatePrecision.UNKNOWN

    # Match YYYY-MM
    match_month = re.match(r"^(\d{4})-(\d{2})$", text)
    if match_month:
        month = int(match_month.group(2))
        if 1 <= month <= 12:
            return text, DatePrecision.MONTH

    # Match YYYY
    match_year = re.match(r"^(\d{4})$", text)
    if match_year:
        year = int(match_year.group(1))
        if 1900 <= year <= 2100:
            return text, DatePrecision.YEAR

    # Match textual patterns: e.g. "March 2024", "Q3 2024", "2024"
    year_only_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year_only_match:
        # Check if month name precedes
        month_names = {
            "january": "01", "jan": "01",
            "february": "02", "feb": "02",
            "march": "03", "mar": "03",
            "april": "04", "apr": "04",
            "may": "05",
            "june": "06", "jun": "06",
            "july": "07", "jul": "07",
            "august": "08", "aug": "08",
            "september": "09", "sep": "09", "sept": "09",
            "october": "10", "oct": "10",
            "november": "11", "nov": "11",
            "december": "12", "dec": "12",
        }
        lower = text.lower()
        matched_month = None
        for m_name, m_num in month_names.items():
            if re.search(r"\b" + m_name + r"\b", lower):
                matched_month = m_num
                break

        year_str = year_only_match.group(1)
        if matched_month:
            return f"{year_str}-{matched_month}", DatePrecision.MONTH
        return year_str, DatePrecision.YEAR

    return None, DatePrecision.UNKNOWN


def get_current_iso_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# Predicates that describe instantaneous point occurrences vs durative ongoing states
EVENT_PREDICATES: Set[str] = {
    "facility_opened",
    "facility_closed",
    "funding_raised",
    "acquisition_acquired",
    "partnership_formed",
    "hiring_executed",
    "leadership_appointed",
    "leadership_departed",
    "product_launched",
}

STATE_PREDICATES: Set[str] = {
    "headquartered_in",
    "operating_in",
    "employee_count",
    "hiring_planned",
}


def determine_temporal_kind(predicate_str: Optional[str]) -> TemporalKind:
    """Determine whether a predicate represents a point-in-time event or durative state."""
    if not predicate_str:
        return TemporalKind.EVENT

    clean = predicate_str.strip().lower()
    if clean in STATE_PREDICATES:
        return TemporalKind.STATE
    if clean in EVENT_PREDICATES:
        return TemporalKind.EVENT

    # Heuristic for unmapped verbs
    state_markers = ("is", "has", "operates", "serves as", "headcount", "currently", "plans", "planning")
    if any(m in clean for m in state_markers):
        return TemporalKind.STATE

    return TemporalKind.EVENT


def normalize_to_comparable_bounds(val_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Convert a partial or full date into lower and upper ISO bounds for comparison.
    
    - "2024"       -> ("2024-01-01", "2024-12-31")
    - "2024-03"    -> ("2024-03-01", "2024-03-31")
    - "2024-03-15" -> ("2024-03-15", "2024-03-15")
    """
    if not val_str:
        return None, None

    clean = val_str.split("T")[0].strip()
    parts = clean.split("-")

    if len(parts) == 1 and len(parts[0]) == 4:
        # Year only
        y = parts[0]
        return f"{y}-01-01", f"{y}-12-31"

    if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
        # Year and Month
        y, m = parts[0], parts[1]
        days_in_month = "31"
        if m in ("04", "06", "09", "11"):
            days_in_month = "30"
        elif m == "02":
            # Leap year heuristic
            year_int = int(y)
            is_leap = (year_int % 4 == 0 and year_int % 100 != 0) or (year_int % 400 == 0)
            days_in_month = "29" if is_leap else "28"
        return f"{y}-{m}-01", f"{y}-{m}-{days_in_month}"

    if len(parts) == 3 and len(parts[0]) == 4:
        return clean, clean

    return clean, clean


def intervals_overlap(
    start_a: Optional[str],
    end_a: Optional[str],
    start_b: Optional[str],
    end_b: Optional[str],
) -> Tuple[bool, bool]:
    """Check if two validity intervals overlap.
    
    Returns:
        (is_overlapping, is_certain)
        If dates are missing, is_certain is False.
    """
    # If both start dates are completely missing, we cannot establish overlap with certainty
    if not start_a or not start_b:
        return False, False

    low_a, high_a = normalize_to_comparable_bounds(start_a)
    low_b, high_b = normalize_to_comparable_bounds(start_b)

    # Use end dates if provided, otherwise assume end bound of start date
    if end_a:
        _, high_a = normalize_to_comparable_bounds(end_a)
    if end_b:
        _, high_b = normalize_to_comparable_bounds(end_b)

    if not low_a or not high_a or not low_b or not high_b:
        return False, False

    # Overlap condition: start_a <= end_b and start_b <= end_a
    overlap = (low_a <= high_b) and (low_b <= high_a)
    return overlap, True


def is_temporally_sequential(
    date_a: Optional[str],
    date_b: Optional[str],
    end_a: Optional[str] = None,
    end_b: Optional[str] = None,
) -> bool:
    """Return True if date_a and date_b are known and distinct non-overlapping sequential points/intervals."""
    if not date_a or not date_b:
        return False

    low_a, high_a = normalize_to_comparable_bounds(date_a)
    low_b, high_b = normalize_to_comparable_bounds(date_b)

    if end_a:
        _, high_a = normalize_to_comparable_bounds(end_a)
    if end_b:
        _, high_b = normalize_to_comparable_bounds(end_b)

    if not low_a or not high_a or not low_b or not high_b:
        return False

    # Non-overlapping if interval A is strictly before B or interval B is strictly before A
    is_strictly_before = high_a < low_b
    is_strictly_after = high_b < low_a
    return is_strictly_before or is_strictly_after
