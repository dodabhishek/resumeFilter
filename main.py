#!/usr/bin/env python3
"""
CLI Entry point for the AI Resume Screening & Ranking System.
Usage:
    python main.py --input ./resumes --output ./output/results.json
    python main.py --role "AI Engineer" --project-required "LangGraph, RAG" --required "Python, FastAPI"
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

from src.pipeline import run_screening_pipeline
from src.models.schemas import ScreeningOutput, JobCriteria

def render_terminal_table(output: ScreeningOutput):
    """Print an attractive summary table to stdout."""
    summary = output.summary
    print("\n" + "=" * 80)
    print(" 🚀 AI RESUME SCREENING & RANKING SYSTEM — BATCH REPORT")
    print("=" * 80)
    print(f" Total Resumes Processed : {summary.total_resumes}")
    print(f" Successfully Parsed     : {summary.successfully_parsed}")
    print(f" Accepted / Eligible     : {summary.eligible}")
    print(f" Rejected                : {summary.rejected}")
    print(f" Unreadable / Malformed  : {summary.failed_unreadable}")
    print(f" Execution Time          : {summary.execution_time_seconds:.2f} seconds")
    print("-" * 80)

    print("\n🏆 ACCEPTED CANDIDATES (RANKED BY FIT):")
    if not output.ranked_candidates:
        print("  (No eligible candidates matched criteria)")
    else:
        print(f"{'Rank':<5} {'Candidate Name':<22} {'Source File':<20} {'Matched Project Skills'}")
        print("-" * 85)
        for c in output.ranked_candidates:
            skills_str = ", ".join(c.matched_skills[:4])
            print(f"#{c.rank:<4} {c.candidate_name[:20]:<22} {c.source_file[:18]:<20} {skills_str}")
            if c.remarks_accepted:
                for rem in c.remarks_accepted[:2]:
                    print(f"       ✓ {rem[:75]}")

    if output.rejected_candidates:
        print("\n❌ REJECTED CANDIDATES:")
        print(f"{'Candidate Name':<25} {'Rejection Reason'}")
        print("-" * 85)
        for c in output.rejected_candidates:
            reason = c.remarks_rejected[0] if c.remarks_rejected else (c.rejection_reasons[0] if c.rejection_reasons else "Requirements not met")
            print(f"{c.candidate_name[:23]:<25} {reason[:58]}")

    print("\n" + "=" * 80 + "\n")


def parse_comma_list(val: Optional[str]) -> List[str]:
    """Parse comma-separated string into list of strings."""
    if not val:
        return []
    return [item.strip() for item in val.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="AI Resume Screening & Ranking System")
    parser.add_argument("--input", "-i", default="./resumes", help="Directory path containing resumes (PDF/DOCX/TXT)")
    parser.add_argument("--output", "-o", default="./output/results.json", help="Output JSON file path")
    parser.add_argument("--role", default="Python + AI/Agentic Engineer", help="Target Job Role title")
    parser.add_argument("--project-required", default="LangGraph, RAG, AI", help="Skills candidate MUST have built projects with (comma-separated)")
    parser.add_argument("--required", default="Python", help="Core mandatory skills (comma-separated)")
    parser.add_argument("--preferred", default="FastAPI, Docker", help="Preferred bonus skills (comma-separated)")
    parser.add_argument("--criteria-file", default=None, help="JSON file containing JobCriteria")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Build or load JobCriteria
    if args.criteria_file and os.path.exists(args.criteria_file):
        with open(args.criteria_file, 'r', encoding='utf-8') as f:
            criteria_dict = json.load(f)
            criteria = JobCriteria(**criteria_dict)
    else:
        criteria = JobCriteria(
            role_title=args.role,
            required_keywords=parse_comma_list(args.required),
            must_have_project_keywords=parse_comma_list(args.project_required),
            preferred_keywords=parse_comma_list(args.preferred)
        )

    print(f"🔍 Screening resumes against Role: '{criteria.role_title}'")
    print(f"   • Must Have Built Projects With: {', '.join(criteria.must_have_project_keywords) or 'None'}")
    print(f"   • Core Required Skills: {', '.join(criteria.required_keywords) or 'None'}")
    print(f"   • Preferred Skills: {', '.join(criteria.preferred_keywords) or 'None'}")

    screening_output = run_screening_pipeline(input_dir=input_dir, criteria=criteria)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    export_data = {
        "batch_summary": screening_output.summary.model_dump(),
        "criteria": criteria.model_dump(),
        "ranked_candidates": [c.model_dump(exclude_none=True) for c in screening_output.ranked_candidates],
        "rejected_candidates": [c.model_dump(exclude_none=True) for c in screening_output.rejected_candidates]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    render_terminal_table(screening_output)
    print(f"💾 Full results saved successfully to: {output_file}")


if __name__ == "__main__":
    main()
