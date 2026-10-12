"""
Pipeline Orchestrator for batch resume ingestion, filtering, scoring, and ranking with detailed logging.
"""
import os
import glob
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

from .config import config
from .models.schemas import (
    ParsedResume,
    CandidateResult,
    ScoreBreakdown,
    PenaltyDetails,
    BatchSummary,
    ScreeningOutput
)
from .parser.extractor import parse_resume
from .parser.cleaner import extract_github_url
from .screening.filter import evaluate_eligibility
from .screening.scorer import detect_penalties
from .enrichment.github import analyze_github_profile
from .llm.client import get_llm_client

logger = logging.getLogger("ResumeScreener.Pipeline")

def process_single_resume(file_path: str) -> Tuple[Optional[CandidateResult], Optional[Dict[str, str]], bool]:
    """
    Process an individual resume through parsing, hard filtering, LLM/heuristic evaluation, and GitHub enrichment.
    Returns: (CandidateResult, failure_info_if_any, is_eligible)
    """
    filename = os.path.basename(file_path)
    logger.debug(f"Starting processing for file: {filename}")
    
    try:
        parsed = parse_resume(file_path)
    except Exception as e:
        logger.error(f"Uncaught exception parsing {filename}: {e}")
        return None, {"file": filename, "error": str(e)}, False

    if parsed.is_malformed:
        logger.warning(f"File {filename} is malformed or unreadable: {parsed.error_message}")
        result = CandidateResult(
            candidate_name=parsed.candidate_name,
            eligible=False,
            rejection_reasons=[f"Unreadable resume: {parsed.error_message}"],
            source_file=filename
        )
        return result, {"file": filename, "error": parsed.error_message or "Malformed"}, False

    # 1. Hard Eligibility Filter
    is_eligible, rejection_reasons, matched_skills = evaluate_eligibility(parsed)
    if not is_eligible:
        logger.info(f"[Hard Filter: REJECTED] '{parsed.candidate_name}' ({filename}) -> Reasons: {', '.join(rejection_reasons)}")
        result = CandidateResult(
            candidate_name=parsed.candidate_name,
            email=parsed.email,
            github_url=parsed.github_url,
            eligible=False,
            rejection_reasons=rejection_reasons,
            matched_skills=matched_skills,
            source_file=filename
        )
        return result, None, False

    logger.info(f"[Hard Filter: PASSED] '{parsed.candidate_name}' ({filename}) - Eligible for depth scoring.")

    # 2. GitHub Enrichment (if available)
    github_score = 0.0
    github_summary = "No GitHub profile provided."
    _, github_username = extract_github_url(parsed.raw_text)
    if github_username:
        logger.info(f"[Enrichment] Found GitHub handle '@{github_username}' on resume for '{parsed.candidate_name}'")
        github_score, github_summary = analyze_github_profile(github_username)

    # 3. LLM / Heuristic Semantic Evaluation & Scoring
    llm_client = get_llm_client()
    llm_res = llm_client.evaluate_resume(parsed)

    # Calculate penalties explicitly
    penalties, penalty_deduction = detect_penalties(parsed.raw_text, parsed.projects_text)

    # If LLM specified penalties, take the maximum
    if llm_res.thin_wrapper_penalty > 0:
        penalties.thin_wrapper_deduction = max(penalties.thin_wrapper_deduction, llm_res.thin_wrapper_penalty)
    if llm_res.tutorial_penalty > 0:
        penalties.tutorial_deduction = max(penalties.tutorial_deduction, llm_res.tutorial_penalty)

    if penalties.penalty_reasons:
        logger.info(f"[Penalties Applied] '{parsed.candidate_name}': {', '.join(penalties.penalty_reasons)}")

    # Compute final score breakdown
    raw_ai_score = max(0.0, min(40.0, llm_res.ai_project_depth_score))
    # Apply penalty deduction to AI project score
    final_ai_score = max(0.0, raw_ai_score - penalties.thin_wrapper_deduction - penalties.tutorial_deduction - penalties.keyword_stuffing_deduction)
    
    py_score = max(0.0, min(30.0, llm_res.python_backend_score))
    cloud_score = max(0.0, min(15.0, llm_res.cloud_fullstack_score))
    eng_score = max(0.0, min(5.0, llm_res.engineering_depth_score))
    gh_score = min(10.0, github_score)

    total_score = round(final_ai_score + py_score + cloud_score + gh_score + eng_score, 1)

    breakdown = ScoreBreakdown(
        ai_project_depth=round(final_ai_score, 1),
        python_backend=round(py_score, 1),
        cloud_fullstack=round(cloud_score, 1),
        github=round(gh_score, 1),
        engineering_depth=round(eng_score, 1)
    )

    # Merge matched skills
    all_matched = sorted(list(set(matched_skills + llm_res.matched_skills)))

    logger.info(f"[Scoring Complete] '{parsed.candidate_name}': Total Score={total_score} (AI={final_ai_score}, Py={py_score}, Cloud={cloud_score}, GH={gh_score}, Eng={eng_score})")

    result = CandidateResult(
        candidate_name=parsed.candidate_name,
        email=parsed.email,
        github_url=parsed.github_url,
        eligible=True,
        total_score=total_score,
        score_breakdown=breakdown,
        penalties=penalties,
        matched_skills=all_matched,
        project_summary=llm_res.project_summary or "Python & AI development projects.",
        github_summary=github_summary,
        strengths=llm_res.strengths,
        concerns=llm_res.concerns,
        source_file=filename
    )
    return result, None, True


def run_screening_pipeline(input_dir: str) -> ScreeningOutput:
    """
    Run full screening pipeline across all resumes in the input directory.
    Employs bounded concurrency and aggregates results.
    """
    start_time = time.time()
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Gather files
    supported_extensions = ('*.pdf', '*.docx', '*.doc', '*.txt', '*.md')
    file_paths = []
    for ext in supported_extensions:
        file_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        file_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    # Deduplicate file paths
    file_paths = sorted(list(set(file_paths)))
    total_resumes = len(file_paths)

    logger.info(f"Discovered {total_resumes} resume files to evaluate in: {input_dir}")

    eligible_results: List[CandidateResult] = []
    rejected_results: List[CandidateResult] = []
    failed_files: List[Dict[str, str]] = []
    successfully_parsed = 0

    max_workers = min(config.concurrency_limit, max(1, total_resumes))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_resume, fp): fp for fp in file_paths}
        for future in as_completed(future_to_file):
            candidate_res, failure, is_eligible = future.result()
            
            if failure:
                failed_files.append(failure)

            if candidate_res:
                if not candidate_res.rejection_reasons or not any("Unreadable" in r for r in candidate_res.rejection_reasons):
                    successfully_parsed += 1

                if is_eligible:
                    eligible_results.append(candidate_res)
                else:
                    rejected_results.append(candidate_res)

    # Sort eligible candidates by total_score descending
    eligible_results.sort(key=lambda c: c.total_score, reverse=True)
    for rank_idx, candidate in enumerate(eligible_results, start=1):
        candidate.rank = rank_idx

    exec_time = round(time.time() - start_time, 2)
    summary = BatchSummary(
        total_resumes=total_resumes,
        successfully_parsed=successfully_parsed,
        eligible=len(eligible_results),
        rejected=len(rejected_results),
        failed_unreadable=len(failed_files),
        failed_files=failed_files,
        execution_time_seconds=exec_time
    )

    logger.info(f"Batch completed: {eligible_results.__len__()} eligible, {rejected_results.__len__()} rejected in {exec_time}s.")

    return ScreeningOutput(
        summary=summary,
        ranked_candidates=eligible_results,
        rejected_candidates=rejected_results
    )
