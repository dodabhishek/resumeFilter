"""
Pipeline Orchestrator for dynamic resume ingestion, filtering, scoring, and ranking.
Supports custom user criteria, project verification, and explainable acceptance/rejection remarks.
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
    ScreeningOutput,
    JobCriteria
)
from .parser.extractor import parse_resume
from .parser.cleaner import extract_github_url
from .screening.filter import evaluate_dynamic_eligibility
from .screening.scorer import detect_penalties
from .enrichment.github import analyze_github_profile
from .llm.client import get_llm_client

logger = logging.getLogger("ResumeScreener.Pipeline")

def process_single_resume(
    file_path: str,
    criteria: Optional[JobCriteria] = None
) -> Tuple[Optional[CandidateResult], Optional[Dict[str, str]], bool]:
    """
    Process an individual resume through document parsing, dynamic hard filtering,
    project verification, LLM/heuristic scoring, and GitHub enrichment.
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
            rejection_reasons=[f"Unreadable file: {parsed.error_message or 'Insufficient text'}"],
            remarks_rejected=[f"Resume could not be read: {parsed.error_message or 'Corrupted file or image format'}"],
            source_file=filename
        )
        return result, {"file": filename, "error": parsed.error_message or "Malformed"}, False

    # 1. Dynamic Eligibility & Project Verification Filter
    (
        is_eligible,
        rejection_reasons,
        acceptance_remarks,
        project_evidence,
        missing_proj_skills,
        missing_req_skills,
        matched_skills
    ) = evaluate_dynamic_eligibility(parsed, criteria)

    if not is_eligible:
        logger.info(f"[Filter: REJECTED] '{parsed.candidate_name}' ({filename}) -> {', '.join(rejection_reasons)}")
        
        # Build human-readable rejected remarks
        remarks_rejected = []
        for r in rejection_reasons:
            remarks_rejected.append(r)

        result = CandidateResult(
            candidate_name=parsed.candidate_name,
            email=parsed.email,
            github_url=parsed.github_url,
            eligible=False,
            rejection_reasons=rejection_reasons,
            remarks_rejected=remarks_rejected,
            missing_project_skills=missing_proj_skills,
            missing_required_skills=missing_req_skills,
            matched_skills=matched_skills,
            source_file=filename
        )
        return result, None, False

    logger.info(f"[Filter: PASSED] '{parsed.candidate_name}' ({filename}) - Candidate meets all required project criteria.")

    # 2. GitHub Enrichment (if available)
    github_score = 0.0
    github_summary = "No GitHub profile provided."
    _, github_username = extract_github_url(parsed.raw_text)
    if github_username:
        logger.info(f"[Enrichment] Found GitHub handle '@{github_username}' for '{parsed.candidate_name}'")
        github_score, github_summary = analyze_github_profile(github_username)

    # 3. LLM / Heuristic Semantic Evaluation & Scoring
    llm_client = get_llm_client()
    llm_res = llm_client.evaluate_resume(parsed, criteria)

    # Detect penalties
    all_reqs = []
    if criteria:
        all_reqs = criteria.required_keywords + criteria.must_have_project_keywords
    penalties, _ = detect_penalties(parsed.raw_text, parsed.projects_text, all_reqs)

    if llm_res.thin_wrapper_penalty > 0:
        penalties.thin_wrapper_deduction = max(penalties.thin_wrapper_deduction, llm_res.thin_wrapper_penalty)
    if llm_res.tutorial_penalty > 0:
        penalties.tutorial_deduction = max(penalties.tutorial_deduction, llm_res.tutorial_penalty)

    # Calculate total score for internal ranking
    raw_proj_score = max(0.0, min(40.0, llm_res.ai_project_depth_score))
    deductions = penalties.thin_wrapper_deduction + penalties.tutorial_deduction + penalties.keyword_stuffing_deduction
    final_proj_score = max(0.0, raw_proj_score - deductions)
    core_score = max(0.0, min(30.0, llm_res.python_backend_score))
    cloud_score = max(0.0, min(15.0, llm_res.cloud_fullstack_score))
    eng_score = max(0.0, min(5.0, llm_res.engineering_depth_score))
    gh_score = min(10.0, github_score)

    total_score = round(final_proj_score + core_score + cloud_score + gh_score + eng_score, 1)

    breakdown = ScoreBreakdown(
        ai_project_depth=round(final_proj_score, 1),
        python_backend=round(core_score, 1),
        cloud_fullstack=round(cloud_score, 1),
        github=round(gh_score, 1),
        engineering_depth=round(eng_score, 1)
    )

    # Merge matched skills
    all_matched = sorted(list(set(matched_skills + llm_res.matched_skills)))

    # Synthesize rich explainable remarks on what they built & why accepted
    final_acceptance_remarks = list(acceptance_remarks)
    
    # Add project evidence highlights
    for kw, ev in project_evidence.items():
        if ev and not any(kw in rem for rem in final_acceptance_remarks):
            final_acceptance_remarks.append(f"Demonstrated project implementation with {kw}: {ev}")

    # Add strengths from LLM/heuristic
    for s in llm_res.strengths:
        if s not in final_acceptance_remarks:
            final_acceptance_remarks.append(f"Strength: {s}")

    if gh_score >= 5.0 and github_username:
        final_acceptance_remarks.append(f"Active public GitHub activity found (@{github_username}): {github_summary}")

    result = CandidateResult(
        candidate_name=parsed.candidate_name,
        email=parsed.email,
        github_url=parsed.github_url,
        eligible=True,
        total_score=total_score,
        score_breakdown=breakdown,
        penalties=penalties,
        matched_skills=all_matched,
        project_summary=llm_res.project_summary or "Demonstrated hands-on engineering project implementation.",
        github_summary=github_summary,
        strengths=llm_res.strengths,
        concerns=llm_res.concerns,
        source_file=filename,
        remarks_accepted=final_acceptance_remarks,
        remarks_rejected=[],
        project_evidence_matched=project_evidence,
        missing_project_skills=[],
        missing_required_skills=[]
    )
    return result, None, True


def run_screening_pipeline(
    input_dir: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    criteria: Optional[JobCriteria] = None
) -> ScreeningOutput:
    """
    Run screening pipeline across resumes with dynamic criteria.
    Accepts either an input_dir or explicit list of file_paths.
    """
    start_time = time.time()
    
    candidate_files = []
    if file_paths:
        candidate_files = [fp for fp in file_paths if os.path.isfile(fp)]
    elif input_dir:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        supported_extensions = ('*.pdf', '*.docx', '*.doc', '*.txt', '*.md')
        for ext in supported_extensions:
            candidate_files.extend(glob.glob(os.path.join(input_dir, ext)))
            candidate_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    else:
        raise ValueError("Either input_dir or file_paths must be provided.")

    # Deduplicate files
    candidate_files = sorted(list(set(candidate_files)))
    total_resumes = len(candidate_files)

    logger.info(f"Screening batch: {total_resumes} resume(s) against criteria: '{criteria.role_title if criteria else 'Default'}'")

    eligible_results: List[CandidateResult] = []
    rejected_results: List[CandidateResult] = []
    failed_files: List[Dict[str, str]] = []
    successfully_parsed = 0

    max_workers = min(config.concurrency_limit, max(1, total_resumes))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_resume, fp, criteria): fp
            for fp in candidate_files
        }
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

    logger.info(f"Screening completed: {len(eligible_results)} accepted, {len(rejected_results)} rejected in {exec_time}s.")

    return ScreeningOutput(
        summary=summary,
        ranked_candidates=eligible_results,
        rejected_candidates=rejected_results
    )
