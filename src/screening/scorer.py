"""
Scoring engine implementing dynamic criteria scoring, penalties, and explainable remarks.
Supports custom user requirements while maintaining guardrails against shallow wrappers and tutorial copies.
"""
import re
from typing import Dict, List, Tuple, Optional
from ..models.schemas import ScoreBreakdown, PenaltyDetails, ParsedResume, JobCriteria
from ..config import config

# Regex signals for legacy depth evaluation
AGENTIC_DEPTH_SIGNALS = [
    (r'\blanggraph\b|\bmulti[- ]agent\b|\bautogen\b|\bcrewai\b|\bstategraph\b', 12.0, "Multi-agent / Stateful orchestration"),
    (r'\brag\b|\bretrieval[- ]augmented\b|\bhybrid search\b|\bre-ranking\b|\breranking\b|\bvector\b', 10.0, "RAG pipeline & vector search"),
    (r'\btool[- ]calling\b|\bfunction[- ]calling\b|\bmcp\b|\btool use\b', 8.0, "Tool-calling / Function integration"),
    (r'\bevaluation\b|\bragas\b|\bdeepeval\b|\bprompt evaluation\b|\bbenchmark\b', 6.0, "AI evaluation / Quality benchmarks"),
    (r'\blangchain\b|\bllamaindex\b|\bsemantic search\b|\bembeddings\b', 4.0, "AI Framework implementation")
]

PYTHON_BACKEND_SIGNALS = [
    (r'\bfastapi\b|\basyncio\b|\basynchronous\b|\bcelery\b|\buvicorn\b', 12.0, "Async FastAPI / Concurrency"),
    (r'\bpostgresql\b|\bpostgres\b|\bmysql\b|\bsqlalchemy\b|\balembic\b', 8.0, "Relational Database / ORM"),
    (r'\bredis\b|\bmemcached\b|\bcaching\b', 5.0, "Caching & Fast Store (Redis)"),
    (r'\bflask\b|\bdjango\b|\brest api\b|\bgrpc\b|\bwebsockets?\b', 5.0, "Backend API Architecture")
]

CLOUD_FULLSTACK_SIGNALS = [
    (r'\bdocker\b|\bcontainerization\b|\bdocker-compose\b', 5.0, "Docker containerization"),
    (r'\bgcp\b|\bgoogle cloud\b|\baws\b|\bazure\b|\bkubernetes\b|\bk8s\b', 6.0, "Cloud infrastructure (GCP/AWS)"),
    (r'\breact\b|\bnext\.?js\b|\btypescript\b|\btailwind\b|\bvue\b', 4.0, "Modern frontend integration")
]

ENGINEERING_DEPTH_SIGNALS = [
    (r'\bpytest\b|\bunit testing\b|\bintegration test\b|\btdd\b|\bmock\b', 2.0, "Automated Testing & QA"),
    (r'\bci/cd\b|\bgithub actions\b|\bprometheus\b|\bgrafana\b|\blogging\b|\bobservability\b', 2.0, "CI/CD & Observability"),
    (r'\brate limiting\b|\bfault tolerance\b|\bretry\b|\bcircuit breaker\b|\bconcurrency\b', 1.0, "System Resilience & Concurrency")
]

# Penalty Detection Patterns
THIN_WRAPPER_PATTERNS = [
    (r'\bsimple (?:chatbot|wrapper|summarizer)\b', 8.0, "Simple single-prompt wrapper without retrieval or workflow"),
    (r'\bstreamlit (?:app|bot) calling openai api\b', 10.0, "Basic UI calling single LLM completion endpoint"),
    (r'\bwrapped openai (?:api|chat)\b', 10.0, "Thin API wrapper with no backend data pipelines or state management"),
    (r'\b(api wrapper|simple wrapper|chatgpt clone without)\b', 12.0, "Shallow wrapper without custom logic or agent architecture")
]

TUTORIAL_PROJECT_PATTERNS = [
    (r'\btutorial project\b|\bfollowed tutorial\b|\bcode along\b', 8.0, "Tutorial-based project without independent implementation details"),
    (r'\bbasic todo app\b|\btoy project\b|\bclone tutorial\b', 5.0, "Standard boilerplate/tutorial clone")
]


def detect_penalties(text: str, projects_text: str = "", required_skills: Optional[List[str]] = None) -> Tuple[PenaltyDetails, float]:
    """
    Detect project-quality penalties:
    1. Deduct 5-15 pts for thin AI wrapper without workflow/state/retrieval.
    2. Deduct 5-10 pts for tutorial-style projects without ownership.
    3. Keyword stuffing: Detect if frameworks are listed in skills without project details.
    """
    combined = (text + " " + projects_text).lower()
    thin_penalty = 0.0
    tutorial_penalty = 0.0
    keyword_penalty = 0.0
    reasons = []

    # 1. Check for thin wrapper markers
    for pattern, penalty, reason in THIN_WRAPPER_PATTERNS:
        if re.search(pattern, combined):
            thin_penalty = max(thin_penalty, penalty)
            reasons.append(f"Thin Wrapper Penalty (-{penalty} pts): {reason}")

    # Check if candidate lists AI API but has zero RAG/agents/data/state in projects
    has_api_call = bool(re.search(r'\bopenai api\b|\bllm api\b|\bchatgpt api\b', combined))
    has_meaningful_workflow = bool(re.search(r'\brag\b|\bretrieval\b|\bagent\b|\bvector\b|\btools?\b|\bworkflow\b|\bstate\b|\bevaluation\b', combined))
    if has_api_call and not has_meaningful_workflow and thin_penalty == 0:
        thin_penalty = 10.0
        reasons.append("Thin Wrapper Penalty (-10.0 pts): AI project relies on direct API calls with no RAG, state management, or tool orchestration.")

    # 2. Check for tutorial markers
    for pattern, penalty, reason in TUTORIAL_PROJECT_PATTERNS:
        if re.search(pattern, combined):
            tutorial_penalty = max(tutorial_penalty, penalty)
            reasons.append(f"Tutorial Project Penalty (-{penalty} pts): {reason}")

    # 3. Check for skill keyword stuffing
    if projects_text and len(projects_text.strip()) > 30:
        projects_lower = projects_text.lower()
        if "langgraph" in combined and "langgraph" not in projects_lower and "graph" not in projects_lower:
            keyword_penalty += 2.0
            reasons.append("Keyword Discount (-2.0 pts): Framework listed in skills with limited usage evidence in project details.")

    total_deduction = thin_penalty + tutorial_penalty + keyword_penalty
    details = PenaltyDetails(
        thin_wrapper_deduction=thin_penalty,
        tutorial_deduction=tutorial_penalty,
        keyword_stuffing_deduction=keyword_penalty,
        penalty_reasons=reasons
    )
    return details, total_deduction


def calculate_category_score(text: str, signals: List[Tuple[str, float, str]], max_score: float) -> Tuple[float, List[str]]:
    """Calculate category points based on verified evidence signals."""
    score = 0.0
    evidences = []
    text_lower = text.lower()
    
    for pattern, weight, desc in signals:
        if re.search(pattern, text_lower):
            score += weight
            evidences.append(desc)
            
    return min(score, max_score), evidences


def compute_deterministic_score(parsed: ParsedResume, github_score: float = 0.0) -> Tuple[ScoreBreakdown, PenaltyDetails, List[str], List[str], str]:
    """
    Original 100-point rubric implementation for backward compatibility and benchmark tests.
    """
    text = parsed.raw_text
    
    ai_score, ai_evidences = calculate_category_score(text, AGENTIC_DEPTH_SIGNALS, 40.0)
    py_score, _ = calculate_category_score(text, PYTHON_BACKEND_SIGNALS, 30.0)
    cloud_score, _ = calculate_category_score(text, CLOUD_FULLSTACK_SIGNALS, 15.0)
    eng_score, _ = calculate_category_score(text, ENGINEERING_DEPTH_SIGNALS, 5.0)
    
    penalties, total_deduction = detect_penalties(text, parsed.projects_text)
    
    if total_deduction > 0:
        ai_score = max(0.0, ai_score - total_deduction)

    breakdown = ScoreBreakdown(
        ai_project_depth=round(ai_score, 1),
        python_backend=round(py_score, 1),
        cloud_fullstack=round(cloud_score, 1),
        github=round(min(github_score, 10.0), 1),
        engineering_depth=round(eng_score, 1)
    )

    strengths = []
    concerns = []

    if ai_score >= 25:
        strengths.append("Strong agentic/RAG project implementation with orchestration")
    elif ai_score >= 10:
        strengths.append("Demonstrated foundational AI framework integration")
    else:
        concerns.append("Limited AI/agentic project depth")

    if py_score >= 20:
        strengths.append("Robust asynchronous Python backend architecture")
    elif py_score < 10:
        concerns.append("Limited modern Python backend exposure (FastAPI/async)")

    if "Redis" not in text and "caching" not in text.lower():
        concerns.append("Limited Redis / caching evidence")

    if penalties.penalty_reasons:
        for r in penalties.penalty_reasons:
            concerns.append(r)

    if ai_evidences:
        project_summary = f"Demonstrated engineering work involving {', '.join(ai_evidences[:2])} and backend services."
    else:
        project_summary = "General software development with Python and supporting libraries."

    return breakdown, penalties, strengths, concerns, project_summary


def compute_dynamic_score(
    parsed: ParsedResume,
    criteria: Optional[JobCriteria] = None,
    github_score: float = 0.0
) -> Tuple[ScoreBreakdown, PenaltyDetails, List[str], List[str], str]:
    """
    Dynamically compute match score and qualitative remarks based on user criteria.
    """
    if not criteria or (not criteria.must_have_project_keywords and not criteria.required_keywords):
        return compute_deterministic_score(parsed, github_score)

    text = parsed.raw_text
    text_lower = text.lower()
    projects_lower = (parsed.projects_text or "").lower()

    # 1. Project Requirements Depth (Max 40 pts)
    project_score = 0.0
    matched_project_details = []
    if criteria.must_have_project_keywords:
        points_per_kw = 40.0 / len(criteria.must_have_project_keywords)
        for kw in criteria.must_have_project_keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                continue
            if kw_clean in projects_lower:
                project_score += points_per_kw
                matched_project_details.append(kw)
            elif kw_clean in text_lower:
                project_score += (points_per_kw * 0.7)
                matched_project_details.append(kw)
    else:
        project_score = 30.0

    project_score = min(40.0, project_score)

    # 2. Core Required Skills Score (Max 30 pts)
    core_score = 0.0
    if criteria.required_keywords:
        points_per_req = 30.0 / len(criteria.required_keywords)
        for req in criteria.required_keywords:
            req_clean = req.strip().lower()
            if req_clean and req_clean in text_lower:
                core_score += points_per_req
    else:
        core_score = 25.0

    core_score = min(30.0, core_score)

    # 3. Preferred / Supporting Stack Score (Max 15 pts)
    preferred_score = 0.0
    if criteria.preferred_keywords:
        points_per_pref = 15.0 / len(criteria.preferred_keywords)
        for pref in criteria.preferred_keywords:
            pref_clean = pref.strip().lower()
            if pref_clean and pref_clean in text_lower:
                preferred_score += points_per_pref
    else:
        for pat in [r'\bdocker\b', r'\baws\b|\bgcp\b', r'\bci/cd\b|\bgit\b']:
            if re.search(pat, text_lower):
                preferred_score += 5.0

    preferred_score = min(15.0, preferred_score)

    # 4. Engineering Depth Signals (Max 5 pts)
    eng_score, _ = calculate_category_score(text, ENGINEERING_DEPTH_SIGNALS, 5.0)

    # 5. Penalties
    all_reqs = criteria.required_keywords + criteria.must_have_project_keywords
    penalties, total_deduction = detect_penalties(text, parsed.projects_text, all_reqs)

    final_project_score = max(0.0, project_score - total_deduction)

    breakdown = ScoreBreakdown(
        ai_project_depth=round(final_project_score, 1),
        python_backend=round(core_score, 1),
        cloud_fullstack=round(preferred_score, 1),
        github=round(min(github_score, 10.0), 1),
        engineering_depth=round(eng_score, 1)
    )

    strengths = []
    concerns = []

    if matched_project_details:
        strengths.append(f"Demonstrated verified hands-on projects incorporating: {', '.join(matched_project_details)}")
    if core_score >= 20.0:
        strengths.append(f"Strong foundation across required technical competencies for {criteria.role_title}")
    if preferred_score >= 8.0:
        strengths.append("Additional experience with preferred supporting tools and cloud technologies")

    if penalties.penalty_reasons:
        for r in penalties.penalty_reasons:
            concerns.append(r)

    if matched_project_details:
        project_summary = f"Practical project portfolio highlights work with {', '.join(matched_project_details)} tailored for {criteria.role_title}."
    else:
        project_summary = f"Technical background and relevant project experience for {criteria.role_title}."

    return breakdown, penalties, strengths, concerns, project_summary
