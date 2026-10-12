"""
Dynamic eligibility and project-verification filter for screening candidate resumes.
Removes static hardcoded assumptions and dynamically evaluates against user-specified requirements,
while retaining backward-compatible functions for legacy suite.
"""
import re
from typing import Tuple, List, Dict, Optional, Set
from ..models.schemas import ParsedResume, JobCriteria

# Python Stack Keywords & Patterns (for backward compatibility & Python verification)
PYTHON_PATTERNS = [
    r'\bpython\b',
    r'\bpython3\b',
    r'\bfastapi\b',
    r'\bflask\b',
    r'\bdjango\b',
    r'\bpytorch\b',
    r'\btensorflow\b',
    r'\bpandas\b',
    r'\bnumpy\b',
    r'\bscikit-learn\b',
    r'\bpoetry\b',
    r'\bpipenv\b'
]

# AI / Agentic / RAG Patterns (for backward compatibility)
AI_AGENTIC_PATTERNS = [
    r'\blangchain\b',
    r'\blanggraph\b',
    r'\bllamaindex\b',
    r'\bllama-index\b',
    r'\bgoogle adk\b',
    r'\bautogen\b',
    r'\bcrewai\b',
    r'\brag\b',
    r'\bretrieval[- ]augmented generation\b',
    r'\bvector database\b|\bvector db\b|\bvector search\b|\bembeddings?\b|\bchromadb\b|\bpinecone\b|\bqdrant\b|\bweaviate\b|\bfaiss\b',
    r'\btool[- ]calling\b|\btool[- ]use\b|\bfunction[- ]calling\b',
    r'\bmulti[- ]agent\b|\bagentic\b|\bautonomous agent\b|\bllm agent\b|\bai agent\b',
    r'\bprompt engineering\b|\bevaluation pipeline\b|\bllm evaluation\b|\bfinetun\w*\b|\bfine-tun\w*\b',
    r'\bopenai\b|\banthropic\b|\bclaude\b|\bgpt[- ]?4\b|\bgpt[- ]?3\.5\b|\bgemini\b|\bllm\b|\blarge language models?\b'
]

# Common tech catalog for general skill extraction
GENERAL_SKILLS = [
    "Python", "FastAPI", "Flask", "Django", "PostgreSQL", "MySQL", "Redis", "Docker", "Kubernetes", "GCP", "AWS", "Azure",
    "LangChain", "LangGraph", "LlamaIndex", "ChromaDB", "Pinecone", "Qdrant", "Weaviate", "FAISS", "OpenAI", "Anthropic",
    "PyTorch", "TensorFlow", "Pandas", "NumPy", "Scikit-Learn", "Celery", "Kafka", "RabbitMQ", "GraphQL", "MongoDB",
    "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "Vue", "Java", "Spring Boot", "C++", "Golang", "Rust", "SQL"
]

# Action verbs commonly found in project descriptions
ACTION_VERBS = [
    r'\bbuilt\b', r'\bimplemented\b', r'\bdeveloped\b', r'\bdesigned\b', r'\barchitected\b',
    r'\bengineered\b', r'\bcreated\b', r'\bdeployed\b', r'\borchestrated\b', r'\bintegrated\b',
    r'\boptimized\b', r'\btrained\b', r'\btuned\b', r'\bevaluated\b', r'\bpipeline\b', r'\bworkflow\b'
]
ACTION_VERBS_REGEX = re.compile('|'.join(ACTION_VERBS), re.IGNORECASE)


def check_python_evidence(text: str) -> bool:
    """Check if Python appears genuinely in the resume."""
    text_lower = text.lower()
    for pattern in PYTHON_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def check_ai_agentic_evidence(text: str) -> bool:
    """Check if meaningful AI/LLM/RAG/agentic evidence exists."""
    text_lower = text.lower()
    for pattern in AI_AGENTIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def build_keyword_regex(keyword: str) -> re.Pattern:
    """Construct a flexible regex pattern for a given skill or requirement keyword."""
    cleaned = keyword.strip()
    if not cleaned:
        return re.compile(r'$^')
        
    escaped = re.escape(cleaned)
    prefix = r'\b' if cleaned[0].isalnum() else r''
    suffix = r'\b' if cleaned[-1].isalnum() else r''
    return re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)


def search_keyword_in_text(keyword: str, text: str) -> bool:
    """Check if keyword appears in text with boundary awareness."""
    pattern = build_keyword_regex(keyword)
    return bool(pattern.search(text))


def extract_context_snippet(keyword: str, text: str, max_chars: int = 140) -> Optional[str]:
    """Extract a concise readable sentence/snippet around where the keyword was used."""
    pattern = build_keyword_regex(keyword)
    match = pattern.search(text)
    if not match:
        return None
        
    start_pos = match.start()
    end_pos = match.end()
    
    line_start = max(0, text.rfind('\n', 0, start_pos) + 1)
    line_end = text.find('\n', end_pos)
    if line_end == -1:
        line_end = len(text)
        
    snippet = text[line_start:line_end].strip()
    snippet = re.sub(r'^[•\-\*\d\.\s]+', '', snippet).strip()
    
    if len(snippet) > max_chars:
        mid = (start_pos - line_start)
        left = max(0, mid - 60)
        right = min(len(snippet), mid + 60)
        snippet = ("..." if left > 0 else "") + snippet[left:right].strip() + ("..." if right < len(snippet) else "")
        
    return snippet or None


def verify_project_experience(parsed: ParsedResume, keyword: str) -> Tuple[bool, Optional[str], bool]:
    """
    Verify whether the candidate genuinely built projects or has hands-on experience
    with the specific requirement, rather than merely listing it in a skills section.

    Returns:
        (has_project_evidence, snippet, found_in_skills_only)
    """
    keyword_clean = keyword.strip()
    if not keyword_clean:
        return True, None, False

    pattern = build_keyword_regex(keyword_clean)

    # 1. Check segmented projects section
    if parsed.projects_text and pattern.search(parsed.projects_text):
        snippet = extract_context_snippet(keyword_clean, parsed.projects_text)
        return True, snippet, False

    # 2. Check segmented experience/work section
    if parsed.experience_text and pattern.search(parsed.experience_text):
        snippet = extract_context_snippet(keyword_clean, parsed.experience_text)
        return True, snippet, False

    # 3. Check for bullet points in raw text that describe building or implementation
    lines = parsed.raw_text.split('\n')
    for line in lines:
        line_clean = line.strip()
        if pattern.search(line_clean):
            is_action_line = bool(ACTION_VERBS_REGEX.search(line_clean))
            is_bullet = line_clean.startswith(('•', '-', '*', '–')) or (len(line_clean) > 30 and ' ' in line_clean)
            is_not_skill_header = not re.match(r'^(?:skills?|technologies|tools?|languages?):', line_clean, re.IGNORECASE)
            
            if (is_action_line or is_bullet) and is_not_skill_header and len(line_clean) > 25:
                snippet = extract_context_snippet(keyword_clean, line_clean)
                return True, snippet, False

    # 4. If keyword is present in raw text, but failed project verification -> it's in skills section only!
    if pattern.search(parsed.raw_text):
        return False, None, True

    # 5. Not found at all
    return False, None, False


def extract_matched_skills(text: str, custom_keywords: Optional[List[str]] = None) -> List[str]:
    """Extract list of recognized technologies present in the text."""
    matched = []
    text_lower = text.lower()
    
    vocabulary = list(GENERAL_SKILLS)
    if custom_keywords:
        for kw in custom_keywords:
            if kw and kw not in vocabulary:
                vocabulary.append(kw)

    for skill in vocabulary:
        pattern = build_keyword_regex(skill)
        if pattern.search(text_lower):
            matched.append(skill)
            
    return sorted(list(set(matched)))


def evaluate_dynamic_eligibility(
    parsed: ParsedResume,
    criteria: Optional[JobCriteria] = None
) -> Tuple[bool, List[str], List[str], Dict[str, str], List[str], List[str], List[str]]:
    """
    Dynamically evaluate candidate eligibility against user criteria.
    Enforces that candidates MUST have built projects on `must_have_project_keywords`.

    Returns:
        (is_eligible, rejection_reasons, acceptance_remarks, project_evidence, missing_project_skills, missing_required_skills, matched_skills)
    """
    if parsed.is_malformed:
        reason = f"Unreadable file: {parsed.error_message or 'Corrupted format or empty document'}"
        return False, [reason], [], {}, [], [], []

    # If no criteria supplied, default to Python + AI
    if not criteria:
        criteria = JobCriteria(
            role_title="Python + AI/Agentic Engineer",
            required_keywords=["Python"],
            must_have_project_keywords=["AI"],
            preferred_keywords=["FastAPI", "Docker"]
        )

    rejection_reasons: List[str] = []
    acceptance_remarks: List[str] = []
    project_evidence: Dict[str, str] = {}
    missing_project_skills: List[str] = []
    missing_required_skills: List[str] = []

    all_target_skills = criteria.required_keywords + criteria.must_have_project_keywords + criteria.preferred_keywords
    matched_skills = extract_matched_skills(parsed.raw_text, all_target_skills)

    # 1. Check Mandatory Project Requirements (MUST HAVE BUILT PROJECTS ON THESE!)
    for req in criteria.must_have_project_keywords:
        req_clean = req.strip()
        if not req_clean:
            continue
            
        has_proj, snippet, in_skills_only = verify_project_experience(parsed, req_clean)
        
        if has_proj:
            project_evidence[req_clean] = snippet or f"Demonstrated project implementation with {req_clean}"
            acceptance_remarks.append(f"Verified hands-on project experience with '{req_clean}': {snippet or 'Active implementation found.'}")
        elif in_skills_only:
            missing_project_skills.append(req_clean)
            rejection_reasons.append(
                f"Lacks required project experience with '{req_clean}'. (Skill is listed in resume, but no project implementation or work usage was found)."
            )
        else:
            missing_project_skills.append(req_clean)
            rejection_reasons.append(
                f"No evidence or project experience found for required technology '{req_clean}'."
            )

    # 2. Check Core Required Skills (Must appear on resume)
    for req in criteria.required_keywords:
        req_clean = req.strip()
        if not req_clean:
            continue
            
        if not search_keyword_in_text(req_clean, parsed.raw_text):
            missing_required_skills.append(req_clean)
            rejection_reasons.append(f"Missing required core skill: '{req_clean}'.")
        else:
            acceptance_remarks.append(f"Demonstrated core proficiency in required skill: '{req_clean}'.")

    # 3. Add remarks for preferred/bonus skills matched
    for pref in criteria.preferred_keywords:
        pref_clean = pref.strip()
        if pref_clean and search_keyword_in_text(pref_clean, parsed.raw_text):
            has_proj, snippet, _ = verify_project_experience(parsed, pref_clean)
            if has_proj:
                acceptance_remarks.append(f"Bonus: Hands-on project implementation with preferred technology '{pref_clean}'.")
            else:
                acceptance_remarks.append(f"Bonus: Familiarity with preferred technology '{pref_clean}'.")

    is_eligible = (len(rejection_reasons) == 0)

    if is_eligible:
        if criteria.must_have_project_keywords:
            acceptance_remarks.insert(0, f"Passed all required project filters for role '{criteria.role_title}' with verified hands-on engineering evidence.")
        else:
            acceptance_remarks.insert(0, f"Satisfied all specified requirements for role '{criteria.role_title}'.")

    return is_eligible, rejection_reasons, acceptance_remarks, project_evidence, missing_project_skills, missing_required_skills, matched_skills


def evaluate_eligibility(parsed: ParsedResume) -> Tuple[bool, List[str], List[str]]:
    """
    Backward-compatible evaluator for original test suite.
    Enforces genuine Python evidence and AI/agentic presence.
    """
    if parsed.is_malformed:
        return False, [f"Unreadable file: {parsed.error_message or 'Corrupted format'}"], []

    text = parsed.raw_text
    has_python = check_python_evidence(text)
    has_ai = check_ai_agentic_evidence(text)
    matched_skills = extract_matched_skills(text)

    rejection_reasons = []
    if not has_python:
        rejection_reasons.append("No evidence of Python stack")
    if not has_ai:
        rejection_reasons.append("No AI/agentic project evidence")

    is_eligible = len(rejection_reasons) == 0
    return is_eligible, rejection_reasons, matched_skills
