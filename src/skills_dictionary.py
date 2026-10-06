"""Skill definitions and category-specific keywords for ATS-like screening."""

CATEGORY_SKILLS = {
    "INFORMATION-TECHNOLOGY": [
        "python", "java", "c++", "c#", "javascript", "react", "node.js", "sql", "nosql",
        "aws", "azure", "docker", "kubernetes", "linux", "git", "api", "rest", "devops",
        "cybersecurity", "networking", "cloud computing", "machine learning", "microservices"
    ],
    "ENGINEERING": [
        "autocad", "solidworks", "matlab", "python", "ansys", "mechanical design", "plc",
        "circuit design", "project management", "quality assurance", "troubleshooting",
        "embedded systems", "robotics", "finite element analysis", "lean manufacturing"
    ],
    "BUSINESS-DEVELOPMENT": [
        "lead generation", "sales pipeline", "crm", "salesforce", "client relations",
        "market research", "revenue growth", "negotiation", "b2b", "strategic partnerships",
        "business strategy", "cold calling", "pitching", "contract negotiation"
    ],
    "FINANCE": [
        "financial modeling", "excel", "valuation", "forecasting", "budgeting", "risk analysis",
        "accounting", "quickbooks", "sap", "portfolio management", "capital markets",
        "financial statements", "variance analysis", "compliance", "auditing"
    ],
    "ACCOUNTANT": [
        "general ledger", "balance sheet", "reconciliation", "tax preparation", "gaap", "ifrs",
        "auditing", "accounts payable", "accounts receivable", "payroll", "quickbooks",
        "tally", "sap", "financial reporting", "cpa"
    ],
    "ADVOCATE": [
        "litigation", "legal research", "contract drafting", "corporate law", "arbitration",
        "compliance", "due diligence", "intellectual property", "court appearance",
        "dispute resolution", "legal writing", "negotiation", "regulatory compliance"
    ],
    "CHEF": [
        "culinary arts", "menu planning", "food safety", "haccp", "kitchen management",
        "inventory control", "recipe development", "pastry", "catering", "sanitation",
        "food presentation", "cost control", "cooking techniques"
    ],
    "FITNESS": [
        "personal training", "strength conditioning", "nutrition", "workout planning",
        "aerobics", "cpr", "first aid", "client assessment", "weight management",
        "rehabilitation", "kinesiology", "wellness coaching", "group fitness"
    ],
    "AVIATION": [
        "pilot", "flight operations", "faa regulations", "aircraft maintenance", "navigation",
        "cockpit management", "aviation safety", "crew resource management", "avionics",
        "flight planning", "atc communication", "aerodynamics"
    ],
    "SALES": [
        "b2b sales", "b2c sales", "cold calling", "lead conversion", "crm", "salesforce",
        "account management", "product demonstration", "upselling", "closing deals",
        "pipeline management", "quota attainment", "customer retention"
    ],
    "BANKING": [
        "retail banking", "credit analysis", "loans", "mortgages", "underwriting",
        "risk management", "kyc", "aml", "wealth management", "commercial banking",
        "branch operations", "treasury", "customer service"
    ],
    "HEALTHCARE": [
        "patient care", "nursing", "clinical assessment", "electronic health records", "ehr",
        "vital signs", "pharmacology", "triage", "hipaa", "medical billing", "infection control",
        "cpr", "bls", "patient education"
    ],
    "CONSULTANT": [
        "management consulting", "business transformation", "stakeholder management",
        "process improvement", "kpi analysis", "change management", "data analysis",
        "strategy formulation", "executive presentation", "root cause analysis"
    ],
    "CONSTRUCTION": [
        "project management", "site supervision", "building codes", "osha compliance",
        "cost estimation", "blueprint reading", "contractor management", "safety protocols",
        "subcontractor scheduling", "quality control", "civil engineering"
    ],
    "PUBLIC-RELATIONS": [
        "media relations", "press releases", "crisis communication", "brand messaging",
        "event planning", "copywriting", "social media strategy", "publicity",
        "spokesperson", "reputation management", "stakeholder engagement"
    ],
    "HR": [
        "talent acquisition", "recruitment", "employee onboarding", "payroll", "hrms",
        "employee relations", "performance appraisal", "labor laws", "conflict resolution",
        "training & development", "hr policies", "benefits administration"
    ],
    "DESIGNER": [
        "ui/ux", "figma", "adobe photoshop", "illustrator", "wireframing", "prototyping",
        "typography", "branding", "indesign", "graphic design", "user research",
        "responsive design", "visual hierarchy"
    ],
    "ARTS": [
        "illustration", "fine arts", "creative direction", "sculpting", "painting",
        "curation", "art history", "digital painting", "concept art", "storyboarding",
        "visual arts", "exhibition design"
    ],
    "TEACHER": [
        "curriculum development", "classroom management", "lesson planning", "pedagogy",
        "student assessment", "special education", "interactive teaching", "e-learning",
        "parent communication", "educational technology", "mentoring"
    ],
    "APPAREL": [
        "fashion design", "garment construction", "textile sourcing", "pattern making",
        "merchandising", "cad fashion", "trend forecasting", "apparel production",
        "quality inspection", "fabric testing"
    ],
    "DIGITAL-MEDIA": [
        "social media marketing", "seo", "sem", "content creation", "google analytics",
        "video editing", "premiere pro", "copywriting", "email campaigns", "branding",
        "digital advertising", "influencer outreach"
    ],
    "AGRICULTURE": [
        "crop management", "agronomy", "soil health", "irrigation systems", "pest control",
        "agricultural machinery", "sustainable farming", "livestock management",
        "farm operations", "harvesting"
    ],
    "AUTOMOBILE": [
        "automotive repair", "engine diagnostics", "powertrain", "chassis", "brake systems",
        "vehicle maintenance", "electric vehicles", "can bus", "mechanics", "obd-ii",
        "transmission systems"
    ],
    "BPO": [
        "customer support", "inbound calls", "outbound calls", "ticketing systems",
        "sla management", "telemarketing", "technical support", "voice process",
        "non-voice process", "chat support", "client escalations"
    ]
}

def extract_matched_skills(text: str, category: str = None) -> dict:
    """
    Scans text for keywords corresponding to the predicted or specified category.
    Returns matched skills and overall skill matches.
    """
    text_lower = text.lower()
    matches = {}
    
    # If category specified, find matches for that category
    if category and category in CATEGORY_SKILLS:
        cat_matches = [skill for skill in CATEGORY_SKILLS[category] if skill in text_lower]
        matches[category] = cat_matches
    else:
        for cat, skills in CATEGORY_SKILLS.items():
            cat_matches = [skill for skill in skills if skill in text_lower]
            if cat_matches:
                matches[cat] = cat_matches
                
    return matches
