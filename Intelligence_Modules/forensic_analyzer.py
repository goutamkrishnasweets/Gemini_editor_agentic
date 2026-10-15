"""
Intelligence_Modules/forensic_analyzer.py
Forwarding module to Gemini_Modules.forensic_analyzer for backward/cross compatibility.
"""
from Gemini_Modules.forensic_analyzer import *
from Gemini_Modules.forensic_analyzer import (
    ForensicVideoAnalyzer,
    get_analyzer,
    analyze_video,
    analyze,
    render_explanation,
    DEFAULT_RESULT,
)
