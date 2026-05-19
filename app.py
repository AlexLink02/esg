import os
import json
import time
from io import BytesIO
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="ESG AI Article Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_1 = "gpt-4.1-mini"
MODEL_2 = "gpt-5"


# =========================
# FORCE LIGHT THEME
# =========================

FORCE_LIGHT_THEME = """
<style>

:root {
    color-scheme: light !important;
}

html, body, [class*="css"] {
    background-color: #f4faf7 !important;
    color: #111827 !important;
}

.stApp {
    background: linear-gradient(135deg, #f4faf7 0%, #eef8f2 50%, #f7fbff 100%) !important;
    color: #111827 !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb;
}

.card {
    background: #ffffff !important;
    padding: 1.4rem;
    border-radius: 22px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #111827 !important;
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
}

.stButton > button:hover {
    background-color: #dc2626 !important;
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

textarea,
input,
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #111827 !important;
}

button[data-baseweb="tab"] {
    color: #111827 !important;
}

details {
    background-color: white !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    padding: 0.5rem !important;
}

[data-testid="stDataFrame"] {
    background-color: white !important;
}

.js-plotly-plot .plotly,
.js-plotly-plot .plotly div {
    background: white !important;
}

</style>
"""

st.markdown(FORCE_LIGHT_THEME, unsafe_allow_html=True)


# =========================
# HELPERS
# =========================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def clean_json_response(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return json.loads(text)


def normalize_score(value):
    allowed_scores = [0, 20, 40, 60, 80, 100]

    try:
        value = int(value)
    except Exception:
        return 0

    return min(allowed_scores, key=lambda x: abs(x - value))


def normalize_result(result):
    return {
        "summary": result.get("summary", ""),
        "environmental_score": normalize_score(result.get("environmental_score", 0)),
        "social_score": normalize_score(result.get("social_score", 0)),
        "governance_score": normalize_score(result.get("governance_score", 0)),
        "score_explanation": result.get("score_explanation", ""),
        "main_esg_dimension": result.get("main_esg_dimension", ""),
        "research_purpose": result.get("research_purpose", ""),
        "methodology": result.get("methodology", ""),
        "main_findings": result.get("main_findings", []),
        "key_insights": result.get("key_insights", []),
        "practical_implications": result.get("practical_implications", []),
        "limitations_or_gaps": result.get("limitations_or_gaps", []),
        "section_summaries": result.get("section_summaries", []),
        "interpretation": result.get("interpretation", "")
    }


def save_feedback_to_google_sheets(feedback: dict):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )

    gc = gspread.authorize(credentials)

    sheet = gc.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
    worksheet = sheet.sheet1

    existing_values = worksheet.get_all_values()

    headers = list(feedback.keys())

    if len(existing_values) == 0:
        worksheet.append_row(headers)

    worksheet.append_row(list(feedback.values()))


# =========================
# AI ANALYSIS
# =========================

def analyze_article(article_text, model_name):
    max_chars = 45000

    if model_name == "gpt-5":
        max_chars = 90000

    prompt = f"""
You are an ESG research analyst.

Analyze the following scientific ESG article.

The goal is that the user should understand the most important content of the article WITHOUT reading the full article.

Return ONLY valid JSON.

Use this exact JSON structure:

{{
  "summary": "Detailed executive summary in multiple paragraphs",

  "environmental_score": one of [0, 20, 40, 60, 80, 100],

  "social_score": one of [0, 20, 40, 60, 80, 100],

  "governance_score": one of [0, 20, 40, 60, 80, 100],

  "score_explanation": "Explain what the ESG content focus scores mean and why these scores were assigned.",

  "main_esg_dimension": "Environmental / Social / Governance / Mixed",

  "research_purpose": "Explain the main purpose of the article.",

  "methodology": "Explain methodology, dataset and research approach in detail.",

  "main_findings": [
    "finding 1",
    "finding 2",
    "finding 3",
    "finding 4",
    "finding 5"
  ],

  "key_insights": [
    "insight 1",
    "insight 2",
    "insight 3",
    "insight 4",
    "insight 5"
  ],

  "practical_implications": [
    "implication 1",
    "implication 2",
    "implication 3"
  ],

  "limitations_or_gaps": [
    "gap 1",
    "gap 2",
    "gap 3"
  ],

  "section_summaries": [
    {{
      "section": "Section name",
      "summary": "Detailed summary with minimum 3 full sentences."
    }}
  ],

  "interpretation": "Overall interpretation and value of the article."
}}

SCORING RULE:
- ESG scores are NOT quality scores.
- ESG scores are CONTENT FOCUS SCORES.
- Higher score = article focuses more on this ESG dimension.
- Scores must be selected only from: 0, 20, 40, 60, 80, 100.
- 0 = not present
- 20 = barely mentioned
- 40 = minor focus
- 60 = moderate focus
- 80 = strong focus
- 100 = dominant focus
- Assign scores based on explicit article content, not loose interpretation.
- Keep scoring conservative and consistent.

IMPORTANT:
- Avoid repeating the same information across sections.
- Section summaries must contain at least 3 full sentences.
- Include ALL major sections from the article.
- Prefer 5 or more section summaries when possible.
- Be detailed and specific.
- Avoid generic AI wording.
- Return ONLY JSON.
- No markdown.

ARTICLE:
{article_text[:max_chars]}
"""

    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    return normalize_result(clean_json_response(response.output_text))


# =========================
# VISUALS
# =========================

def make_bar_chart(result_1, result_2):
    dimensions = ["Environmental", "Social", "Governance"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dimensions,
        y=[
            result_1["environmental_score"],
            result_1["social_score"],
            result_1["governance_score"]
        ],
        name=MODEL_1,
        marker_color="#2563eb",
        text=[
            result_1["environmental_score"],
            result_1["social_score"],
            result_1["governance_score"]
        ],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=dimensions,
        y=[
            result_2["environmental_score"],
            result_2["social_score"],
            result_2["governance_score"]
        ],
        name=MODEL_2,
        marker_color="#f97316",
        text=[
            result_2["environmental_score"],
            result_2["social_score"],
            result_2["governance_score"]
        ],
        textposition="outside"
    ))

    fig.update_layout(
        title="ESG content focus comparison by AI model",
        barmode="group",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            range=[0, 110],
            title="Content focus score",
            tickvals=[0, 20, 40, 60, 80, 100],
            gridcolor="#e5e7eb"
        ),
        xaxis=dict(title="ESG dimension"),
        legend=dict(orientation="h", y=1.15, x=0.7),
        font=dict(color="#111827")
    )

    return fig


def make_pie_chart(result, title):
    values = [
        result["environmental_score"],
        result["social_score"],
        result["governance_score"]
    ]

    fig = go.Figure(data=[go.Pie(
        labels=["Environmental", "Social", "Governance"],
        values=values,
        hole=0.45,
        marker_colors=["#22c55e", "#3b82f6", "#f97316"],
        textinfo="label+percent"
    )])

    fig.update_layout(
        title=title,
        height=360,
        paper_bgcolor="white",
        font=dict(color="#111827")
    )

    return fig


# =========================
# PDF REPORT HELPERS
# =========================

def add_text_block(story, styles, title, text):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Paragraph(text if text else "Not provided", styles["Normal"]))
    story.append(Spacer(1, 10))


def add_list_block(story, styles, title, items):
    story.append(Paragraph(title, styles["Heading2"]))

    if items:
        for item in items:
            story.append(Paragraph(f"- {item}", styles["Normal"]))
    else:
        story.append(Paragraph("Not provided", styles["Normal"]))

    story.append(Spacer(1, 10))


def add_section_summaries(story, styles, sections):
    story.append(Paragraph("Section summaries", styles["Heading2"]))

    if sections:
        for sec in sections:
            story.append(
                Paragraph(
                    sec.get("section", "Unnamed section"),
                    styles["Heading3"]
                )
            )
            story.append(
                Paragraph(
                    sec.get("summary", "Not provided"),
                    styles["Normal"]
                )
            )
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Not provided", styles["Normal"]))

    story.append(Spacer(1, 10))


# =========================
# PDF REPORT
# =========================

def create_pdf_report(result_1, result_2, analysis_time):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ESG AI Article Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"]
        )
    )

    story.append(Paragraph(f"Model 1: {MODEL_1}", styles["Normal"]))
    story.append(Paragraph(f"Model 2: {MODEL_2}", styles["Normal"]))
    story.append(Paragraph(f"Analysis time: {analysis_time} seconds", styles["Normal"]))

    story.append(Spacer(1, 20))

    data = [
        ["Dimension", MODEL_1, MODEL_2],
        ["Environmental", result_1["environmental_score"], result_2["environmental_score"]],
        ["Social", result_1["social_score"], result_2["social_score"]],
        ["Governance", result_1["governance_score"], result_2["governance_score"]],
    ]

    table = Table(data, colWidths=[130, 180, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    story.append(table)

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Note: ESG scores are content focus scores, not quality scores. "
            "They show how strongly the article focuses on each ESG dimension.",
            styles["Normal"]
        )
    )

    for model_name, result in [(MODEL_1, result_1), (MODEL_2, result_2)]:
        story.append(PageBreak())

        story.append(
            Paragraph(
                f"Analysis results - {model_name}",
                styles["Title"]
            )
        )

        story.append(Spacer(1, 12))

        add_text_block(story, styles, "ESG score explanation", result["score_explanation"])
        add_text_block(story, styles, "Executive summary", result["summary"])
        add_text_block(story, styles, "Research purpose", result["research_purpose"])
        add_text_block(story, styles, "Methodology", result["methodology"])
        add_list_block(story, styles, "Main findings", result["main_findings"])
        add_list_block(story, styles, "Key insights", result["key_insights"])
        add_list_block(story, styles, "Practical implications", result["practical_implications"])
        add_list_block(story, styles, "Limitations or gaps", result["limitations_or_gaps"])
        add_section_summaries(story, styles, result["section_summaries"])
        add_text_block(story, styles, "Overall interpretation", result["interpretation"])

    doc.build(story)
    buffer.seek(0)

    return buffer


# =========================
# UI
# =========================

st.title("🌿 ESG AI Article Analyzer")

st.markdown("""
Compare two predefined AI models for ESG article summarization, classification and visualization.
""")


with st.sidebar:
    st.header("Models")
    st.info(MODEL_1)
    st.info(MODEL_2)

    st.markdown("---")
    st.caption("Upload an ESG article as PDF.")


tab1, tab2, tab3 = st.tabs([
    "Article input",
    "AI analysis",
    "User evaluation"
])


# =========================
# TAB 1
# =========================

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload ESG article PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        article_text = extract_pdf_text(uploaded_file)

        if len(article_text) < 1000:
            st.error("""
PDF extraction returned very little text.
This PDF may be scanned or extraction may have failed.
""")
        else:
            st.session_state["article_text"] = article_text
            st.session_state["uploaded_filename"] = uploaded_file.name

            st.success("PDF text extracted successfully.")
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**Extracted characters:** {len(article_text)}")

    if "article_text" in st.session_state:
        st.markdown("### Article preview")

        st.text_area(
            "Extracted PDF text",
            st.session_state["article_text"],
            height=420
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 2
# =========================

with tab2:
    if "article_text" not in st.session_state:
        st.warning("Please upload a PDF first.")

    else:
        analyze_button = st.button(
            "Run AI analysis",
            type="primary"
        )

        if analyze_button:
            with st.spinner("Analyzing with both AI models..."):
                start_time = time.time()

                try:
                    result_1 = analyze_article(
                        st.session_state["article_text"],
                        MODEL_1
                    )

                    result_2 = analyze_article(
                        st.session_state["article_text"],
                        MODEL_2
                    )

                    st.session_state["result_1"] = result_1
                    st.session_state["result_2"] = result_2
                    st.session_state["analysis_time"] = round(
                        time.time() - start_time,
                        2
                    )

                    st.success("Analysis completed.")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "result_1" in st.session_state and "result_2" in st.session_state:
            result_1 = st.session_state["result_1"]
            result_2 = st.session_state["result_2"]

            st.markdown("## Results overview")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Analysis time",
                f"{st.session_state['analysis_time']} sec"
            )

            m2.metric("Model 1", MODEL_1)
            m3.metric("Model 2", MODEL_2)

            st.plotly_chart(
                make_bar_chart(result_1, result_2),
                use_container_width=True
            )

            st.info("""
ESG scores are NOT quality scores.

The scores represent how strongly the article focuses on Environmental, Social and Governance topics.

Scoring scale:
0 = not present, 20 = barely mentioned, 40 = minor focus, 60 = moderate focus, 80 = strong focus, 100 = dominant focus.
""")

            pie1, pie2 = st.columns(2)

            with pie1:
                st.plotly_chart(
                    make_pie_chart(
                        result_1,
                        f"ESG distribution - {MODEL_1}"
                    ),
                    use_container_width=True
                )

            with pie2:
                st.plotly_chart(
                    make_pie_chart(
                        result_2,
                        f"ESG distribution - {MODEL_2}"
                    ),
                    use_container_width=True
                )

            st.markdown("## AI model comparison")

            comparison_points = []

            if len(result_2["summary"]) > len(result_1["summary"]):
                comparison_points.append(
                    f"{MODEL_2} generated a more detailed executive summary."
                )

            if len(result_2["section_summaries"]) > len(result_1["section_summaries"]):
                comparison_points.append(
                    f"{MODEL_2} identified more article sections."
                )

            if len(result_2["methodology"]) > len(result_1["methodology"]):
                comparison_points.append(
                    f"{MODEL_2} provided a more detailed methodology explanation."
                )

            if not comparison_points:
                comparison_points.append(
                    "Both models produced relatively similar levels of detail."
                )

            for point in comparison_points:
                st.write(f"• {point}")

            col1, col2 = st.columns(2)

            def render_list(items):
                if items:
                    for item in items:
                        st.write(f"• {item}")
                else:
                    st.write("Not provided")

            def render_model_card(title, result):
                st.markdown(f"## {title}")

                c1, c2, c3 = st.columns(3)

                c1.metric("Environmental", result["environmental_score"])
                c2.metric("Social", result["social_score"])
                c3.metric("Governance", result["governance_score"])

                st.markdown("### ESG score explanation")
                st.info(result["score_explanation"])

                st.markdown("### Executive summary")
                st.write(result["summary"])

                st.markdown("### Research purpose")
                st.write(result["research_purpose"])

                st.markdown("### Methodology")
                st.write(result["methodology"])

                st.markdown("### Main findings")
                render_list(result["main_findings"])

                st.markdown("### Key insights")
                render_list(result["key_insights"])

                st.markdown("### Practical implications")
                render_list(result["practical_implications"])

                st.markdown("### Limitations or gaps")
                render_list(result["limitations_or_gaps"])

                st.markdown("### Section summaries")

                if result["section_summaries"]:
                    for sec in result["section_summaries"]:
                        with st.expander(sec.get("section", "Section")):
                            st.write(sec.get("summary", "Not provided"))
                else:
                    st.write("Not provided")

                st.markdown("### Overall interpretation")
                st.write(result["interpretation"])

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)

                render_model_card(
                    f"Model 1: {MODEL_1}",
                    result_1
                )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)

                render_model_card(
                    f"Model 2: {MODEL_2}",
                    result_2
                )

                st.markdown("</div>", unsafe_allow_html=True)

            pdf_report = create_pdf_report(
                result_1,
                result_2,
                st.session_state["analysis_time"]
            )

            st.download_button(
                "Download full analysis report as PDF",
                data=pdf_report,
                file_name="esg_ai_analysis_report.pdf",
                mime="application/pdf"
            )


# =========================
# TAB 3
# =========================

with tab3:
    left, center, right = st.columns([1, 2.2, 1])

    with center:
        st.markdown("## User evaluation questionnaire")

        with st.form("feedback_form"):
            user_group = st.radio(
                "Are you a student or an analyst?",
                ["Student", "Analyst", "Other"]
            )

            esg_familiarity = st.slider(
                "How familiar are you with ESG-related articles?",
                1,
                5,
                3
            )

            ai_familiarity = st.slider(
                "How familiar are you with AI tools?",
                1,
                5,
                3
            )

            time_without_tool = st.number_input(
                "How many minutes would this analysis take without this tool?",
                min_value=0,
                max_value=300,
                value=0
            )

            time_with_tool = st.number_input(
                "How many minutes did the analysis take using this tool?",
                min_value=0,
                max_value=300,
                value=0
            )

            output_quality = st.slider(
                "The AI outputs were high quality.",
                1,
                5,
                3
            )

            usefulness = st.slider(
                "The tool was useful for understanding the article.",
                1,
                5,
                3
            )

            interpretability = st.slider(
                "The ESG visualizations were easy to interpret.",
                1,
                5,
                3
            )

            understanding_improvement = st.slider(
                "The tool improved my understanding of the article.",
                1,
                5,
                3
            )

            trust = st.slider(
                "I would trust this tool to support ESG article analysis.",
                1,
                5,
                3
            )

            future_use = st.radio(
                "Would you use this tool in the future?",
                ["Yes", "No", "Maybe"]
            )

            preferred_model = st.radio(
                "Which model output did you prefer?",
                [
                    MODEL_1,
                    MODEL_2,
                    "No clear preference"
                ]
            )

            open_comment = st.text_area(
                "What did you like or dislike about the tool?"
            )

            submitted = st.form_submit_button(
                "Save feedback"
            )

            if submitted:
                feedback = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "uploaded_filename": st.session_state.get("uploaded_filename", ""),
                    "user_group": user_group,
                    "esg_familiarity": esg_familiarity,
                    "ai_familiarity": ai_familiarity,
                    "time_without_tool_minutes": time_without_tool,
                    "time_with_tool_minutes": time_with_tool,
                    "output_quality": output_quality,
                    "usefulness": usefulness,
                    "interpretability": interpretability,
                    "understanding_improvement": understanding_improvement,
                    "trust": trust,
                    "future_use": future_use,
                    "preferred_model": preferred_model,
                    "open_comment": open_comment
                }

                try:
                    save_feedback_to_google_sheets(feedback)
                    st.success("Feedback saved successfully.")
                except Exception as e:
                    st.error(f"Feedback saving failed: {e}")

        st.caption("Feedback is saved privately to Google Sheets.")