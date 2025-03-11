import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import openai
import time

###############################################
# 0. PAGE CONFIG & OPENAI KEY
###############################################
st.set_page_config(page_title="Vocal Justice Analysis w/ AI", layout="wide")
st.markdown(
    """
    <style>
    /* Set Background & Text Colors */
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    /* Style Sidebar */
    .stSidebar {
        background-color: #222;
    }
    /* Style Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF;
    }
    /* Style DataFrames */
    .dataframe {
        border-radius: 8px;
        background-color: #2A2A2A;
        color: white;
    }
    /* Style Buttons */
    .stButton>button {
        background-color: #4A90E2 !important;
        color: white !important;
        border-radius: 8px;
        font-size: 16px;
    }
    /* Style Input Fields */
    .stTextInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #4A90E2;
        padding: 8px;
        font-size: 14px;
        color: white;
        background-color: #2A2A2A;
    }

    /* Button Hover Effect */
    .stButton>button:hover {
        background-color: #2EC4B6 !important;
    }
    
    .stButton>button:hover {
        background-color: #2EC4B6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Montserrat:wght@700&display=swap');
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif;
    }
    body, p, div {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Change color of the warning box
st.markdown(
    """
    <style>
    /* Override st.warning box color & text */
    .stAlert {
        background-color: #9053A4 !important; /* Light purple */
        color: white !important;             /* White text */
    }
    .stAlert p {
        color: white !important; /* Ensure any paragraph text inside is also white */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Vocal Justice Survey AI Analysis Tool")

# Provide or request OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.info("Please enter your OpenAI API key to enable AI features.")
else:
    st.sidebar.success("OpenAI API key loaded!")  # Optional confirmation

###############################################
# 1. FILE UPLOADS
###############################################
st.sidebar.header("Upload Your Data")
pre_file = st.sidebar.file_uploader("Pre-Survey CSV (Onboarding)", type=["csv"])
post_file = st.sidebar.file_uploader("Post-Survey CSV (Post-Program)", type=["csv"])
students_post_file = st.sidebar.file_uploader("Students Post Survey Results CSV", type=["csv"])

# Require ALL 3 files at once
if not pre_file or not post_file or not students_post_file:
    st.warning("Please upload all three CSV files: Pre, Post, and Students Post Survey Results.")
    st.stop()

# Read them all
onboarding_df = pd.read_csv(pre_file)
post_program_df = pd.read_csv(post_file)
students_post_df = pd.read_csv(students_post_file)

# Simulate processing delay
with st.spinner("Loading data..."):
    time.sleep(1)

st.write("## Data Preview")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Pre-Survey (Onboarding) - First 5 Rows**")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(onboarding_df.head())
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.write("**Post-Survey (Post) - First 5 Rows**")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(post_program_df.head())
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.write("**Students Post Survey - First 5 Rows**")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(students_post_df.head())
    st.markdown('</div>', unsafe_allow_html=True)

###############################################
# 2. DEFINE COMPOSITE COLUMNS
###############################################
confidence_cols = [
    "I know how to help my students communicate persuasively about social justice issues.",
    "I know how to help my students feel confident.",
    "I know how to help my students build their critical consciousness."
]
advocacy_cols = [
    "I frequently talk with my students about social justice issues.",
    "I push my school leadership to integrate social justice education into our core curriculum."
]

# Convert columns to numeric
for col in confidence_cols + advocacy_cols:
    onboarding_df[col] = pd.to_numeric(onboarding_df[col], errors='coerce')
    post_program_df[col] = pd.to_numeric(post_program_df[col], errors='coerce')

# Create composite scores
onboarding_df["Confidence_Composite"] = onboarding_df[confidence_cols].mean(axis=1, skipna=True)
onboarding_df["Advocacy_Composite"]   = onboarding_df[advocacy_cols].mean(axis=1, skipna=True)
post_program_df["Confidence_Composite"] = post_program_df[confidence_cols].mean(axis=1, skipna=True)
post_program_df["Advocacy_Composite"]   = post_program_df[advocacy_cols].mean(axis=1, skipna=True)

###############################################
# 3. CALCULATE PRE/POST MEANS
###############################################
pre_conf_mean  = onboarding_df["Confidence_Composite"].dropna().mean()
post_conf_mean = post_program_df["Confidence_Composite"].dropna().mean()
pre_adv_mean   = onboarding_df["Advocacy_Composite"].dropna().mean()
post_adv_mean  = post_program_df["Advocacy_Composite"].dropna().mean()

###############################################
# A HELPER: Summarize chart data numerically
###############################################
def summarize_chart_data(description, data_points):
    if isinstance(data_points, dict):
        items = []
        for k, v in data_points.items():
            if isinstance(v, (int, float)):
                items.append(f"{k}={v:.2f}")
            else:
                items.append(f"{k}={str(v)}")
        stats_text = ", ".join(items)
    elif isinstance(data_points, list):
        stats_text = ", ".join(f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in data_points)
    else:
        stats_text = str(data_points)

    summary = (
        f"{description}\nData summary: {stats_text}."
        " This numeric context helps interpret the chart."
    )
    return summary

###############################################
# FULL CONTEXT TEXTS (Long Versions)
###############################################
prepost_context_text = """PRE/POST COMPOSITE SCORE BAR CHART ..."""
teacherchange_context_text = """INDIVIDUAL TEACHER CHANGE IN COMPOSITE SCORES ..."""
pctchange_context_text = """PERCENTAGE OF TEACHERS WITH SCORE INCREASES PER QUESTION ..."""
gender_context_text = """MEAN LIKERT RESPONSES BY GENDER ..."""

###############################################
# CREATE_GRAPH_CHAT (unchanged)
###############################################
def create_graph_chat(heading, purpose_text, figure, session_key, chat_context):
    st.write("---")
    st.subheader(heading)
    st.markdown(purpose_text)

    left_col, right_col = st.columns([1,1])
    with left_col:
        if figure is not None:
            st.pyplot(figure)

    with right_col:
        st.markdown(
            """
            <style>
            .chat-container {
                background-color: #1E1E1E;
                padding: 10px;
                border-radius: 8px;
                max-height: 350px;
                overflow-y: auto;
            }
            .chat-bubble {
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 10px;
                max-width: 80%;
                font-size: 14px;
            }
            .user-bubble {
                background-color: #4A90E2;
                color: white;
                float: right;
                text-align: right;
                margin-right: 10px;
            }
            .assistant-bubble {
                background-color: #2A2A2A;
                color: white;
                float: left;
                margin-left: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if session_key not in st.session_state:
            st.session_state[session_key] = []
            st.session_state[session_key].append({"role": "system", "content": chat_context})

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state[session_key]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                continue
            bubble_class = "assistant-bubble" if role == "assistant" else "user-bubble"
            st.markdown(
                f'<div class="chat-bubble {bubble_class}">{content}</div><div style="clear: both;"></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.form(key=f"{session_key}_form"):
            user_input = st.text_input("Your question:", key=f"{session_key}_input")
            submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            st.session_state[session_key].append({"role": "user", "content": user_input})

            if openai_api_key:
                try:
                    msgs_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state[session_key]
                        if m["role"] != "system"
                    ]
                    client = openai.OpenAI(api_key=openai_api_key)
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": chat_context}] + msgs_for_api,
                        max_tokens=300,
                        temperature=0.7
                    )
                    answer = resp.choices[0].message.content.strip()
                    st.session_state[session_key].append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    st.session_state[session_key].append({
                        "role": "assistant",
                        "content": f"Error calling OpenAI: {e}"
                    })
            else:
                st.session_state[session_key].append({
                    "role": "assistant",
                    "content": "No OpenAI API key provided."
                })

###############################################
# 4. FIRST GRAPH: Pre/Post Composite
###############################################
sns.set_theme(style="whitegrid")
fig_prepost, axz = plt.subplots(1, 2, figsize=(3.5, 3), sharey=True)

conf_means = [pre_conf_mean, post_conf_mean]
conf_labels = ["Pre", "Post"]
conf_bars = axz[0].bar(conf_labels, conf_means, color=["skyblue","coral"], edgecolor="black")
axz[0].set_title("Confidence", fontsize=10, fontweight='bold')
axz[0].set_ylim(0,5)
for bar in conf_bars:
    h = bar.get_height()
    axz[0].text(bar.get_x()+bar.get_width()/2, h+0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

adv_means = [pre_adv_mean, post_adv_mean]
adv_labels = ["Pre","Post"]
adv_bars = axz[1].bar(adv_labels, adv_means, color=["skyblue","coral"], edgecolor="black")
axz[1].set_title("Advocacy", fontsize=10, fontweight='bold')
axz[1].set_ylim(0,5)
for bar in adv_bars:
    h = bar.get_height()
    axz[1].text(bar.get_x()+bar.get_width()/2, h+0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

fig_prepost.tight_layout()

purpose_text_prepost = """
**Purpose**: Show the aggregate changes in Confidence and Advocacy composite scores before vs. after the program.  
**Why It’s Helpful**: Offers a quick, high-level overview of whether the group as a whole is trending up or down.
"""

prepost_numeric_summary = summarize_chart_data(
    "",
    {
        "Confidence_Pre": pre_conf_mean,
        "Confidence_Post": post_conf_mean,
        "Advocacy_Pre": pre_adv_mean,
        "Advocacy_Post": post_adv_mean
    }
)
chart_context_prepost = prepost_context_text + "\n\n" + prepost_numeric_summary

with st.spinner("Generating Pre/Post Composite Score Chart..."):
    time.sleep(1)

create_graph_chat(
    heading="Pre/Post Composite Score Bar Chart",
    purpose_text=purpose_text_prepost,
    figure=fig_prepost,
    session_key="chat_prepost",
    chat_context=chart_context_prepost
)

###############################################
# 5. SECOND GRAPH: Individual Teacher Change
###############################################
if "Username" in onboarding_df.columns and "Username" in post_program_df.columns:
    ...
    # (Same code as you had for the teacher change graph)
    ...

###############################################
# 6. THIRD GRAPH: Percentage of Teachers Improved (100% Stacked)
###############################################
...
# (Same code as you had for the stacked bar chart)
...

###############################################
# 7. FOURTH GRAPH: Mean Likert by Gender
###############################################
...
# (Same code as you had for the gender chart)
...

###############################################
# 8. FIFTH GRAPH: Student Post Survey (Gender Analysis)
###############################################
# We'll create a new graph from the "students_post_df"
if "Gender Identity" in students_post_df.columns:
    # Adjust these columns to match your actual code
    bucket_cols = ["Main", "Social Awareness Score", "Social Change Score"]

    # Check if all required columns exist
    missing_cols = [c for c in bucket_cols if c not in students_post_df.columns]
    if missing_cols:
        st.error(f"Missing columns in Students Post Survey Results: {missing_cols}")
    else:
        # Group by Gender Identity
        gender_means = students_post_df.groupby("Gender Identity")[bucket_cols].mean()

        fig_student, ax_student = plt.subplots(figsize=(6,4))
        x = np.arange(len(gender_means.index))
        width = 0.2

        for i, col in enumerate(bucket_cols):
            ax_student.bar(x + i*width, gender_means[col], width, label=col)

        ax_student.set_xticks(x + width*(len(bucket_cols)-1)/2)
        ax_student.set_xticklabels(gender_means.index, rotation=30, ha="right")
        ax_student.set_ylabel("Average Score")
        ax_student.set_title("Gender Analysis: Average Bucket Scores")
        ax_student.legend()

        purpose_text_students = """
        **Purpose**: Analyze average bucket scores by gender identity from the Student Post Survey.
        **Why It’s Helpful**: Reveals differences or similarities across various 'buckets' (e.g. Social Awareness, etc.) 
        based on gender identity.
        """
        chart_context_students = (
            "This chart shows the average scores for each bucket grouped by 'Gender Identity' "
            "from the Students Post Survey Results. "
        )

        with st.spinner("Generating Student Post Survey Gender Analysis..."):
            time.sleep(1)

        create_graph_chat(
            heading="Student Post Survey: Gender Analysis",
            purpose_text=purpose_text_students,
            figure=fig_student,
            session_key="chat_studentsurvey",
            chat_context=chart_context_students
        )
else:
    st.warning("No 'Gender Identity' column found in the Students Post Survey Results CSV.")

###############################################
# 9. DONE
###############################################
st.success("All analyses completed. Scroll up to review results and chat with each graph’s AI.")
