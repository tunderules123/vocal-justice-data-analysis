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

st.title("Vocal Lens: A Vocal Justice AI Data Analysis Tool")

# Provide or request OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.sidebar.info("Please enter your OpenAI API key to enable AI features.")
else:
    st.sidebar.success("OpenAI API key loaded!")  # Optional confirmation

###############################################
# 1. FILE UPLOADS
###############################################
st.sidebar.header("Upload All 3 CSVs")
pre_file = st.sidebar.file_uploader("Pre-Survey CSV (Onboarding)", type=["csv"])
post_file = st.sidebar.file_uploader("Post-Survey CSV (Post-Program)", type=["csv"])
students_post_file = st.sidebar.file_uploader("Students Post Survey Results CSV", type=["csv"])

if not pre_file or not post_file or not students_post_file:
    st.warning("Please upload all three CSV files (Pre, Post, and Students Post Survey).")
    st.stop()

# Read the CSVs
onboarding_df = pd.read_csv(pre_file)
post_program_df = pd.read_csv(post_file)
students_post_df = pd.read_csv(students_post_file)

# Simulate processing delay
with st.spinner("Loading data..."):
    time.sleep(1)

# Display first 5 rows of each
st.write("## Data Preview")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Pre-Survey (Onboarding)** - First 5 Rows")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(onboarding_df.head())
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.write("**Post-Survey (Post)** - First 5 Rows")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(post_program_df.head())
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.write("**Students Post Survey** - First 5 Rows")
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

# Convert columns to numeric for pre/post
for col in confidence_cols + advocacy_cols:
    onboarding_df[col] = pd.to_numeric(onboarding_df[col], errors='coerce')
    post_program_df[col] = pd.to_numeric(post_program_df[col], errors='coerce')

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

prepost_context_text = """\
PRE/POST COMPOSITE SCORE BAR CHART

**Relevant Data Columns & Setup**:
- Each survey participant has composite scores in two areas:
  1) Confidence_Composite: Derived as the mean of three columns:
     - "I know how to help my students communicate persuasively about social justice issues."
     - "I know how to help my students feel confident."
     - "I know how to help my students build their critical consciousness."
  2) Advocacy_Composite: Derived as the mean of two columns:
     - "I frequently talk with my students about social justice issues."
     - "I push my school leadership to integrate social justice education into our core curriculum."
- The CSV files (“pre” and “post”) each contain these questions plus a “Username” column and possibly other demographic fields.

**Code Approach**:
- We read both CSVs into dataframes (onboarding_df for “Pre” and post_program_df for “Post”).
- We calculate Confidence_Composite and Advocacy_Composite for each user by averaging the relevant columns.
- We then compute the mean of these composites across all users pre vs. post.
- Using matplotlib/seaborn, we create a **two-panel bar chart**:
  - Left panel compares mean Confidence_Composite (Pre vs. Post).
  - Right panel compares mean Advocacy_Composite (Pre vs. Post).

**Results Summary**:
- The chart typically shows that both Confidence_Composite and Advocacy_Composite **increased** from pre to post.
- For example, you might see Confidence go from ~3.32 (Pre) to ~3.59 (Post) and Advocacy from ~3.39 to ~3.76.
- This suggests that, on average, teachers feel slightly more confident and more engaged in advocacy after the program.
- Each bar has a label on top indicating the mean score (ranging from 0 to 5).
"""

teacherchange_context_text = """\
INDIVIDUAL TEACHER CHANGE IN COMPOSITE SCORES

**Relevant Data Columns & Setup**:
- The CSV includes a "Username" column to identify each teacher.
- For each teacher, we have “Confidence_Composite_Pre,” “Confidence_Composite_Post,” “Advocacy_Composite_Pre,” “Advocacy_Composite_Post.”
- We then create "Confidence_Change" = (Post - Pre) and "Advocacy_Change" = (Post - Pre).

**Code Approach**:
- After merging the pre and post dataframes on "Username," we compute the difference in each teacher’s Confidence_Composite and Advocacy_Composite.
- The code then uses two horizontal bar plots (matplotlib) side-by-side, often called a “diverging bar chart,” where bars extending to the right (green) indicate an increase, and bars extending to the left (red) indicate a decrease.

**Results Summary**:
- Each teacher’s username is on the y-axis, and the x-axis shows how much their score changed (e.g., +0.5 means they increased by half a point).
- Typically, you’ll see most teachers in the green region, indicating positive growth, and a smaller number in the red region, indicating a drop.
- Some teachers might have a large jump in Advocacy compared to Confidence (or vice versa), highlighting individual differences.
- Overall, this graph visually demonstrates that most participants improved from pre to post, but a few either stayed the same or declined slightly in certain areas.
"""

pctchange_context_text = """\
PERCENTAGE OF TEACHERS WITH SCORE INCREASES PER QUESTION

**Relevant Data Columns & Setup**:
- We look at each of the **five** Likert-scale questions individually:
  1) "I frequently talk with my students about social justice issues."
  2) "I know how to help my students communicate persuasively about social justice issues."
  3) "I know how to help my students feel confident."
  4) "I know how to help my students build their critical consciousness."
  5) "I push my school leadership to integrate social justice education into our core curriculum."
- For each user, we compare Pre vs. Post responses to each question.

**Code Approach**:
- The script merges the pre and post files on "Username."
- For each question, it calculates how many teachers improved (post > pre), stayed the same (post == pre), or declined (post < pre).
- It then creates a **100% stacked bar chart** in matplotlib/seaborn:
  - The horizontal axis is the proportion of teachers (0-100%).
  - Each question is on the y-axis.
  - Within each bar, green = improved, gray = same, red = declined.

**Results Summary**:
- Each question’s bar is subdivided into the percentage who improved, stayed the same, or declined.
- For instance, you might see 62% improved on one question, 38% stayed the same, and 0% declined. Another question might have more declines.
- This helps show exactly which skill areas or beliefs changed the most across the group.
- Often, we see **most** teachers improved in at least one question area, but the magnitude and direction vary across questions.
"""

gender_context_text = """\
MEAN LIKERT RESPONSES BY GENDER

**Relevant Data Columns & Setup**:
- The “Gender” column in the CSV indicates each teacher’s reported gender (e.g., Female, Male, Unknown).
- We again reference the **same five** Likert questions used in the Confidence/Advocacy composites:
  - "I frequently talk with my students about social justice issues."
  - "I know how to help my students communicate persuasively about social justice issues."
  - "I know how to help my students feel confident."
  - "I know how to help my students build their critical consciousness."
  - "I push my school leadership to integrate social justice education into our core curriculum."
- We focus here only on the **pre-survey** or “onboarding_df,” grouping by gender.

**Code Approach**:
- The code groups the onboarding dataframe by “Gender,” then computes the mean Likert score for each of the five questions within each gender group.
- It plots a **grouped bar chart** with question labels on the x-axis and mean scores on the y-axis, colored by gender (Female, Male, Unknown).

**Results Summary**:
- The chart typically shows 3 bars for each question, one per gender category, each bar labeled with its average Likert score.
- It visually compares whether, for instance, Female respondents rated themselves higher on “helping students feel confident” vs. Male or Unknown.
- Differences may appear across certain items; others may be similar across genders.
- Generally, it reveals if there are any notable gender-based variations in attitudes or self-reported abilities prior to the program.
"""

###############################################
# CREATE_GRAPH_CHAT (unchanged from your code)
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
    merged_teachers = pd.merge(
        onboarding_df[["Username","Confidence_Composite","Advocacy_Composite"]],
        post_program_df[["Username","Confidence_Composite","Advocacy_Composite"]],
        on="Username", how="inner", suffixes=("_Pre","_Post")
    )
    if len(merged_teachers)>0:
        merged_teachers["Confidence_Change"] = (
            merged_teachers["Confidence_Composite_Post"]
            - merged_teachers["Confidence_Composite_Pre"]
        )
        merged_teachers["Advocacy_Change"]   = (
            merged_teachers["Advocacy_Composite_Post"]
            - merged_teachers["Advocacy_Composite_Pre"]
        )

        fig_conf, ax_conf = plt.subplots(figsize=(4,4))
        sorted_conf = merged_teachers.sort_values("Confidence_Change")
        colors_conf = sorted_conf["Confidence_Change"].apply(lambda x: "green" if x>=0 else "red")
        ax_conf.barh(sorted_conf["Username"], sorted_conf["Confidence_Change"], color=colors_conf, edgecolor="black")
        ax_conf.axvline(0, color="black", linewidth=1)
        ax_conf.set_title("Confidence Change", fontsize=10, fontweight='bold')
        ax_conf.set_xlabel("Change in Score")

        fig_adv, ax_adv = plt.subplots(figsize=(4,4))
        sorted_adv = merged_teachers.sort_values("Advocacy_Change")
        colors_adv = sorted_adv["Advocacy_Change"].apply(lambda x: "green" if x>=0 else "red")
        ax_adv.barh(sorted_adv["Username"], sorted_adv["Advocacy_Change"], color=colors_adv, edgecolor="black")
        ax_adv.axvline(0, color="black", linewidth=1)
        ax_adv.set_title("Advocacy Change", fontsize=10, fontweight='bold')
        ax_adv.set_xlabel("Change in Score")

        fig_teacher, (ax1, ax2) = plt.subplots(1,2, figsize=(6,4), sharey=True)
        sorted_conf.plot(
            kind="barh",
            x="Username",
            y="Confidence_Change",
            color=sorted_conf["Confidence_Change"].apply(lambda x:"green" if x>=0 else "red"),
            edgecolor="black",
            ax=ax1
        )
        ax1.axvline(0, color="black")
        ax1.set_title("Confidence Change", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Change in Score")

        sorted_adv.plot(
            kind="barh",
            x="Username",
            y="Advocacy_Change",
            color=sorted_adv["Advocacy_Change"].apply(lambda x:"green" if x>=0 else "red"),
            edgecolor="black",
            ax=ax2
        )
        ax2.axvline(0, color="black")
        ax2.set_title("Advocacy Change", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Change in Score")

        fig_teacher.tight_layout()

        purpose_text_teacher = """
        **Purpose**: Display each teacher’s personal change in the composites (post - pre).  
        **Why It’s Helpful**: Emphasizes variation—some teachers might have risen substantially, others dropped. 
        """

        avg_conf_change = merged_teachers["Confidence_Change"].mean()
        avg_adv_change  = merged_teachers["Advocacy_Change"].mean()
        teacher_change_numeric_summary = summarize_chart_data(
            "",
            {
                "Avg_Confidence_Change": avg_conf_change,
                "Min_Confidence_Change": merged_teachers["Confidence_Change"].min(),
                "Max_Confidence_Change": merged_teachers["Confidence_Change"].max(),
                "Avg_Advocacy_Change": avg_adv_change,
                "Min_Advocacy_Change": merged_teachers["Advocacy_Change"].min(),
                "Max_Advocacy_Change": merged_teachers["Advocacy_Change"].max(),
            }
        )
        chart_context_teacherchange = teacherchange_context_text + "\n\n" + teacher_change_numeric_summary

        with st.spinner("Generating Individual Teacher Change Chart..."):
            time.sleep(1)

        create_graph_chat(
            heading="Individual Teacher Change in Composite Scores",
            purpose_text=purpose_text_teacher,
            figure=fig_teacher,
            session_key="chat_teacherchange",
            chat_context=chart_context_teacherchange
        )

###############################################
# 6. THIRD GRAPH: Percentage of Teachers Improved (100% Stacked)
###############################################
question_list = confidence_cols + advocacy_cols
merged_questions = pd.merge(
    onboarding_df[["Username"]+question_list],
    post_program_df[["Username"]+question_list],
    on="Username", how="inner", suffixes=("_pre","_post")
)
change_dict = {}
for q in question_list:
    pre_col = q + "_pre"
    post_col= q + "_post"
    df_temp= merged_questions[[pre_col, post_col]].dropna()
    if len(df_temp)>0:
        diff= df_temp[post_col] - df_temp[pre_col]
        improved = (diff>0).sum()
        same = (diff==0).sum()
        declined= (diff<0).sum()
        total= len(diff)
        change_dict[q] = {
            "Improved": improved/total,
            "Same": same/total,
            "Declined": declined/total
        }
if change_dict:
    df_pct = pd.DataFrame(change_dict).T
    fig_pct, ax_pct = plt.subplots(figsize=(5,4))
    df_pct[["Improved","Same","Declined"]].plot(
        kind="barh", stacked=True, ax=ax_pct,
        color=["green","gray","red"], edgecolor="black"
    )
    ax_pct.set_xlim(0,1)
    ax_pct.set_xlabel("Proportion of Teachers")
    ax_pct.set_ylabel("Question")
    ax_pct.set_title("Percentage Improved / Same / Declined by Question", fontsize=10, fontweight="bold")
    for container in ax_pct.containers:
        lbls=[f"{val*100:.0f}%" if val>0.02 else "" for val in container.datavalues]
        ax_pct.bar_label(container, labels=lbls, label_type='center', color="white", fontsize=8)

    purpose_text_pct = """
    **Purpose**: Illustrate the proportion who improved / stayed same / declined for each question.  
    **Why It’s Helpful**: Drills down into which items had the most improvement.
    """

    sample_dict = {}
    for i, k in enumerate(df_pct.index):
        if i<3:
            improved_pct = df_pct.loc[k,"Improved"]*100
            same_pct     = df_pct.loc[k,"Same"]*100
            declined_pct = df_pct.loc[k,"Declined"]*100
            sample_dict[k] = f"Improved={improved_pct:.0f}%, Same={same_pct:.0f}%, Declined={declined_pct:.0f}%"

    pct_numeric_summary = summarize_chart_data("", sample_dict)
    chart_context_pctteachers = pctchange_context_text + "\n\n" + pct_numeric_summary
    
    with st.spinner("Generating Percentage of Teachers Improved Chart..."):
        time.sleep(1)

    create_graph_chat(
        heading="Percentage of Teachers with Score Increases per Question",
        purpose_text=purpose_text_pct,
        figure=fig_pct,
        session_key="chat_pctteachers",
        chat_context=pct_numeric_summary
    )

###############################################
# 7. FOURTH GRAPH: Mean Likert by Gender
###############################################
def categorize_gender(pronouns):
    """Categorizes gender based on pronoun text."""
    if pd.isna(pronouns):
        return "Unknown"
    pronouns = pronouns.lower()
    if "she" in pronouns:
        return "Female"
    elif "he" in pronouns:
        return "Male"
    elif "they" in pronouns or "them" in pronouns:
        return "Non-Binary"
    return "Other/Unknown"

if "Full Name & Pronouns" in onboarding_df.columns:
    onboarding_df["Gender"] = onboarding_df["Full Name & Pronouns"].str.extract(r'\((.*?)\)')
    onboarding_df["Gender"] = onboarding_df["Gender"].apply(categorize_gender)
else:
    onboarding_df["Gender"] = "Unknown"

likert_cols = confidence_cols + advocacy_cols
onboarding_df[likert_cols] = onboarding_df[likert_cols].apply(pd.to_numeric, errors="coerce")

gender_mean_df = onboarding_df.groupby("Gender")[likert_cols].mean().reset_index()

if not gender_mean_df.empty:
    df_melted = gender_mean_df.melt(id_vars="Gender", var_name="Question", value_name="Mean Score")

    fig_gender, ax_g = plt.subplots(figsize=(6,4))
    sns.barplot(data=df_melted, x="Question", y="Mean Score", hue="Gender",
                palette="Set2", edgecolor="black", ax=ax_g)
    
    ax_g.set_xticklabels(
        [q if len(q) < 30 else q[:28] + "..." for q in df_melted["Question"].unique()],
        rotation=30, ha="right"
    )
    ax_g.set_ylim(0,5)
    ax_g.set_title("Mean Likert Responses by Gender", fontsize=10, fontweight="bold")
    
    for container in ax_g.containers:
        ax_g.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=8)

    purpose_text_gender = """
    **Purpose**: Compare average Likert responses by gender.
    **Why It’s Helpful**: Shows potential differences or similarities across demographics.
    """

    gender_summary = {}
    for g in gender_mean_df["Gender"]:
        row = gender_mean_df[gender_mean_df["Gender"] == g][likert_cols].squeeze()
        overall_avg = row.mean()
        gender_summary[g] = overall_avg

    gender_numeric_summary = summarize_chart_data("", gender_summary)
    chart_context_gender = gender_context_text + "\n\n" + gender_numeric_summary

    with st.spinner("Generating Mean Likert Responses by Gender Chart..."):
        time.sleep(1)

    create_graph_chat(
        heading="Mean Likert Responses by Gender",
        purpose_text=purpose_text_gender,
        figure=fig_gender,
        session_key="chat_gender",
        chat_context=chart_context_gender
    )

###############################################
# 8. FIFTH GRAPH: Student Post Survey (SAME LOGIC AS COLAB)
###############################################
# EXACT CODE FROM YOUR COLAB, adapted to 'students_post_df'
st.write("---")
st.subheader("Student Post Survey: Gender Analysis")

# 1) Define your buckets dictionary (as in your screenshot)
buckets = {
    "Main": [
        "Q1: Did you participate in Vocal Justice this year?",
        "Q2: Some question here",
        # etc... (Adjust these to match your actual columns)
    ],
    "Social Awareness Score": [
        "Q4: Another question",
        "Q5: Another question",
        # ...
    ],
    "Social Change Score": [
        "Q6: Another question",
        "Q7: Another question",
        # ...
    ]
}

# 2) A function that calculates each row's bucket averages
def get_bucket_scores(row):
    scores = {}
    for bucket_name, question_list in buckets.items():
        # average across the columns in question_list
        # If any column doesn't exist, this might raise KeyError
        # so make sure the columns match your CSV exactly
        val = row[question_list].mean()
        scores[bucket_name] = val
    return pd.Series(scores)

# 3) Create a new df_v with the buckets + keep "Gender Identity"
try:
    df_v = students_post_df.apply(get_bucket_scores, axis=1)
except KeyError as e:
    st.error(f"Column mismatch: {e}")
    st.stop()

if "Gender Identity" not in students_post_df.columns:
    st.warning("No 'Gender Identity' column found in Students Post Survey. Cannot produce this graph.")
else:
    df_v["Gender Identity"] = students_post_df["Gender Identity"]

    # 4) Group by Gender Identity for the 3 buckets
    try:
        gender_avg_scores = df_v.groupby("Gender Identity")[["Main","Social Awareness Score","Social Change Score"]].mean()
    except KeyError as e:
        st.error(f"Column mismatch in df_v: {e}")
        st.stop()

    # 5) Plot the grouped bar chart
    fig_student, ax_student = plt.subplots(figsize=(6,4))
    x = np.arange(len(gender_avg_scores.index))
    width = 0.2
    bucket_list = ["Main","Social Awareness Score","Social Change Score"]

    for i, col in enumerate(bucket_list):
        ax_student.bar(x + i*width, gender_avg_scores[col], width, label=col)

    ax_student.set_xticks(x + width*(len(bucket_list)-1)/2)
    ax_student.set_xticklabels(gender_avg_scores.index, rotation=30, ha="right")
    ax_student.set_ylabel("Average Score")
    ax_student.set_title("Gender Analysis: Average Bucket Scores")
    ax_student.legend()

    # Provide the question box on the right
    student_purpose_text = """
    **Purpose**: Analyze average bucket scores by gender identity from the Students Post Survey.
    **Why It’s Helpful**: Replicates your Colab logic exactly, showing how each 'bucket' differs by gender.
    """
    chart_context_students = (
        "This chart replicates the exact logic from your Colab code, computing average bucket scores "
        "for 'Main', 'Social Awareness Score', and 'Social Change Score' grouped by 'Gender Identity'."
    )

    with st.spinner("Generating Student Post Survey Gender Analysis..."):
        time.sleep(1)

    create_graph_chat(
        heading="Student Post Survey: Gender Analysis",
        purpose_text=student_purpose_text,
        figure=fig_student,
        session_key="chat_studentsurvey",
        chat_context=chart_context_students
    )

###############################################
# 9. DONE
###############################################
st.success("All analyses completed. Scroll up to review results and chat with each graph’s AI.")
