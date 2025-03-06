import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import openai

###############################################
# 0. PAGE CONFIG & OPENAI KEY
###############################################
st.set_page_config(page_title="Vocal Justice Analysis w/ AI", layout="wide")
st.title("Vocal Justice Survey Analysis with AI Insights")

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

if not pre_file or not post_file:
    st.warning("Please upload both Pre and Post CSV files.")
    st.stop()

# Read the CSVs
onboarding_df = pd.read_csv(pre_file)
post_program_df = pd.read_csv(post_file)

st.write("## Data Preview")
col_a, col_b = st.columns(2)
with col_a:
    st.write("**Pre-Survey (Onboarding) - First 5 Rows**")
    st.dataframe(onboarding_df.head())
with col_b:
    st.write("**Post-Survey (Post) - First 5 Rows**")
    st.dataframe(post_program_df.head())

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
# ADDITION: Function to summarize chart data
###############################################
def summarize_chart_data(description, data_points):
    """
    Returns a short textual summary of the data to help the AI
    "visualize" what's in the chart. `description` is a short statement
    about what the chart is. `data_points` is a list or dict of
    relevant numeric data behind the chart.
    """
    if isinstance(data_points, dict):
        # If data_points is a dict of label -> numeric
        stats_text = ", ".join(f"{k}={v:.2f}" for k,v in data_points.items())
    elif isinstance(data_points, list):
        # If data_points is a list of numeric
        stats_text = ", ".join(f"{v:.2f}" for v in data_points)
    else:
        stats_text = str(data_points)

    summary = (
        f"{description}\nData summary: {stats_text}."
        " This is the numerical context to help you interpret the chart."
    )
    return summary

###############################################
# A HELPER: Creates the Chat for Each Graph
###############################################
def create_graph_chat(heading, purpose_text, figure, session_key, chat_context):
    """
    heading: Subheader text for this chart section
    purpose_text: Explanation of why it's helpful
    figure: The Matplotlib figure to display on the left
    session_key: Unique key to store the conversation in st.session_state
    chat_context: A short user prompt describing the chart/data for context
    """

    # Horizontal rule + Subheader
    st.write("---")
    st.subheader(heading)
    st.markdown(purpose_text)

    left_col, right_col = st.columns([1,1])

    # Left: Render the figure
    with left_col:
        if figure is not None:
            st.pyplot(figure)

    # Right: Chat container
    with right_col:
        # Minimal styling for bubble chat
        st.markdown(
            """
            <style>
            .chat-container {
                background-color: #f0f0f0;
                overflow-y: auto;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                max-height: 300px;
            }
            .chat-bubble {
                display: inline-block;
                padding: 8px 12px;
                margin: 6px 0;
                border-radius: 10px;
                max-width: 80%;
                clear: both;
            }
            .user-bubble {
                background-color: #007BFF;
                color: #fff;
                float: right;
                margin-right: 10px;
            }
            .assistant-bubble {
                background-color: #e2e2e2;
                color: #000;
                float: left;
                margin-left: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 1) Initialize chat if not exist
        if session_key not in st.session_state:
            st.session_state[session_key] = []
            # Add a system message describing the chart
            st.session_state[session_key].append({
                "role": "system",
                "content": (
                    "You are a helpful data analysis assistant. "
                    + chat_context
                )
            })

        # 2) Render existing messages
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state[session_key]:
            role = msg["role"]
            content = msg["content"]
            bubble_class = "assistant-bubble" if role=="assistant" else "user-bubble"
            if role=="system":
                bubble_class = "assistant-bubble"  # treat system as assistant
            st.markdown(
                f'<div class="chat-bubble {bubble_class}">{content}</div><div style="clear: both;"></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 3) Use a form so that text input and button submission happen together
        with st.form(key=f"{session_key}_form"):
            user_input = st.text_input("Your question:", key=f"{session_key}_input")
            submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            # Add user message
            st.session_state[session_key].append({"role": "user", "content": user_input})
    
            # If we have an API key, call ChatCompletion
            if openai_api_key:
                try:
                    # Build messages for API
                    msgs_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state[session_key]
                    ]
                    
                    # Initialize openai with given API key
                    # Instead of openai.OpenAI, we use openai directly
                    # because the library version might differ
                    openai.api_key = openai_api_key
                    
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=msgs_for_api,
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
# 4. CREATE FIRST GRAPH (Pre/Post Composite)
###############################################
sns.set_theme(style="whitegrid")
fig_prepost, axz = plt.subplots(1, 2, figsize=(3.5, 3), sharey=True)

# Confidence
conf_means = [pre_conf_mean, post_conf_mean]
conf_labels = ["Pre", "Post"]
conf_bars = axz[0].bar(conf_labels, conf_means, color=["skyblue", "coral"], edgecolor="black")
axz[0].set_title("Confidence", fontsize=10, fontweight='bold')
axz[0].set_ylim(0,5)
for bar in conf_bars:
    h = bar.get_height()
    axz[0].text(bar.get_x()+bar.get_width()/2, h+0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

# Advocacy
adv_means = [pre_adv_mean, post_adv_mean]
adv_labels = ["Pre", "Post"]
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

# Create a short numerical summary to pass to the AI:
prepost_summary = summarize_chart_data(
    "This chart shows two bar groups: one for Confidence (pre vs. post), another for Advocacy (pre vs. post).",
    {
        "Confidence_Pre": pre_conf_mean,
        "Confidence_Post": post_conf_mean,
        "Advocacy_Pre": pre_adv_mean,
        "Advocacy_Post": post_adv_mean
    }
)

chat_context_prepost = (
    f"{prepost_summary}"
)

create_graph_chat(
    heading="Pre/Post Composite Score Bar Chart",
    purpose_text=purpose_text_prepost,
    figure=fig_prepost,
    session_key="chat_prepost",
    chat_context=chat_context_prepost
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
        merged_teachers["Confidence_Change"] = merged_teachers["Confidence_Composite_Post"] - merged_teachers["Confidence_Composite_Pre"]
        merged_teachers["Advocacy_Change"]   = merged_teachers["Advocacy_Composite_Post"]   - merged_teachers["Advocacy_Composite_Pre"]

        # Confidence diverging bar
        fig_conf, ax_conf = plt.subplots(figsize=(4,4))
        sorted_conf = merged_teachers.sort_values("Confidence_Change")
        colors_conf = sorted_conf["Confidence_Change"].apply(lambda x: "green" if x>=0 else "red")
        ax_conf.barh(sorted_conf["Username"], sorted_conf["Confidence_Change"], color=colors_conf, edgecolor="black")
        ax_conf.axvline(0, color="black", linewidth=1)
        ax_conf.set_title("Confidence Change", fontsize=10, fontweight='bold')
        ax_conf.set_xlabel("Change in Score")

        # Advocacy diverging bar
        fig_adv, ax_adv = plt.subplots(figsize=(4,4))
        sorted_adv = merged_teachers.sort_values("Advocacy_Change")
        colors_adv = sorted_adv["Advocacy_Change"].apply(lambda x: "green" if x>=0 else "red")
        ax_adv.barh(sorted_adv["Username"], sorted_adv["Advocacy_Change"], color=colors_adv, edgecolor="black")
        ax_adv.axvline(0, color="black", linewidth=1)
        ax_adv.set_title("Advocacy Change", fontsize=10, fontweight='bold')
        ax_adv.set_xlabel("Change in Score")

        # We'll combine them in one figure with subplots horizontally
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
        Color-coding (up vs. down) highlights overall patterns quickly.
        """
        # Summarize data for AI:
        avg_conf_change = merged_teachers["Confidence_Change"].mean()
        avg_adv_change  = merged_teachers["Advocacy_Change"].mean()
        teacher_change_summary = summarize_chart_data(
            "Diverging bar charts show each teacher's difference in Confidence_Composite and Advocacy_Composite from pre to post.",
            {
                "Avg_Confidence_Change": avg_conf_change,
                "Min_Confidence_Change": merged_teachers["Confidence_Change"].min(),
                "Max_Confidence_Change": merged_teachers["Confidence_Change"].max(),
                "Avg_Advocacy_Change": avg_adv_change,
                "Min_Advocacy_Change": merged_teachers["Advocacy_Change"].min(),
                "Max_Advocacy_Change": merged_teachers["Advocacy_Change"].max(),
            }
        )

        chat_context_teacher = teacher_change_summary

        create_graph_chat(
            heading="Individual Teacher Change in Composite Scores",
            purpose_text=purpose_text_teacher,
            figure=fig_teacher,
            session_key="chat_teacherchange",
            chat_context=chat_context_teacher
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
    # Summarize data for AI:
    # We'll just show the first few lines of the data dictionary
    top_questions = list(change_dict.keys())[:3]  # example: just show first 3
    sample_dict = {k: change_dict[k] for k in top_questions}
    pct_summary = summarize_chart_data(
        "100% stacked bar chart for each question, showing proportions of teachers who improved, stayed same, or declined.",
        sample_dict
    )

    create_graph_chat(
        heading="Percentage of Teachers with Score Increases per Question",
        purpose_text=purpose_text_pct,
        figure=fig_pct,
        session_key="chat_pctteachers",
        chat_context=pct_summary
    )

###############################################
# 7. FOURTH GRAPH: Mean Likert by Gender
###############################################
if "Gender" in onboarding_df.columns:
    # We'll just plot from the onboarding data
    likert_cols = confidence_cols + advocacy_cols
    gender_mean_df = onboarding_df.groupby("Gender")[likert_cols].mean().reset_index()
    df_melted= gender_mean_df.melt(id_vars="Gender", var_name="Question", value_name="Mean Score")

    fig_gender, ax_g = plt.subplots(figsize=(5,4))
    sns.barplot(data=df_melted, x="Question", y="Mean Score", hue="Gender",
                palette="Set2", edgecolor="black", ax=ax_g)
    ax_g.set_xticklabels([x if len(x)<30 else x[:28]+".." for x in df_melted["Question"].unique()],
                         rotation=30, ha="right")
    ax_g.set_ylim(0,5)
    ax_g.set_title("Mean Likert by Gender (Onboarding)", fontsize=10, fontweight="bold")
    for container in ax_g.containers:
        ax_g.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=8)

    purpose_text_gender = """
    **Purpose**: Compare average Likert responses by gender.  
    **Why It’s Helpful**: Shows potential differences or similarities across demographics.
    """

    # Summarize data for AI:
    gender_summary = {}
    for g in gender_mean_df["Gender"]:
        row = gender_mean_df[gender_mean_df["Gender"] == g][likert_cols].squeeze()
        # Average across all columns for a quick summary
        overall_avg = row.mean()
        gender_summary[g] = overall_avg

    gender_chart_summary = summarize_chart_data(
        "Grouped bar chart of mean Likert by Gender, focusing on confidence/advocacy items in the Onboarding survey.",
        gender_summary
    )

    create_graph_chat(
        heading="Mean Likert Responses by Gender",
        purpose_text=purpose_text_gender,
        figure=fig_gender,
        session_key="chat_gender",
        chat_context=gender_chart_summary
    )

###############################################
# 8. DONE
###############################################
st.success("All analyses completed. Scroll up to review results and chat with each graph’s AI.")
