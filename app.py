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
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.sidebar.info("Please enter your OpenAI API key to enable AI features.")

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
# 4. PLOT SMALLER BAR CHART + AI CHAT
###############################################
st.write("---")
st.subheader("Pre/Post Composite Score Bar Chart")

left_col, right_col = st.columns([1,1])

# -----------------------
# LEFT: THE SMALLER CHART
# -----------------------
with left_col:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3), sharey=True)

    # Confidence
    conf_means = [pre_conf_mean, post_conf_mean]
    conf_labels = ["Pre", "Post"]
    conf_bars = axes[0].bar(conf_labels, conf_means, color=["skyblue", "coral"], edgecolor="black")
    axes[0].set_title("Confidence", fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 5)
    for bar in conf_bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}",
                     ha="center", va="bottom", fontsize=8)

    # Advocacy
    adv_means = [pre_adv_mean, post_adv_mean]
    adv_labels = ["Pre", "Post"]
    adv_bars = axes[1].bar(adv_labels, adv_means, color=["skyblue", "coral"], edgecolor="black")
    axes[1].set_title("Advocacy", fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 5)
    for bar in adv_bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}",
                     ha="center", va="bottom", fontsize=8)

    st.pyplot(fig)

# -----------------------
# RIGHT: PURPOSE & AI CHAT
# -----------------------
with right_col:
    st.markdown("""
    **Purpose**: Show the aggregate changes in Confidence and Advocacy composite scores before vs. after the program.  
    **Why It’s Helpful**: Offers a quick, high-level overview of whether the group as a whole is trending up or down.
    """)

    st.markdown("### AI-Generated Analysis")

    def get_ai_summary(pre_conf, post_conf, pre_adv, post_adv):
        """
        Calls OpenAI to get a 100-word analysis of the pre/post means.
        """
        if not openai_api_key:
            return "OpenAI API key not provided."
        try:
            prompt = (
                "You are a data analysis assistant. Please write exactly 100 words analyzing the changes in two composites: "
                f"Confidence (pre={pre_conf:.2f}, post={post_conf:.2f}) and Advocacy (pre={pre_adv:.2f}, post={post_adv:.2f}). "
                "Discuss implications for the organization."
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    # Generate the AI summary (just once)
    summary_text = get_ai_summary(pre_conf_mean, post_conf_mean, pre_adv_mean, post_adv_mean)
    st.write(summary_text)

    st.markdown("### Ask More Questions")

    # 1) Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 2) FIXED CHAT CONTAINER HEIGHT (CHANGED FROM 1px TO 250px)
    st.markdown(
        """
        <style>
        .chat-container {
            height: 250px;  # CHANGED THIS LINE
            overflow-y: auto;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
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

    # 3) Function to render chat in bubble style
    def render_chat(messages):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble user-bubble">{msg["content"]}</div><div style="clear: both;"></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble assistant-bubble">{msg["content"]}</div><div style="clear: both;"></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # 4) Display existing conversation
    render_chat(st.session_state["messages"])

    # 5) SIMPLIFIED INPUT SECTION (REMOVED CONTAINER/COLUMNS)
    user_input = st.text_input(
        "Your question:", 
        placeholder="Type your message here..."
    )
    
    if st.button("Send"):
        if user_input.strip():
            # Append user message
            st.session_state["messages"].append({"role": "user", "content": user_input})

            if openai_api_key:
                # Build the messages for ChatCompletion
                chat_msgs = [{"role": "system", "content": "You are a helpful data analysis assistant."}]
                for m in st.session_state["messages"]:
                    chat_msgs.append({"role": m["role"], "content": m["content"]})

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=chat_msgs,
                        max_tokens=300,
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Error calling OpenAI: {e}"
                    })
            else:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "OpenAI API key not provided."
                })
