# app.py
import streamlit as st
from agent import MediTrackAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="MediTrack AI (Gemini)",
    page_icon="🧑‍⚕️",
    layout="centered"
)

# --- Application Title and Disclaimer ---
st.title("🧑‍⚕️ MediTrack AI")
st.markdown("Powered by Google Gemini")

st.warning(
    "**Disclaimer:** I am an AI assistant, not a medical professional. "
    "This tool is for logging symptoms only. **If you are experiencing a medical emergency, call your local emergency number immediately.**"
)

# --- Agent Initialization ---
# This will automatically use the API key from the .env file
try:
    agent = MediTrackAgent()
except Exception as e:
    st.error(f"Failed to initialize the agent. Please check your API key setup. Error: {e}")
    st.stop()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"
    # Initial greeting message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm MediTrack AI. Let's log your health symptoms. How are you feeling today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Conversation Handler ---
def handle_chat_input():
    user_input = st.session_state.get("user_input")
    if not user_input:
        return

    # 1. Critical Symptom Check (Rule-Based Engine)
    if agent.check_for_critical_symptoms(user_input):
        critical_message = agent.get_critical_symptom_response()
        with st.chat_message("assistant"):
            st.error(critical_message) # Use Streamlit's error box for emphasis
        st.session_state.messages.append({"role": "assistant", "content": critical_message})
        st.session_state.stage = "done"
        return

    # 2. Stateful Conversation Flow
    current_stage = st.session_state.stage
    response = ""

    if current_stage == "greeting":
        agent.log_initial_symptoms(user_input)
        response = agent.ask_next_question()
        if not response: # No known symptoms found
            response = "I see. Could you please tell me more? Mention specific symptoms like 'headache', 'fever', etc., so I can ask relevant follow-up questions."
        else:
            st.session_state.stage = "collecting_details"

    elif current_stage == "collecting_details":
        agent.log_detail(user_input)
        response = agent.ask_next_question()
        if not response: # No more questions
            st.session_state.stage = "summary"
            with st.spinner('Analyzing and generating your summary...'):
                summary = agent.generate_summary()
                advice = agent.generate_advice()

            # Display final summary and advice
            summary_header = "### Your Health Log Summary"
            with st.chat_message("assistant"):
                st.markdown(summary_header)
                st.info(summary)
            st.session_state.messages.append({"role": "assistant", "content": f"{summary_header}\n{summary}"})

            advice_header = "### General Wellness Advice"
            with st.chat_message("assistant"):
                st.markdown(advice_header)
                st.success(advice)
            st.session_state.messages.append({"role": "assistant", "content": f"{advice_header}\n{advice}"})

            response = "Your session is complete. You can copy the summary above for your records. Stay well!"
            st.session_state.stage = "done"

    elif current_stage == "done":
        response = "This session has concluded. To start a new health log, please click the button below."

    # Display assistant response
    if response:
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- User Input Field ---
if st.session_state.stage != "done":
    if prompt := st.chat_input("Describe your symptoms..."):
        st.session_state.user_input = prompt
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        handle_chat_input()

# --- Reset Button ---
if st.button("Start New Log"):
    # Clear all session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
