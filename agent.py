# agent.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please create a .env file and add your key.")
genai.configure(api_key=api_key)

class MediTrackAgent:
    """
    Implements the hybrid AI system: a rule-based engine for safety and
    an LLM (Gemini) for summarization and advice, all managed by a
    stateful conversation manager.
    """
    def __init__(self):
        # Initialize the Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Stateful Conversation Manager: Initialize state in Streamlit's session
        if 'log_data' not in st.session_state:
            st.session_state.log_data = {"initial_symptoms": [], "details": {}}
        if 'symptoms_to_query' not in st.session_state:
            st.session_state.symptoms_to_query = []
        
        # Point instance variables to session state for easy access
        self.log_data = st.session_state.log_data
        self.symptoms_to_query = st.session_state.symptoms_to_query

        # Rule-Based Engine Components
        self.CRITICAL_SYMPTOMS = {
            "chest pain", "difficulty breathing", "trouble breathing", "can't breathe",
            "severe pain", "slurred speech", "numbness", "face drooping",
            "unconscious", "loss of consciousness", "severe bleeding", "uncontrolled bleeding"
        }
        self.SYMPTOM_QUESTIONS = {
            "headache": ["On a scale of 1 to 10, how severe is it?", "Where is the pain located?"],
            "fever": ["Have you measured your temperature? If so, what is it?", "How long have you felt feverish?"],
            "cough": ["Is it a dry cough, or are you coughing anything up?", "How frequently are you coughing?"]
        }
        self.current_question_context = None

    def check_for_critical_symptoms(self, text: str) -> bool:
        """RULE-BASED ENGINE: Checks user input for critical keywords."""
        return any(keyword in text.lower() for keyword in self.CRITICAL_SYMPTOMS)

    def get_critical_symptom_response(self) -> str:
        """Returns the hard-coded safety response."""
        return (
            "**This sounds potentially serious.** Based on your description, "
            "it is highly recommended that you **contact a medical professional or emergency services immediately.** "
            "This chat will now end to ensure your safety."
        )

    def log_initial_symptoms(self, text: str):
        """STATEFUL MANAGER: Parses initial symptoms and populates the question queue."""
        detected = [s for s in self.SYMPTOM_QUESTIONS if s in text.lower()]
        self.log_data['initial_symptoms'] = detected
        # Create a queue of questions to ask
        self.symptoms_to_query = [(s, q) for s in detected for q in self.SYMPTOM_QUESTIONS[s]]
        st.session_state.symptoms_to_query = self.symptoms_to_query

    def ask_next_question(self) -> str or None:
        """STATEFUL MANAGER: Asks the next follow-up question."""
        if self.symptoms_to_query:
            symptom, question = self.symptoms_to_query.pop(0)
            self.current_question_context = (symptom, question)
            return question
        return None

    def log_detail(self, answer: str):
        """STATEFUL MANAGER: Logs the user's answer to a question."""
        if self.current_question_context:
            symptom, question = self.current_question_context
            if symptom not in self.log_data['details']:
                self.log_data['details'][symptom] = []
            self.log_data['details'][symptom].append({"question": question, "answer": answer})

    def _call_gemini(self, prompt: str) -> str:
        """LLM INTEGRATION: Helper function to call the Gemini API."""
        try:
            response = self.model.generate_content(prompt)
            # Handle cases where the model might not return text (e.g., safety blocks)
            return response.text if hasattr(response, 'text') else "The model could not generate a response for this query."
        except Exception as e:
            st.error(f"Error communicating with Gemini API: {e}")
            return "An error occurred while trying to generate a response."

    def generate_summary(self) -> str:
        """LLM INTEGRATION: Generates a structured summary of the log."""
        prompt = f"""
        Please summarize the following health log in a clean, easy-to-read markdown format.
        Use bullet points. Do not add any information not present in the log.
        Absolutely do not provide a medical diagnosis or any form of medical opinion.

        Health Log Data:
        {self.log_data}
        """
        return self._call_gemini(prompt)

    def generate_advice(self) -> str:
        """LLM INTEGRATION: Generates general, non-prescriptive wellness advice."""
        prompt = f"""
        Based on the following symptoms, provide some general, safe, non-prescriptive wellness advice.
        Focus only on common sense actions like rest, hydration, and monitoring symptoms.
        Explicitly state that this is not medical advice and a doctor should be consulted for medical issues.
        NEVER provide a diagnosis or recommend specific medication.

        Symptoms: {self.log_data['initial_symptoms']}
        """
        return self._call_gemini(prompt)
