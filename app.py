import streamlit as st
import os
from PyPDF2 import PdfReader
from email.message import EmailMessage
from sentence_transformers import SentenceTransformer, util
import smtplib
import re
import concurrent.futures
import string


#############################################################EmailAgent

def send_interview_email(recipient_email, candidate_filename, score):
    """
    Sends an email invitation for an interview.
    Update the SMTP settings with your actual details.
    """
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "upload your email"
    smtp_password = "upload your passkey"  
    sender_email = "upload your email"

    subject = "Interview Invitation"
    body = (f"Dear Candidate,\n\n"
            f"Congratulations ! Your resume ({candidate_filename}) has matched our job description with a similarity score of {score:.2f}%, "
            "and we would like to invite you for an interview.\n\n"
            "Please reply with your availability at your earliest convenience.\n\n"
            "Best regards,\n"
            "HR Team")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content(body)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        st.success(f"Email sent to {recipient_email}")
        return True
    except Exception as e:
        st.error(f"Failed to send email to {recipient_email}: {e}")
        return False



def extract_matching_keywords(jd_text, resume_text):
    """
    Extracts common keywords between JD and resume texts.
    Stopwords and punctuation are removed.
    """

    translator = str.maketrans("", "", string.punctuation)
    jd_clean = jd_text.translate(translator).lower()
    resume_clean = resume_text.translate(translator).lower()


    jd_words = set(jd_clean.split())
    resume_words = set(resume_clean.split())
    stopwords = set(["the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "for", "with", "in", "to", "from", "are", "have"])
    jd_keywords = jd_words - stopwords
    resume_keywords = resume_words - stopwords

    matching = jd_keywords.intersection(resume_keywords)
    return list(matching)


############################################### Agent Class: ResumeProcessingAgent

class ResumeProcessingAgent:
    def __init__(self, resume_file, model, jd_embedding):
        self.resume_file = resume_file
        self.model = model
        self.jd_embedding = jd_embedding
        self.result = None

    def process_resume(self):
        """Process the resume: extract text, compute embedding & similarity score."""
        try:
            pdf_reader = PdfReader(self.resume_file)
            resume_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    resume_text += page_text + "\n"

            resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
            score = util.cos_sim(self.jd_embedding, resume_embedding)[0][0].item()
            match_percentage = score * 100

            self.result = {
                "filename": self.resume_file.name,
                "score": match_percentage,
                "text": resume_text
            }
            return self.result
        except Exception as e:
            st.error(f"Error processing {self.resume_file.name}: {e}")
            return None


###################################Loading fine tuned Model

@st.cache_resource
def load_model():
    model = SentenceTransformer('fine_tuned_model')
    return model


###################################################Main App Function

def main():
    st.title("Multi-Agent AI Resume Matcher")
    st.write("Upload a job description and candidate resumes to automatically shortlist the most qualified candidates using multiple agents.")

    st.sidebar.title("Configuration")
    threshold = st.sidebar.slider("Match Threshold (%)", 0, 100, 75)
    
    st.sidebar.subheader("Job Description")
    jd_input_type = st.radio("Select JD input method:", ("Paste JD text", "Upload JD file"))
    jd_text = ""
    if jd_input_type == "Paste JD text":
        jd_text = st.text_area("Paste the job description here")
    else:
        uploaded_jd_file = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"])
        if uploaded_jd_file is not None:
            jd_text = uploaded_jd_file.read().decode("utf-8", errors='replace')

    st.sidebar.subheader("Candidate Resumes")
    uploaded_resumes = st.file_uploader("Upload one or more resumes (PDF)", type=["pdf"], accept_multiple_files=True)


    start_button = st.sidebar.button("Start Processing")
    reset_button = st.sidebar.button("Reset")

    if reset_button:
        try:
            st.experimental_rerun()
        except AttributeError:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.warning("App state reset. Please refresh the page manually.")
            return

    if start_button:

        if not jd_text.strip():
            st.error("Please provide a job description (paste text or upload a file).")
            return
        if not uploaded_resumes:
            st.error("Please upload at least one resume PDF.")
            return


        model = load_model()


        jd_embedding = model.encode(jd_text, convert_to_tensor=True)


################################################## Multi-Agent: Create ResumeProcessing Agents
        agents = []
        for resume_file in uploaded_resumes:
            agent = ResumeProcessingAgent(resume_file, model, jd_embedding)
            agents.append(agent)

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_agent = {executor.submit(agent.process_resume): agent for agent in agents}
            for future in concurrent.futures.as_completed(future_to_agent):
                result = future.result()
                if result is not None:
                    results.append(result)


        results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader("Matching Results")
        for res in results:
            is_above_threshold = res["score"] >= threshold
            status_icon = "✅" if is_above_threshold else "❌"

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{res['filename']}** - Similarity: {res['score']:.2f}% {status_icon}")
            with col2:

                keywords = extract_matching_keywords(jd_text, res["text"])
                if keywords:
 
                    st.info("Matching Keywords: " + ", ".join(keywords))
                else:
                    st.info("No matching keywords found")
                    

            if is_above_threshold:
                emails = re.findall(r"[a-zA-Z0-9\.\-+_]+@[a-zA-Z0-9\.\-+_]+\.[a-zA-Z]+", res["text"])
                if emails:
                    candidate_email = emails[0]
                    st.write(f"Sending interview invitation to: {candidate_email}")
                    send_interview_email(candidate_email, res["filename"], res["score"])
                else:
                    st.warning(f"No email found in {res['filename']}; could not send interview invitation.")

if __name__ == "__main__":
    main()
