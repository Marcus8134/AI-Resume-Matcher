# AI-Resume-Matcher

# Overview

AI Resume Matcher is a Streamlit application designed to automatically shortlist candidates by comparing candidate resumes (PDF format) with a job description. It leverages advanced Natural Language Processing techniques using Sentence Transformers to calculate similarity scores. The application uses a multi-agent approach to process multiple resumes concurrently, increasing efficiency. If a resume's similarity score exceeds a defined threshold, it also extracts matching keywords and sends out an interview invitation via email.



# Features

**Multi-Agent Processing:** Uses Python's concurrent.futures to process multiple resumes concurrently.

**Resume Matching:** Computes the similarity between the job description and each resume using a fine-tuned Sentence Transformer model.

**Keyword Extraction**: Displays matching keywords (extracted from both the job description and resume) alongside the similarity score.

**Email Invitation:** Automatically sends interview invitations to candidates whose resumes exceed a specified similarity threshold.

**User-Friendly Interface:** All inputs and configurations are available via the Streamlit sidebar.

# Requirements

Python 3.7 or higher

Streamlit

PyPDF2

SentenceTransformers

Required libraries: smtplib, email, re, concurrent.futures, string.

Install the necessary Python libraries with the following command:  pip install streamlit PyPDF2 sentence-transformers


# Installation

Clone the repository:

git clone https://github.com/Marcus8134/AI-Resume-Matcher


# Customization

**Model Tuning:**
The application uses a pre-trained Sentence Transformer model labeled 'fine_tuned_model'. You can change this to your fine-tuned model as required.

**Email Settings:**
Modify the SMTP settings (server, port, username, and password) in the code to match your email provider credentials.

**Keyword Extraction:**
The keyword extraction function uses a basic method to remove common stopwords and punctuation. Feel free to customize it to better suit your requirements.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

# License

This project is released under the MIT License.
