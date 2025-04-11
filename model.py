# import os
# import pandas as pd
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer, util

# # Load the pre-trained sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # -------------------------------------------
# # Step 1: Load the Job Descriptions CSV file
# # -------------------------------------------
# # Replace 'jds.csv' with your CSV file path and ensure the column name matches your data.
# jd_df = pd.read_csv("job_description.csv",encoding='latin1')
# # Assuming that the CSV has a column "jd_text" that holds the job description text.
# # jd_texts = jd_df['Job Title']['Job Description'].tolist()
# job_descriptions = jd_df['Job Description'].tolist()
# job_titles = jd_df['Job Title'].tolist()
# # Combine "Job Title" and "Job Description" into one string per row
# combined_jd_texts = (jd_df['Job Title'] + " " + jd_df['Job Description']).tolist()

# # Now combined_jd_texts is a list where each element is "Job Title" concatenated with "Job Description"
# print(combined_jd_texts)

# # -------------------------------------------
# # Step 2: Load and parse resume PDFs from folder
# # -------------------------------------------
# resume_folder = 'CVs1'
# resume_texts = []
# resume_files = []

# for filename in os.listdir(resume_folder):
#     if filename.endswith('.pdf'):
#         filepath = os.path.join(resume_folder, filename)
#         try:
#             reader = PdfReader(filepath)
#             text = ""
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#             resume_texts.append(text)
#             resume_files.append(filename)
#         except Exception as e:
#             print(f"Error reading {filename}: {e}")

# if not resume_texts:
#     raise ValueError("No resume texts were loaded; please check your PDF folder.")

# # -------------------------------------------
# # Step 3: Compute Embeddings for JD and Resumes
# # -------------------------------------------
# # Here we encode all JD texts. You can choose which JD to compare against.
# jd_embeddings = model.encode(combined_jd_texts, convert_to_tensor=True)
# resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

# # -------------------------------------------
# # Step 4: Compute Cosine Similarity for Matching
# # -------------------------------------------
# # For example, let’s pick the first JD in our list to compare to every resume.
# jd_index = 0
# selected_jd = combined_jd_texts[jd_index]
# print("Job Description Text:\n", selected_jd, "\n")

# # Compute cosine similarity between the chosen JD embedding and all resume embeddings.
# cosine_scores = util.cos_sim(jd_embeddings[jd_index], resume_embeddings)[0]

# # Organize scores along with file names for clarity.
# matches = []
# for file, score in zip(resume_files, cosine_scores):
#     matches.append((file, score.item()))

# # Sort matches by score in descending order
# matches.sort(key=lambda x: x[1], reverse=True)

# print("Matching scores for resumes (in descending order):")
# for file, score in matches:
#     print(f"{file}: {score:.4f}")



import os
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load CSV file with job details (Job Title and Job Description)
jd_df = pd.read_csv("job_description.csv", encoding='latin1')

# Combine "Job Title" and "Job Description" into one text for each row
jd_df['combined'] = jd_df['Job Title'] + " " + jd_df['Job Description']

# Convert combined job texts into a list
job_texts = jd_df['combined'].tolist()

# 3. Extract resume texts from PDF folder (if not already done)
resume_folder = 'CVs1'
resume_texts = []
resume_files = []

for filename in os.listdir(resume_folder):
    if filename.endswith('.pdf'):
        filepath = os.path.join(resume_folder, filename)
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            resume_texts.append(text)
            resume_files.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if not resume_texts:
    raise ValueError("No resume texts were loaded; please check your PDF folder.")

# 4. Create Training Examples
# Yahan hum assume kar rahe hain ki tumhare paas label data available hai.
# Agar ground truth available hai toh uske hisaab se label assign karo.
# For demonstration, hum sabhi job-resume pair ko dummy label 1.0 de rahe hain.

train_examples = []
for job_text in job_texts:
    for resume_text in resume_texts:
        # Dummy label 1.0; replace with actual label if available
        train_examples.append(InputExample(texts=[job_text, resume_text], label=1.0))

# 5. Create DataLoader and define Loss function for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 6. Fine-Tuning the Model
num_epochs = 1  # epochs ka number dataset ke size aur requirement ke hisaab se adjust karo
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=100)  # Warmup steps bhi adjust kar sakte ho

# Save the fine-tuned model
model.save("fine_tuned_model")
