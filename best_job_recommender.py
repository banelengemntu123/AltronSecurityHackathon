import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import numpy as np

# Sample data
candidates_df = pd.read_csv('candidate.csv')
job_roles_df = pd.read_csv('altron_roles.csv')
candidates_df['Name'] = candidates_df['First Name'] + ' ' + candidates_df['Last Name']

def best_job_app():

    def get_best_matching_job_role(candidate_name):
        # Get the skills of the selected candidate
        candidate_skills = candidates_df[candidates_df['Name'] == candidate_name]['Skills'].values[0]

        # Combine candidate skills with job role descriptions
        all_texts = job_roles_df['Skills required'].tolist() + [candidate_skills]

        # Create TF-IDF vectors
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

        # Compute cosine similarity between the candidate's skills and all job roles
        cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1])

        # Get the index of the most similar job role
        best_matching_index = cosine_similarities.argmax()

        # Return the best matching job role's details
        return job_roles_df.iloc[best_matching_index]

    # Streamlit interface
    st.title("Best Job Role Recommender for Candidates")

    selected_candidate_name = st.selectbox('Select Candidate', candidates_df['Name'].tolist())

    if st.button('Get Best Matching Job Role'):
        best_matching_job_role = get_best_matching_job_role(selected_candidate_name)
        st.write(f"**Best Matching Job Role for {selected_candidate_name}:**")
        st.write(f"Job Role Name: {best_matching_job_role['Position_Title']}")
        st.write(f"Skills: {best_matching_job_role['Skills required']}")
        st.write(f"Description: {best_matching_job_role['Description ']}")