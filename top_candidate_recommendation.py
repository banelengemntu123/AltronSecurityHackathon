import streamlit as st
import pandas as pd
import numpy as np
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


candidates_df = pd.read_csv('candidate.csv')
job_roles_df = pd.read_csv('altron_roles.csv')

def top_candidate_app():
    # Combine textual columns for candidates
    candidates_df['text'] = candidates_df['Job Title'] + ' ' + candidates_df['Skills']
    job_roles_df['text'] = job_roles_df['Position_Title'] + ' ' + job_roles_df['Skills required'] + ' ' + job_roles_df['Description ']

    # Combine textual columns for candidates
    candidates_df['text'] = candidates_df['Job Title'] + ' ' + candidates_df['Skills']
    job_roles_df['text'] = job_roles_df['Position_Title'] + ' ' + job_roles_df['Skills required'] + ' ' + job_roles_df['Description ']

    # Convert text data into vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    candidates_matrix = vectorizer.fit_transform(candidates_df['text'])

    def get_top_skills(candidate_index):
        # Extract the skills of the candidate
        candidate_skills = candidates_df.loc[candidate_index, 'Skills'].split(',')
    
        # Extract the skills of the job
        job_text = job_roles_df[job_roles_df['Position_Title'] == st.session_state.selected_job_role]['Skills'].iloc[0]
        job_skills = job_text.split(',')
    
        # Determine the intersection of the two skill sets
        common_skills = list(set(candidate_skills) & set(job_skills))
    
        # Return the top 3 skills or all common skills if less than 3
        return common_skills[:3]


    def get_top_candidates(job_title):
        # Extract textual data for the specified job title
        job_text = job_roles_df[job_roles_df['Position_Title'] == job_title]['text'].iloc[0]
    
        # Transform the job text into a vector
        job_vector = vectorizer.transform([job_text])
    
        # Calculate cosine similarity
        cosine_similarities = linear_kernel(job_vector, candidates_matrix)
    
        # Get the top 2 candidates' indices and scores
        sorted_indices = cosine_similarities[0].argsort()
        top_2_candidates_indices = sorted_indices[-2:][::-1]
        top_2_scores = cosine_similarities[0][top_2_candidates_indices]
    
        # Extract details of the top 2 candidates
        top_2_candidates = candidates_df.iloc[top_2_candidates_indices].copy()
        top_2_candidates['Cosine Similarity Score'] = top_2_scores
    
        return top_2_candidates[['First Name', 'Last Name', 'Job Title', 'Cosine Similarity Score']]
    st.title("Top Candidates Recommender")


    if "display_skills_for" in st.session_state:
        st.title(f"Top Skills for {st.session_state.display_skills_for}")
        top_skills = get_top_skills(st.session_state.candidate_index)
        st.write(", ".join(top_skills))
        if st.button("Back to Candidates"):
            del st.session_state.display_skills_for
    else:
        selected_job_role = st.selectbox('Select Job Role', job_roles_df['Position_Title'].tolist())
        st.session_state.selected_job_role = selected_job_role

    if st.button('Get Top Candidates'):
        top_candidates = get_top_candidates(selected_job_role)
    
        # Display candidates
        st.markdown(f"**Top Recommended Candidates for {selected_job_role}**")
    
        # 1st Candidate
        st.markdown(f"<div style='background-color:pink; padding:10px; border: 1px solid red; border-radius:5px;'>"
                    f"<p><b>Name:</b> {top_candidates['First Name'].iloc[0]}</p>"
                    f"<p><b>Surname:</b> {top_candidates['Last Name'].iloc[0]}</p>"
                    f"<p><b>Job Title:</b> {top_candidates['Job Title'].iloc[0]}</p>"
                    f"<p><b>Similarity Score:</b> {top_candidates['Cosine Similarity Score'].iloc[0]:.4f}</p>"
                    f"</div>", unsafe_allow_html=True)

        # 2nd Candidate
        st.markdown(f"<div style='background-color:red; padding:10px; border: 1px solid pink; border-radius:5px; margin-top:10px;'>"
                    f"<p><b>Name:</b> {top_candidates['First Name'].iloc[1]}</p>"
                    f"<p><b>Surname:</b> {top_candidates['Last Name'].iloc[1]}</p>"
                    f"<p><b>Job Title:</b> {top_candidates['Job Title'].iloc[1]}</p>"
                    f"<p><b>Similarity Score:</b> {top_candidates['Cosine Similarity Score'].iloc[1]:.4f}</p>"
                    f"</div>", unsafe_allow_html=True)
        
                # Add clickable functionality for each candidate
        if st.button(f"View skills for {top_candidates['First Name'].iloc[0]}"):
                st.session_state.display_skills_for = top_candidates['First Name'].iloc[0]
                st.session_state.candidate_index = top_candidates.index[0]
            
        if st.button(f"View skills for {top_candidates['First Name'].iloc[1]}"):
                st.session_state.display_skills_for = top_candidates['First Name'].iloc[1]
                st.session_state.candidate_index = top_candidates.index[1]