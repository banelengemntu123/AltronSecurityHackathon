import streamlit as st
from top_candidate_recommendation import top_candidate_app
from best_job_recommender import best_job_app

# Streamlit interface
st.title("Unified Streamlit App")

app_choice = st.selectbox("Choose an Application:", ("Top Candidate Recommender", "Best Job Recommender for Candidates"))

if app_choice == "Top Candidate Recommender":
    top_candidate_app()

elif app_choice == "Best Job Recommender for Candidates":
    best_job_app()