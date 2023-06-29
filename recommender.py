"""
@Created on:  Thursday, June 29, 2023, 3:00:53 AM

"""

import streamlit as st
import pandas as pd
import ast

# Setting th epage size and title
st.set_page_config(layout='centered', page_title='Movie Recommender System')

# App Setup
st.markdown("<h1 style='text-align:center;'>Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Brief Overview</h3>", unsafe_allow_html=True)

st.markdown("""<center>
A movie recommender system is a software or algorithm that suggests movies to users 
based on their preferences and viewing history. It utilizes data analysis and machine learning techniques 
to make personalized movie recommendations.
  </center>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data1 = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
    # data2 = pd.read_csv('dataset/credits.csv')
    data3 = pd.read_csv('dataset/keywords.csv')
    data4 = pd.read_csv('dataset/ratings_small.csv')
    data5 = pd.read_csv('dataset/links_small.csv')
    return [data1, data2, data3, data4, data5]

movie_md = load_data()[0]
movie_cr = load_data()[1]
movie_kw = load_data()[2]
movie_ra = load_data()[3]
movie_lk = load_data()[4]

# Selecting columns of interest
movie_md = movie_md[['genres', 'id', 'imdb_id', 'release_date', 'title', 'vote_average', 'vote_count', 'popularity', 'runtime']]

# extracting the genres
movie_md['genres'] = movie_md['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)])

# change date data type
movie_md['release_date'] = pd.to_datetime(movie_md['release_date'], errors='coerce')
# dropped invalid id
movie_md = movie_md.drop([19730, 29503, 35587])
movie_md['id'] = movie_md['id'].astype('int')

# extract release year
movie_md['release_year'] = movie_md['release_date'].dt.year.fillna(0).astype('int')


m = movie_md['vote_count'].quantile(0.9)
C = movie_md['vote_average'].mean()

@st.cache_data
# Function to calculate WR
def WR(data, m=m, C=C):
    v = data['vote_count']
    R = data['vote_average']
    wr = ((v/(v+m)*R) + (m/(v+m)*C)).round(2)
    return wr

@st.cache_data
# Function to return top overall movies
def top_x_movie(data, m=m, val=100):
    filterd_movie_md = data[data['vote_count'] >= m].copy()
    filterd_movie_md['wr'] = filterd_movie_md.apply(WR, 1)
    top_x = filterd_movie_md.sort_values('wr', ascending=False).loc[:, 'title':'wr'].head(val).reset_index(drop=True)
    return top_x

# Hide index numbers
hide = """
  <style>
  thead tr th:first-child {display:none}
  tbody th {display:none}
  </style>
  """

with st.expander('Overall Top Movies', True):
    num = st.number_input('Enter Top Number')
    top = top_x_movie(movie_md, val=10)
    st.write(top)
