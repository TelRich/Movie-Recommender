"""
@Created on:  Thursday, June 29, 2023, 3:00:53 AM

"""

import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

@st.cache_resource
def load_data():
    data1 = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
    data2 = pd.read_csv('dataset/credits.csv')
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

# extracting the contents
movie_md['genres'] = movie_md['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)])
# movie_md['genres'] + movie_cr['cast']
# type(movie_cr['cast'][0])
# movie_kw['keywords'] = movie_kw['keywords'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])

# change date data type
movie_md['release_date'] = pd.to_datetime(movie_md['release_date'], errors='coerce')
# dropped invalid id
movie_md = movie_md.drop([19730, 29503, 35587])
movie_md['id'] = movie_md['id'].astype('int')

# extract release year
movie_md['release_year'] = movie_md['release_date'].dt.year.fillna(0).astype('int')

# Merged File
merged_data = movie_md.merge(movie_cr.loc[:,['cast', 'id']], on='id')\
    .merge(movie_kw, on='id')

# Create content for content based recommendation
merged_data['content'] = merged_data['genres'] + merged_data['cast'] + merged_data['keywords']
# merged_data['content'] = merged_data['content'].apply(lambda x: ' '.join(x))

# sample = merged_data[:20001]

# count = CountVectorizer(stop_words = 'english')
# count_matrix = count.fit_transform(sample['content'])
# cosine_sim = cosine_similarity(count_matrix, count_matrix)

# indices = pd.Series(sample.index, index=sample['title']).drop_duplicates()


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

@st.cache_data
# Function to return top genre movies
def top_x_genre_movie(data, genre='Drama', m=m, val=100):
    data = data[data['genres'] == genre]
    filterd_movie_md = data[data['vote_count'] >= m].copy()
    filterd_movie_md['wr'] = filterd_movie_md.apply(WR, axis=1)
    top_x = filterd_movie_md.sort_values('wr', ascending=False).loc[:, 'title':'wr'].head(val).reset_index(drop=True)
    return top_x

# @st.cache_data
# # Fucntion to return recommended movie based on content
# def recommended_movie(movie_title, top_x, cosine_sim=cosine_sim):
#     movie_index = indices[movie_title]
#     pairwise_similarity_score = sorted(list(enumerate(cosine_sim[movie_index])), key=lambda x: x[1], reverse=True)
#     top_similar_movie = pairwise_similarity_score[1:top_x]
#     recom_movie = merged_data.iloc[[x[0] for x in top_similar_movie]].reset_index(drop=True)
#     return recom_movie.iloc[:, 4:10]

# d = recommended_movie('The Matrix', 3)
# d

# Hide index numbers
hide = """
  <style>
  thead tr th:first-child {display:none}
  tbody th {display:none}
  </style>
  """
  
st.markdown("<h1 style='text-align:center;'>Overall Top Movies</h1>", unsafe_allow_html=True)

with st.expander('Top Movies', True):
    st.empty()
    num1 = st.number_input('Enter top number', value=0, step=1)
    top = top_x_movie(movie_md, val=num1)
    top.index = top.index+1
    st.write(top)
    
st.markdown("<h1 style='text-align:center;'>Top Movies by Genres</h1>", unsafe_allow_html=True)

with st.expander('Top Movies by Genre', True):
    st.empty()
    num2 = st.number_input('Enter a number', value=0, step=1)
    gn_movie_md = movie_md.explode('genres')
    uniq_genre = gn_movie_md.genres.unique()
    text = ', '.join(str(item) for item in uniq_genre if not pd.isnull(item))
    st.markdown(text)
    typ = st.text_input('Enter a genre')
    top_genre = top_x_genre_movie(gn_movie_md, genre=typ.capitalize(), val=num2)
    top_genre.index = top.index + 1
    st.write(top_genre)
    
