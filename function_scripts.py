# Function to calculate WR
def WR(data, m=m, C=C):
    v = data['vote_count']
    R = data['vote_average']
    wr = ((v/(v+m)*R) + (m/(v+m)*C)).round(2)
    return wr

# Function to return top overall movies
def top_x_movie(data, m=m, val=100):
    filterd_movie_md = data[data['vote_count'] >= m].copy()
    filterd_movie_md['wr'] = filterd_movie_md.apply(WR, 1)
    top_x = filterd_movie_md.sort_values('wr', ascending=False).loc[:, 'title':'wr'].head(val).reset_index(drop=True)
    return top_x

# Function to return top genre movies
def top_x_genre_movie(data, genre='Drama', m=m, val=100):
    data = data[data['genres'] == genre]
    filterd_movie_md = data[data['vote_count'] >= m].copy()
    filterd_movie_md['wr'] = filterd_movie_md.apply(WR, axis=1)
    top_x = filterd_movie_md.sort_values('wr', ascending=False).loc[:, 'title':'wr'].head(val).reset_index(drop=True)
    return top_x