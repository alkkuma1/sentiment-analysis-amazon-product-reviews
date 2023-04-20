# import dash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# create a dash app 
app = dash.Dash(__name__)
server = app.server

def sentiment(merged_df):
    if merged_df['positive_review'] > merged_df['neutral_review'] and merged_df['positive_review'] > merged_df['negative_review']:
        return 'pos'
    elif merged_df['neutral_review'] > merged_df['positive_review'] and merged_df['neutral_review'] > merged_df['negative_review']:
        return 'neu'
    elif merged_df['negative_review'] > merged_df['positive_review'] and merged_df['negative_review'] > merged_df['neutral_review']:
        return 'neg'
    else:
        return 'neu'
    
def get_merged_df():
    merged_df = pd.read_csv("../data/processed/electronics_zero_shot_merged.csv")
    #convert unixReviewTime with datatyoe int64 into datetime object
    merged_df['unixReviewTime'] = pd.to_datetime(merged_df['unixReviewTime'], unit='s')
    #get only date of unixReviewTime
    merged_df['unixReviewTime'] = merged_df['unixReviewTime'].dt.date
    merged_df['positive_review'] = merged_df['positive_review'].round(1)
    merged_df['neutral_review'] = merged_df['neutral_review'].round(1)
    merged_df['negative_review'] = merged_df['negative_review'].round(1)
    merged_df['sentiment'] = merged_df.apply(sentiment, axis=1)
    merged_df['overall'] = merged_df['overall'].apply(lambda x: 'neg' if x == 1 or x == 2 else ('neu' if x == 3 else 'pos'))
    return merged_df

def word_cloud_pos():
    df = get_merged_df()
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
    pos_reviews = df[df['sentiment'] == 'pos']
    pos_reviews = pos_reviews['reviewText'].to_string()
    stopwords_l = pos_reviews.split()
    stopwords_l = [word.translate(str.maketrans('', '', '.,!?()[]{}:;\"\'`+-*/^$#@%&~_|0123456789')) for word in stopwords_l]
    stopwords = set(list(filter(None, stopwords)))
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(pos_reviews)
    px.imshow(wordcloud)
    fig = px.imshow(wordcloud)
    fig.update_layout(title='Positive Reviews Word Cloud')
    return fig

def word_cloud_neg():
    df = get_merged_df()
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
    neg_reviews = df[df['sentiment'] == 'neg']
    neg_reviews = neg_reviews['reviewText'].to_string()
    stopwords_l = neg_reviews.split()
    stopwords_l = [word.translate(str.maketrans('', '', '.,!?()[]{}:;\"\'`+-*/^$#@%&~_|0123456789')) for word in stopwords_l]
    stopwords = set(list(filter(None, stopwords)))
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(neg_reviews)
    px.imshow(wordcloud)
    fig = px.imshow(wordcloud)
    fig.update_layout(title='Negative Reviews Word Cloud')
    return fig


def word_cloud_neu():
    df = get_merged_df()
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
    neu_reviews = df[df['sentiment'] == 'neu']
    neu_reviews = neu_reviews['reviewText'].to_string()
    stopwords_l = neu_reviews.split()
    stopwords_l = [word.translate(str.maketrans('', '', '.,!?()[]{}:;\"\'`+-*/^$#@%&~_|0123456789')) for word in stopwords_l]
    stopwords = set(list(filter(None, stopwords)))
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(neu_reviews)
    px.imshow(wordcloud)
    fig = px.imshow(wordcloud)
    fig.update_layout(title='Neutral Reviews Word Cloud')
    return fig

def prediction():
    df = get_merged_df()
    import plotly.express as px
    df = df.groupby(['overall', 'sentiment']).size().unstack().apply(lambda x: x/x.sum(), axis=1).reset_index()
    df = pd.melt(df, id_vars=['overall'], value_vars=['neg', 'neu', 'pos'], var_name='sentiment', value_name='percentage')
    fig = px.bar(df, x='overall', y='percentage', color='sentiment', color_discrete_map={0: 'red', 1: 'blue', 2: 'green'}, barmode='stack',
                 labels={'overall': 'Overall Rating', 'percentage': 'Percentage', 'sentiment': 'Sentiment'}
                 )
    fig.update_layout(legend_title='Sentiment', 
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="left",
                          x=0.01
                      ))
    return fig

constraints = ['#B34D4D', '#4DB3B3', '#1F77B4', '#FF7F0E', '#2CA02C']
def categorical_variable_summary(col_name):
    df = get_merged_df()
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Countplot', 'Percentage plot'),
                        specs=[[{"type": "xy"}, {"type": "domain"}]])
    fig.add_trace(go.Bar(x=df[col_name].value_counts().index, 
                         y=df[col_name].value_counts().values.tolist(), 
                         textposition='auto',
                         text = df[col_name].value_counts().values.tolist(),
                         name = col_name, 
                         marker = dict(color = constraints, line=dict(color='#DBE6EC'))))
    fig.add_trace(go.Pie(labels=df[col_name].value_counts().keys(),
                         values=df[col_name].value_counts().values.tolist(),
                         textposition='inside',
                         name = col_name,
                         marker = dict(colors = constraints)),
                  row=1, 
                  col=2)
    fig.update_layout(title=col_name,
                  yaxis=dict(title='Count'),
                  xaxis=dict(title=col_name),
                  yaxis2=dict(title='Percentage', side='right', overlaying='y'),
                  legend=dict(title=col_name),
                  yaxis_tickformat=',',
                  yaxis2_tickformat=',',
                  height=500,
                  margin=dict(l=50, r=50, t=100, b=50),
                  xaxis_tickangle=-45,
                  template='plotly_white')
    return fig

# create a dash app layout
app.layout = html.Div([
    html.H1('E-Commerce Sentiment Analysis', style={'textAlign': 'center'}),
    html.H2('Prediction Analysis of every Sentiment', style={'textAlign': 'center'}),
    dcc.Graph(
        figure=prediction(),
    ),
    html.H2('Time Charts:', style={'textAlign': 'center'}),
    dcc.Dropdown(id="select_year",
                 options=[
                     {"label": "2004", "value": 2004},
                     {"label": "2005", "value": 2005},
                     {"label": "2006", "value": 2006},
                     {"label": "2007", "value": 2007},
                     {"label": "2008", "value": 2008},
                     {"label": "2009", "value": 2009},
                     {"label": "2010", "value": 2010},
                     {"label": "2011", "value": 2011},
                     {"label": "2012", "value": 2012},
                     {"label": "2013", "value": 2013},
                     {"label": "2014", "value": 2014},
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2004,
                 style={'width': "40%"}
                 ),
    dcc.Graph(id='time_chart_fig'),
    html.Br(),
    html.H2('Categorical Value Analysis:', style={'textAlign': 'center'}),
    dcc.Graph(
        figure=categorical_variable_summary('positive_review'),
    ),
    dcc.Graph(
        figure=categorical_variable_summary('negative_review'),
    ),
    dcc.Graph(
        figure=categorical_variable_summary('neutral_review'),
    ),
    dcc.Graph(
        figure=categorical_variable_summary('overall'),
    ),
    html.H2('Word Clouds:', style={'textAlign': 'center'}),
    dcc.Graph(
        figure=word_cloud_pos(),
    ),
    dcc.Graph(
        figure=word_cloud_neu(),
    ),
    dcc.Graph(
        figure=word_cloud_neg(),
    )
])


@app.callback(
    Output('time_chart_fig', 'figure'),
    [Input('select_year', 'value')]
)
def time_chart(selected_year):
    df = get_merged_df()
    df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'])
    filtered_data = df[df['unixReviewTime'].dt.year == selected_year]
    # get stacked bar chart
    fig = px.bar(filtered_data, x="unixReviewTime", y=["positive_review", "neutral_review", "negative_review"], barmode="stack")
    return fig

# run the dash app
if __name__ == '__main__':
    app.run_server(debug=True)