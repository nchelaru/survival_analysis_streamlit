import pandas as pd
import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import pickle
import seaborn as sns


df = pd.read_csv('https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_renamed.csv')



def sunburst_fig():
    df = pd.read_csv('https://github.com/nchelaru/data-prep/raw/master/telco_cleaned_renamed.csv')

    ## Get categorical column names
    cat_list = []

    for col in df.columns:
        if df[col].dtype == object:
            cat_list.append(col)

    ## Get all possible levels of every categorical variable and number of data points in each level
    cat_levels = {}

    for col in cat_list:
        levels = df[col].value_counts().to_dict()
        cat_levels[col] = levels

    ## Convert nested dictionary to dataframe
    nestdict = pd.DataFrame(cat_levels).stack().reset_index()

    nestdict.columns = ['Level', 'Category', 'Population']

    nestdict['Category'] = [s + ": " for s in nestdict['Category']]

    cat_list = nestdict['Category'].unique()

    empty_list = [None] * len(cat_list)

    pop_list = ['0'] * len(cat_list)

    df1 = pd.DataFrame()

    df1['Level'] = cat_list

    df1['Category'] = empty_list

    df1['Population'] = pop_list

    df = pd.concat([df1, nestdict])

    df['Population'] = df['Population'].astype(int)

    fig = go.Figure(go.Sunburst(
        labels=df['Level'],
        parents=df['Category'],
        values=df['Population'],
        leaf={"opacity": 0.4},
        hoverinfo='skip'
    ))

    fig.update_layout(width=700, height=900, margin = dict(t=0, l=0, r=0, b=0))

    return fig



