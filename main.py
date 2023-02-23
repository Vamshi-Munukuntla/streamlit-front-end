import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


@st.cache_data
def get_data(filename):
    df = pd.read_csv('data/taxi_data.csv')
    return df


with header:
    st.title('Welcome to my first deployed streamlit app')
    st.text('In this project I look into the transactions of taxis in NYC..')

with dataset:
    st.header('NYC Taxi dataset')
    st.text('I found this dataset on uber.com...')

    taxi_data = get_data('data/taxi_data.csv')
    # st.write(taxi_data.head())

    st.subheader("Pick-up location ID distribution on the NYC dataset")
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts())
    st.bar_chart(pulocation_dist)

with features:
    st.header('The features I created')

    st.markdown('* **first feature**: I created this feature because of this...'
                'I calculated this using this logic')

    st.markdown('* **second feature**: I created this feature because of this...'
                'I calculated this using this logic')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyper-parameters of the model and '
            'how the performance changes.')

    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What should be the max_depth of the model',
                               10, 100, 20, 10)
    n_estimators = sel_col.selectbox('How many tress should be there be?',
                                     options=[100, 200, 300, 'No Limit'], index=0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature',
                                       "PULocationID")

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth,
                                     n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean Squared error', )
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R2_score', )
    disp_col.write(r2_score(y, prediction))
