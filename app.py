import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies  as dd
import plotly.express as px

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, Input, Model, layers

import sklearn.metrics as skm
from sklearn.utils import compute_class_weight

from PIL import Image as pilImage
import io

from base64 import decodestring

import datetime

import pandas as pd

import numpy as np

import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

modelPath = './final-model'  #os.Path('final-model')
model = keras.models.load_model(modelPath)



fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(
  children=[

    html.Div(
      children=[
        dcc.Markdown(
          children = [
            '''
            # The Model: Convulutional Neural Network

            ### General Model Details

            The model you will be intercating with on this app is a Tensorflow/Keras sequential model.
            It's architechture is a convulutional neural network at the base with a few dense layers at
            the head. This network has been tuned specifically for this problem and likely will not be a
            good candidate for transfer learning. 

            At it's final iteration the model classified the set of test X-rays with [INSERT THING] accuracy
            and a [INSERT THING] recall for the pneumonia class. 
            '''],
          style={
            'width': '60%',
            # 'height': '60px',
            'lineHeight': 'auto',
            # 'borderWidth': '1px',
            # 'borderStyle': 'dashed',
            # 'borderRadius': '5px',
            'textAlign': 'left',
            'margin': 'auto',
            'textSize' : '16px'
            },)]),
    dcc.Graph(
        id='example-graph',
        figure=fig,
        style={
            'width': '60%',
            # 'height': '60px',
            'lineHeight': 'auto',
            # 'borderWidth': '1px',
            # 'borderStyle': 'dashed',
            # 'borderRadius': '5px',
            'textAlign': 'left',
            'margin': 'auto'
            }),
    html.Div(children = [
      dcc.Markdown(children=[
        '''
          ### Classify an Image

          If you would like to use the model to classify an x-ray please upload a 
          black and white version (this should be the default for x-rays). The model 
          will automatically classify the image after upload. Please note that this is 
          a model is a tool to aid in classifying whether or not a patient's x-rays exhibit
          pneumonia. In the event it is being used in a professional setting please be 
          aware that the model is not perfect and any predictions should be overseen and used
          in conjustion with traditional medical professional tools and personnelle. 

        '''
      ],
      style={
            'width': '40%',
            # 'height': '60px',
            'lineHeight': 'auto',
            # 'borderWidth': '1px',
            # 'borderStyle': 'dashed',
            # 'borderRadius': '5px',
            'textAlign': 'left',
            'margin': 'auto'
            }),

      dcc.Upload(
        id='upload-image',
        children=[
          html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
          ]),
          html.Br()
        ],
        style={
            'width': '15%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto'
        },
        # Allow multiple files to be uploaded
        multiple=True
      ),
      html.Div(id = 'prediction-output'),
      html.Div(id='output-image-upload'),
      dcc.Store(
        id = 'user-session',)])
  
  ],
  className='app')

def parse_contents(contents, filename):
  

    return html.Div(
    children = [
      html.H4('File Name: '+filename),
      html.Hr(),
      html.Br(),
      # HTML images accept base64 encoded strings in the same format
      # that is supplied by the upload
      html.Img(src=contents, id = filename),
      html.Hr(),],
    style={
          'width': '60%',
          'textAlign': 'center',
          'margin': 'auto'
      })

@app.callback(dd.Output('user-session', 'data'),
              dd.Input('upload-image', 'contents'),
              dd.State('upload-image', 'filename'))
def update_user_session(list_of_contents, list_of_names):
  children = []
  if list_of_contents is not None:
    for content,name in zip(list_of_contents, list_of_names):
      children.append({'contents' : content, 'name' : name})
    return children
  else:
    return children


@app.callback(dd.Output('output-image-upload', 'children'),
              dd.Input('user-session', 'data'),)
def update_output(userData):
  if userData is not None:
    children = [
      parse_contents(decodestring(x['contents'].split(',')[1].encode('ascii')), x['name']) for x in userData]
    return children

# image = image_str.split(',')[1]
# data = decodestring(image.encode('ascii'))




# The following is a callback to allow the user to interact with the model in order to make predictions. 

# @app.callback(dd.Output('prediction-output', 'children'),
#               dd.Input(f'{}', ''),)
# def update_output(userData):
#   try:
#     print(contents)
#     procImg = load_img(
#       contents,
#       color_mode='grayscale',
#       target_size=(256,256))
#     arrayImg = img_to_array(procImg)

#     generator = ImageDataGenerator(
#       rescale = 1./255)

#     # we put the image into an Imagedatagenerator object so it plays nice with the model
#     finalImage = generator.flow(
#       arrayImg,  
#       batch_size = 1)

#     preds = np.where(model.predict(finalImage) > .5, 1,0)
#   except:
#     print('The file image uploaded is not supported')
#     preds = 'The file type you have uploaded is not supported for this model. Plese use: jpeg, png'
      


if __name__ == '__main__':
    app.run_server(debug=True)