import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies  as dd
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, Input, Model, layers

import sklearn.metrics as skm
from sklearn.utils import compute_class_weight

from PIL import Image as pilImage
import io

from base64 import decodebytes

import datetime

import pandas as pd

import numpy as np

import os

from flask import Flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', './assests/app.css']

server = Flask('mod4-dash')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# create a df with the model training info
df = pd.read_csv('model-train-info_sgd-01.csv')
df.rename(columns = {'Unnamed: 0' : 'Generation'}, inplace = True)
df = df.set_index('Generation')

#load in the classification model from the source files
modelPath = './final-model'  #os.Path('final-model')
model = keras.models.load_model(modelPath)

# create a graph object that shows the accuracy scores
accuracyFig = go.Figure()
accuracyFig.add_trace(go.Scatter(
  x=df.index, 
  y=df.accuracy,
  mode='lines+markers',
  name='Training Set Accuracy'))
accuracyFig.add_trace(go.Scatter(x=df.index, y=df.val_accuracy,
                    mode='lines+markers',
                    name='Validation Set Accuracy'))
accuracyFig.update_layout(
  title = {'text':'Training Accuracy Scores', 'font' : {'size': 24}},
  yaxis = {'title' : {'text' : 'Percent Correct', 'font' : {'size' : 18}}},
  xaxis = {'title' : {'text' : 'Generation', 'font' : {'size' : 18}}},
  legend = {
    'title' : {
      'text' : 'Legend',
      'font' : {'size': 16}
    }, 
    'font' : {'size' : 14}
  },  
  height = 500,
  hovermode ='x unified'
)

# create a graph object that shows the loss scores
lossFig = go.Figure()
lossFig.add_trace(go.Scatter(x=df.index, y=df.loss,
                    mode='lines+markers',
                    name='Training Set Loss'))
lossFig.add_trace(go.Scatter(x=df.index, y=df.val_loss,
                    mode='lines+markers',
                    name='Validation Set Loss'))
lossFig.update_layout(
  title = {
    'text':'Training Loss Scores', 
    'font' : {'size': 24}},
  yaxis = {
    'title' : {
      'text' : 'Score', 
      'font' : {'size' : 18}
    }
  },
  xaxis = {'title' : {'text' : 'Generation', 'font' : {'size' : 18}}},
  legend = {
    'title' : {
      'text' : 'Legend',
      'font' : {'size': 16}
    }, 
    'font' : {'size' : 14}},
  height = 500,
  hovermode ='x unified'
)

# create a graph object that shows the AUC scores
aucFig = go.Figure()
aucFig.add_trace(go.Scatter(x=df.index, y=df.auc,
                    mode='lines+markers',
                    name='Training Set AUC'))
aucFig.add_trace(go.Scatter(x=df.index, y=df.val_auc,
                    mode='lines+markers',
                    name='Validation Set AUC'))
aucFig.update_layout(
  title = {
    'text':'Training Area Under The Curve', 
    'font' : {'size': 24},
  },
  yaxis = {
    'title' : {
      'text' : 'Total Area', 
      'font' : {'size' : 18}
    }
  },
  xaxis = {
    'title' : {
      'text' : 'Generation', 
      'font' : {'size' : 18}
    }
  },
  legend = {
    'title' : {
      'text' : 'Legend',
      'font' : {'size': 16}
    }, 
    'font' : {'size' : 14}},
  height = 500,
  hovermode ='x unified'
)

app.layout = html.Div(
  children=[
    dcc.Tabs(
      style = {
        'width' : '60%',
        'margin' : 'auto',
        'font-size' : '24px',
      },
      className = 'tabs',
      children = [
        dcc.Tab(
          className = 'model_info_tab',
          label='Model Info',
          children= [        
            html.Div(
              children=[
                html.H1(
                  children = ['''The Model: Convolutional Neural Network'''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'center',
                    'margin': '2.5% auto',
                    'fontSize' : '3em',
                  },
                ),
                html.H3(
                  children = [
                    '''
                    General Model Details
                    '''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'center',
                    'margin': '1% auto',
                    'fontSize' : '2em',
                  },
                ),
                html.P(
                  children = [
                    '''
                    The model you will be intercating with on this app is a Tensorflow/Keras sequential model.
                    Its architechture is a convulutional neural network at the base with a few dense layers at
                    the head. This network has been tuned specifically for this problem and likely will not be a
                    good candidate for transfer learning. 

                    At its final iteration the model classified the set of test X-rays with 93% accuracy
                    and 97% recall for the pneumonia class. The testing set was composed of 437 images of roughly the same class imbalance as 
                    the training set.  The confusion matrix for the testing evaluation is below. Images that were of the pneumonia 
                    class are label "1" and non-pneumonia images were label "0". The model's preformace is optimized
                    for the medical industry, such that false negatives are to be avoided at the expense of false positives.
                    Ideally both recall scores would be equally great but in this case a small amount of performace
                    on the non-pneumonia case was sacrificed to further increase the ability to capture more patients
                    with pneumonia. 
                    '''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'left',
                    'margin': 'auto auto 1% auto',
                    'fontSize' : '1.2em',
                  },
                ),
                html.Img(
                  className = 'images',
                  src = "https://raw.githubusercontent.com/minthammock-development/mod4Project/master/second-model-confustion-matrix.png",
                  style = {
                    'width' : '30%',
                    'margin' : '1% 34%',
                    'textAlign' : 'center'

                  }
                ),
                html.P(
                  children = [
                    '''
                    The following graphs contain the information saved from the training process. 
                    For this application we chose loss, accuracy, and area under the curve. Both loss
                    and area under the curve are technical metrics that apply specifically to modeling
                    and may not be familiar. Accuracy is the most intuitive for overall scoring and
                    has the most direct implication for the medical industry. The others are included
                    for reference or curiosity sake. 
                    '''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'left',
                    'margin': 'auto auto 1% auto',
                    'fontSize' : '1.2em',
                  },
                ),
              ]
            ),
            dcc.Graph(
              id='accuracy_graph',
              figure=accuracyFig,
              style={
                'width': '60%',
                'lineHeight': 'auto',
                'textAlign': 'left',
                'margin': 'auto'
              }
            ),
            dcc.Graph(
              id='loss_graph',
              figure=lossFig,
              style={
                'width': '60%',
                'lineHeight': 'auto',
                'textAlign': 'left',
                'margin': 'auto',
                'fontSize' : '2em',
              },
            ),
            
            dcc.Graph(
              id='auc_graph',
              figure=aucFig,
              style={
                'width': '60%',
                'lineHeight': 'auto',
                'textAlign': 'left',
                'margin': 'auto'
              }
            ),
          ]
        ),
        dcc.Tab(
          className = 'predictions_tab',
          label = 'Predictions',
          children = [
            html.Div(
              children = [
                html.H1(
                  children = [
                    '''
                    Classify an Image
                    '''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'center',
                    'margin': '2.5% auto',
                    'fontSize' : '3em',
                  },
                ),
                html.P(
                  children = [
                    '''
                    If you would like to use the model to classify an x-ray please upload a 
                    black and white version (this should be the default for x-rays). The model 
                    will automatically classify the image after upload. Please note that this is 
                    a model is a tool to aid in classifying whether or not a patient's x-rays exhibit
                    pneumonia. In the event it is being used in a professional setting please be 
                    aware that the model is not perfect, and any predictions should be overseen and used
                    in conjustion with traditional medical professional tools and personnel.
                    '''],
                  style={
                    'width': '60%',
                    'lineHeight': 'auto',
                    'textAlign': 'center',
                    'margin': 'auto auto 2.5% auto',
                    'fontSize' : '1.2em',
                  },
                ),

                # This is the upload widget that productionized the model and auto predicts the class of the image uploaded.  
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
                      'width': '20%',
                      'height': '60px',
                      'lineHeight': '60px',
                      'borderWidth': '1px',
                      'borderStyle': 'dashed',
                      'borderRadius': '5px',
                      'textAlign': 'center',
                      'margin': 'auto',
                      'font-size': '20px' 
                  },
                  # Allow multiple files to be uploaded
                  multiple=True
                ),
              ]
            )
          ]
        )
      ]
    ),
    html.Div(id = 'prediction-output'),
    html.Div(id='output-image-upload'),
    dcc.Store(
      id = 'user-session',
    )
  ],
  className='app')

def parse_contents(content, filename):
  try:
    imageBytes = decodebytes(content.split(',')[1].encode('utf-8'))
    image = pilImage.open(io.BytesIO(imageBytes))
    image = image.convert('RGB')
    image = imageToDisplay = image.resize((256, 256), pilImage.NEAREST)
    image = img_to_array(image).reshape((1,256,256,3))

    print('fail 2')

    generator = ImageDataGenerator(
      rescale = 1./255)

    print('fail 5')
    pred = model.predict(image)
    label = np.where(model.predict(image) > .5, 'Pneumonia','Normal')
    print(pred)

    print('fail 6')
  except:
    print('The file image uploaded is not supported')
    preds = 'The file type you have uploaded is not supported for this model. Plese use: jpeg, png'

  return html.Div(
  children = [
    html.H4('File Name: '+filename),
    html.H5('The prediction for this image is: '+ str(label).replace('[', '').replace(']', '').replace("'", '')),
    html.H6('The calculated probability of having Pneunonia was: '+ str(pred).replace('[', '').replace(']', '').replace("'", '')),
    html.Hr(),
    html.Br(),

    # HTML images accept base64 encoded strings in the same format
    # that is supplied by the upload
    html.Img(src=imageToDisplay, id = filename),
    html.Hr(),],
  style={
        'width': '60%',
        'textAlign': 'center',
        'margin': 'auto'
    })

# callback to save the users image into the session as JSON
@app.callback(dd.Output('user-session', 'data'),
              dd.Output('output-image-upload', 'children'),
              dd.Input('upload-image', 'contents'),
              dd.State('upload-image', 'filename'))
def update_user_session(list_of_contents, list_of_names):
  # create an empty list to contin our dictonaries
  children = []

  # loop through the uploaded images and save the image to the users session in a dcc.Store
  children = []
  data = []
  if list_of_contents is not None:
    for content,name in zip(list_of_contents, list_of_names):

      # save each of the uploaded images and their file names into a dictonary (JSON)
      data.append({'content':content, 'name':name})
      children.append(parse_contents(content, name))

    return data, children
  else:
    return data, children

if __name__ == '__main__':
    app.run_server(debug=True)