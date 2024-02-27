# requirements

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#  dataset
raw_data = pd.read_csv('data.csv')

# start cleaning by removing timestamp in first column
raw_data = raw_data.iloc[: , 1:]
data = pd.DataFrame()

# organize columns
cat_values = {}
final_column = raw_data.columns[len(raw_data.columns) - 1]

# for each column...
for (colname, colval) in raw_data.iteritems():

  # check if col is already a number; if so, add col directly
  # to new dataframe and skip to next column
  if isinstance(colval.values[0], (np.integer, float)):
    data[colname] = raw_data[colname].copy()
    continue

  # structure: {0: "lilac", 1: "blue", ...}
  new_dict = {}
  val = 0 # first index per column
  transformed_col_vals = [] # new numeric datapoints

  # if not, for each item in that column...
  for (row, item) in enumerate(colval.values):
    
    # if item is not in this col's dict...
    if item not in new_dict:
      new_dict[item] = val
      val += 1
    
    # then add numerical value to transformed dataframe
    transformed_col_vals.append(new_dict[item])
  
  # reverse dictionary only for final col (0, 1) => (vals)
  if colname == final_column:
    new_dict = {value : key for (key, value) in new_dict.items()}

  cat_values[colname] = new_dict
  data[colname] = transformed_col_vals


# train the model

# select features and predicton; automatically selects last column as prediction
cols = len(data.columns)
num_features = cols - 1
x = data.iloc[: , :num_features]
y = data.iloc[: , num_features:]

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# instantiate the model (using default parameters)
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)


# article generation       

def get_feat():
  feats = [abs(x) for x in model.coef_[0]]
  max_val = max(feats)
  idx = feats.index(max_val)
  return data.columns[idx]
  
acc = str(round(metrics.accuracy_score(y_test, y_pred) * 100, 1)) + "%"
most_imp_feat = get_feat()
# info = get_article(acc, most_imp_feat)



# interface creation      

# predictor for generic number of features
def general_predictor(*args):
  features = []

  # transform categorical input
  for colname, arg in zip(data.columns, args):
    if (colname in cat_values):
      features.append(cat_values[colname][arg])
    else:
      features.append(arg)

  # predict single datapoint
  new_input = [features]
  result = model.predict(new_input)
  return cat_values[final_column][result[0]]

# add data labels to replace those lost via star-args


block = gr.Blocks()

with open('info.md') as f:
  with block:
    gr.Markdown(f.readline())
    gr.Markdown('Take the quiz to get a personalized recommendation using AI.')
    
    with gr.Row():
      with gr.Group():
        inputls = []
        for colname in data.columns:
          # skip last column
          if colname == final_column:
            continue
          
          # access categories dict if data is categorical
          # otherwise, just use a number input
          if colname in cat_values:
            radio_options = list(cat_values[colname].keys())
            inputls.append(gr.Dropdown(radio_options, type="value", label=colname))
          else:
            # add numerical input
            inputls.append(gr.Number(label=colname))
          gr.Markdown("<br />")
        
        submit = gr.Button("Click to see result!", variant="primary")
        gr.Markdown("<br />")
        output = gr.Textbox(label="Your recommendation:", placeholder="your recommendation will appear here")
        
        submit.click(fn=general_predictor, inputs=inputls, outputs=output)
        gr.Markdown("<br />")
        
        with gr.Row():
          with gr.Group():
            gr.Markdown(f"<h3>Accuracy: </h3>{acc}")
          with gr.Group():
            gr.Markdown(f"<h3>Most important feature: </h3>{most_imp_feat}")
        
        gr.Markdown("<br />")
        
        with gr.Group():
          gr.Markdown('''Model accuracy is based on the uploaded data.csv  to give an idication of model performance.''')
        
      with gr.Group():
        with open('info.md') as f:
          f.readline()
          gr.Markdown(f.read())

# show the interface
block.launch()