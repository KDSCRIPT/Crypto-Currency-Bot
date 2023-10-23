import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from scipy import stats
import tensorflow as tf
import numpy as np
#Define the window size
#(Based on how many days we are going to predict bitcoin prices)
#Define the horizon size
#(For how many days we are going to predict bitcoin prices)
WINDOW_SIZE=30
HORIZON=1
INTO_FUTURE=7
def make_future_forecasts(values,model,into_future=INTO_FUTURE,window_size=WINDOW_SIZE):
  future_forecast=[]
  last_window=values[-WINDOW_SIZE:]
  for _ in range(into_future):
    future_pred=model.predict(tf.expand_dims(last_window,axis=0))
    print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
    future_forecast.append(tf.squeeze(future_pred).numpy())
    last_window=np.append(last_window,future_pred)[-WINDOW_SIZE:]
  return future_forecast

