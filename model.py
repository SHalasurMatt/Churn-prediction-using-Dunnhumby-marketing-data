
import numpy as np
import pandas as pd
import pickle as pl
import streamlit as st


def predict(month1, month2,month3,month4,month5,month6):
  with open(r"linearreg.pkl", "rb") as input_file:
    e = pl.load(input_file)
    input = [month1,month2,month3,month4,month5,month6]
    input=np.asarray(input)
    p=e.predict(input.reshape(1,-1))

  return p