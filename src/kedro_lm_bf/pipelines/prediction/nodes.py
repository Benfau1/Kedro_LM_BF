import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def predict(data_to_predict: pd.DataFrame, trained_model) :
    predictions = trained_model.predict(data_to_predict)
    predictions_df = pd.DataFrame(predictions, 
                                  columns=[
                                      "after_exam_125_Hz",
                                      "after_exam_250_Hz",
                                      "after_exam_500_Hz",
                                      "after_exam_1000_Hz",
                                      "after_exam_2000_Hz",
                                      "after_exam_4000_Hz",
                                      "after_exam_8000_Hz"])

    return predictions_df
