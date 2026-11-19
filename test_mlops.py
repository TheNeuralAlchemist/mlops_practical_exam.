
import numpy as np
from data import clean_data, normalize_data
from train import train_model

def test_clean_data():
    data = np.array([1, np.nan, 3])
    result = clean_data(data)
    assert not np.isnan(result).any()

def test_normalize_data():
    data = np.array([100, 200, 300])
    result = normalize_data(data)
    assert list(result) == [1, 2, 3]

def test_training():
    model = train_model()
    assert model is not None

def test_model_output():
    model = train_model()
    prediction = model.predict([[2]])
    assert prediction[0] in [0, 1]

