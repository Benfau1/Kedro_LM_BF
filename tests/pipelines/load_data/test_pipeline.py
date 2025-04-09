import pandas as pd
from kedro_lm_bf.pipelines.load_data import nodes

def test_generate_audiogram_data_returns_dataframe():
    df = nodes.generate_audiogram_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty