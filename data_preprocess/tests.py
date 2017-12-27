# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from data_preprocess.time_data import TimeDataPreprocess


class TimeDataPreprocessTestCase(unittest.TestCase):
    def setUp(self):
        self.time_series = pd.Series(list(np.random.uniform(50, 51, size=313)))
    
    def tearDown(self):
        del self.time_series
    
    def test_transform_time_window(self):
        window_size = 12
        offset = 4
        df = TimeDataPreprocess.transform_time_window(self.time_series, window_size, offset, True)
        if len(self.time_series) > window_size:
            record_num = (len(self.time_series) - window_size) // offset + 1
        else:
            record_num = 0
        self.assertEqual(len(df), record_num)
    
    def test_extract_feature(self):
        extract_feature_num = 7
        vector = self.time_series[0:1].values.tolist()
        extract_vector = TimeDataPreprocess.extract_feature(vector)
        if len(vector) > 1:
            self.assertEqual(len(extract_vector), extract_feature_num)
        else:
            self.assertEqual(extract_vector, vector)

if __name__ == 'main':
    unittest.main()
