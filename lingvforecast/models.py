'''

'''
import pandas as pd
import numpy as np
from .ling_methods import ling_method, find_all_rows, CrudeForecast, TopForecast
from typing import Sequence


class LingvForecast:

    def __init__(self) -> None:
        # self._data = []
        pass

    def fit(self, *data, method='top', top=1, corrcoef=0.99):
        '''
        :param data: array[array]

        :param method: str, 'top'/'corrcoef'

        Тут можно сделать преобразование данных
        для ускорения поиска результатов (оптимизация).
        '''
        assert top >= 1

        self._data = data
        self._method = method
        self._top = top
        self._corrcoef = corrcoef

    def predict(self, prow: Sequence, horizon: int = 1):
        '''

        :return array of arrays
        '''
        if self._method == 'top':
            # TopForecast contains CrudeForecast array
            top_correlation_rows = TopForecast(self._top)
        elif self._method == 'corrcoef':
            top_correlation_rows = []

        # --- first step - find patterns
        # collect all the best subrows by correlation
        for el in find_all_rows(prow, self._data, corrcoef=self._corrcoef, horizon=horizon):
            top_correlation_rows.append(el)

        print('--- logs ---', top_correlation_rows)

        # sort values. At the beginning there will be elements with the maximum correlation coefficient
        # when calculating the correlation, the last element is removed, because it is an expert value
        # sort_refine_forecast = sorted(
        #     refine_forecast.get_result(), reverse=True, key=lambda val: row_corr(val.corrdata[:-1], corr_row))

        # --- second step - calculate forecast
        forecast = []
        for cr in top_correlation_rows:
            forecast.append(ling_method(prow, cr.fulldata))

        return forecast
