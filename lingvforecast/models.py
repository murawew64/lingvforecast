'''

'''
import pandas as pd
import numpy as np
from .ling_methods import ling_predict
from typing import Sequence


class LingvForecast:

    def __init__(self) -> None:
        self._data = []

    def fit(self, *data, top=1):
        '''
        :param data: array[array]

        Тут можно сделать преобразование данных
        для ускорения поиска результатов (оптимизация).
        '''
        assert top >= 1
        if top == 1:
            pass

        

    def predict(self, prow: Sequence, horizon: int = 1):
        pass
