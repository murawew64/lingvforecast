'''

'''
import pandas as pd
import numpy as np
from .ling_methods import row_corr, ling_method, find_all_rows, CrudeForecast, TopForecast
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
        self._candidates = []

    def _group_curves_by_correlation(self, curves, threshold=0.95):
        groups = []
        for i, curve in enumerate(curves):
            if i == 0:
                groups.append([curve])
                continue
            added_to_group = False
            for group in groups:
                if any(row_corr(curve, c) >= threshold for c in group):
                    group.append(curve)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([curve])
        return groups

    def _calculate_result_forecast(self, candidates: Sequence[CrudeForecast]) -> Sequence:
        '''
        candidates - отсортированы по коэффициенту подобия от большего к меньшему.
        '''
        if len(candidates) < 2:
            return candidates

        # сделать кластеризацию по коэффициенту корреляции
        groups = self._group_curves_by_correlation(candidates)

        # пройтись по всем подмножествам, выбрать то, у которого больше мощность
        group_length = [len(g) for g in groups]
        max_power_index = group_length.index(max(group_length))
        max_power_group = groups[max_power_index]

        print(f'------ Max power = {max(group_length)} -----')

        # у первого элемента наибольший коэффициент корреляции
        # с прогнозируемой кривой, так как при разбиении на группы
        # упорядоченность сохраняется
        return max_power_group[0]

    def get_candidates(self):
        return self._candidates

    def predict(self, prow: Sequence, horizon: int = 1) -> Sequence:
        '''

        :return array of arrays
        '''
        assert horizon >= 1

        if self._method == 'top':
            # TopForecast contains CrudeForecast array
            top_correlation_rows = TopForecast(self._top)
        elif self._method == 'corrcoef':
            top_correlation_rows = []

        # --- first step - find patterns
        # collect all the best subrows by correlation
        for el in find_all_rows(prow, self._data, corrcoef=self._corrcoef, horizon=horizon):
            top_correlation_rows.append(el)

        # sort curves. At the beginning there will be elements with the maximum correlation coefficient.
        top_correlation_rows.sort(reverse=True, key=lambda el: el.corrcoef)

        # --- second step - calculate candidates
        candidates = []
        for cr in top_correlation_rows:
            candidates.append(ling_method(prow, cr.fulldata))

        # list with candidates
        self._candidates = candidates

        return self._calculate_result_forecast(candidates)
