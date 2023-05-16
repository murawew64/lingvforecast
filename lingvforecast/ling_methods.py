'''

'''

import numpy as np
import math
import pandas as pd
from . import proc_methods
from collections import namedtuple
from typing import Sequence

CrudeForecast = namedtuple(
    'CrudeForecast', ['corrdata', 'fordata', 'fulldata', 'corrcoef'])

# Forecast = namedtuple('Forecast', ['forecast', 'info'])


class TopForecast:

    def __init__(self, top):
        self._top = top
        self._crude_forecast = []

    def append(self, el: CrudeForecast):
        assert isinstance(
            el, CrudeForecast), 'el - object of CrudeForecast class!'
        self._add_top_corr(self._crude_forecast, el)

    def __str__(self) -> str:
        return str(self._crude_forecast)

    def __repr__(self) -> str:
        return str(self._crude_forecast)

    def __iter__(self):
        return iter(self._crude_forecast)

    def sort(self, reverse=False, key=None):
        self._crude_forecast.sort(reverse=reverse, key=key)

    def _add_top_corr(self, values, new_val):
        '''

        values - array type content 'Forecast' and 'None' objects.
        new_val - object of 'Forecast' class
        '''
        # if array content empty place just append
        if len(self._crude_forecast) < self._top:
            self._crude_forecast.append(new_val)
        # if array dont content empty place
        # insert new item at place where old item corrcoef less then new item corrcoef
        else:
            # first - get minimum corr coef in sequence
            min_coef = values[0].corrcoef
            index = 0
            for i, val in enumerate(values):
                if val.corrcoef < min_coef:
                    min_coef = val.corrcoef
                    index = i
            # second - try to change element with minimum corrcoef
            if min_coef < new_val.corrcoef:
                self._crude_forecast[index] = new_val


# forecast


def ling_predict(prow, frows, corrcoef=0.95, horizon=1, hmethod=None, normalize=False):
    '''
    prow - row to be predicted

    frows - array of arrays with which forecast is made

    corrcoef - correlation coefficient.

    if `method` is None will be used median function.

    hmethod - handler method

    normalize - parametr that provide manipulation with normalize forecast (not with result)

    horizon - if horizon equals 1 return one value if great then 1 return array
    of values.
    '''

    assert not np.isscalar(frows) or not np.isscalar(
        frows[0]), "frows - array of arrays!"

    assert -1 <= corrcoef <= 1, "Incorrect corrcoef!"

    if not hmethod:
        hmethod = proc_methods.median

    # array for predicting multiple steps
    if not isinstance(prow, (np.ndarray)):
        prow_arr = np.array(prow)
    else:
        prow_arr = prow
    # forecast array
    res_forecast = np.zeros(horizon)

    # method is used with NORMALIZE forecast values
    if normalize:
        for i in range(horizon):
            forecast = []
            # additional information
            add_info = {}

            for el in find_all_rows(prow_arr, frows, corrcoef=corrcoef, horizon=1):
                # normalize y-row
                fr = _normalize(el.fordata, np.mean(
                    el.corrdata), np.var(el.corrdata))[0]
                add_info[id(fr)] = el
                forecast.append(fr)

            # get result forecast
            res_forecast[i] = hmethod(forecast) * \
                math.sqrt(np.var(prow_arr)) + np.mean(prow_arr)
            # create new prow
            prow_arr = np.append(prow_arr[1:], [res_forecast[i]])

    # method is used with FINISH forecast values
    else:
        for i in range(horizon):
            forecast = []
            # additional information
            add_info = {}

            # predict next value
            for el in find_all_rows(prow_arr, frows, corrcoef=corrcoef, horizon=1):
                fr = ling_method(prow_arr, el.fulldata)
                add_info[id(fr)] = el
                forecast.append(fr)

            # get result forecast
            res_forecast[i] = hmethod(forecast)
            # create new prow
            prow_arr = np.append(prow_arr[1:], [res_forecast[i]])

    if horizon == 1:
        return res_forecast[0]
    else:
        return res_forecast


def ling_predict_top(prow, frows, corrcoef=0.5, top=99, horizon=1, hmethod=None, normalize=False):
    '''
    Use just `top` values to calculate result.

    prow - row to be predicted

    frows - array of arrays with which forecast is made

    corrcoef - correlation coefficient. 

    top - how many forecast set to hmethod (sort by correlation). Use in addition to corrcoef.

    if `method` is None will be used median function.

    hmethod - handler method

    normalize - parametr that provide manipulation with normalize forecast (not with result)

    horizon - if horizon equals 1 return one value if great then 1 return array
    of values.
    '''

    assert not np.isscalar(frows) or not np.isscalar(
        frows[0]), "frows - array of arrays!"

    top = int(top)

    assert -1 <= corrcoef <= 1, "Incorrect corrcoef!"
    assert 1 <= top, "Incorrect top!"

    if hmethod is None:
        hmethod = proc_methods.median

    # array for predicting multiple steps
    if not isinstance(prow, (np.ndarray)):
        prow_arr = np.array(prow)
    else:
        prow_arr = prow
    # forecast array with horizon steps
    res_forecast = np.zeros(horizon)

    for j in range(horizon):
        crude_forecast = TopForecast(top)
        for el in find_all_rows(prow_arr, frows, corrcoef=corrcoef, horizon=1):
            crude_forecast.append(el)

        crude_forecast_result = crude_forecast.get_result()
        forecast = np.zeros(len(crude_forecast_result))
        # method is used with NORMALIZE forecast values
        if normalize:
            for i, el in enumerate(crude_forecast_result):
                forecast[i] = _normalize(el.fordata, np.mean(
                    el.corrdata), np.var(el.corrdata))[0]

            # compute result forecast
            res_forecast[j] = hmethod(forecast) * \
                math.sqrt(np.var(prow_arr)) + np.mean(prow_arr)

        # method is used with FINISH forecast values
        else:
            for i, el in enumerate(crude_forecast_result):
                forecast[i] = ling_method(prow_arr, el.fulldata)

            # compute result forecast
            res_forecast[j] = hmethod(forecast)

        # create new prow
        prow_arr = np.append(prow_arr[1:], [res_forecast[j]])

    if horizon == 1:
        return res_forecast[0]
    else:
        return res_forecast


def ling_predict_frame(prows, frows, indices, corrcoef=0.5, top=99, normalize=False):
    '''
    Create DataFrame with forecast for expert review.
    Use only median in function. Prediction only on one step.

    prows: array of arrays with rows to be predicted

    frows: array of arrays with which forecast is made

    indices: indexes to compute medians. Each index describes column in result table.

    corrcoef: correlation coefficient. 

    top: how many forecast set to hmethod (sort by correlation). Use in addition to corrcoef.

    normalize: parametr that provide manipulation with normalize forecast (not with result)
    '''
    assert not np.isscalar(frows) or not np.isscalar(
        frows[0]), "frows - array of arrays!"

    assert not np.isscalar(prows) or not np.isscalar(
        prows[0]), "prows - array of arrays!"

    top = int(top)

    assert -1 <= corrcoef <= 1, "Incorrect corrcoef!"
    assert 1 <= top, "Incorrect top!"

    df_predict = pd.DataFrame(columns=indices)

    for prow_arr in prows:
        crude_forecast = TopForecast(top)
        for el in find_all_rows(prow_arr, frows, corrcoef=corrcoef, horizon=1):
            crude_forecast.append(el)

        # sort predicted values. At the beginning there will be elements with the maximum correlation coefficient
        sort_crude_forecast = sorted(
            crude_forecast.get_result(), reverse=True, key=lambda val: val.corrcoef)

        arr = [val.corrcoef for val in sort_crude_forecast]
        print(np.min(arr))

        forecast = np.zeros(len(sort_crude_forecast))
        # method is used with NORMALIZE forecast values
        if normalize:
            for i, el in enumerate(sort_crude_forecast):
                forecast[i] = _normalize(el.fordata, np.mean(
                    el.corrdata), np.var(el.corrdata))[0]

        # method is used with FINISH forecast values
        else:
            for i, el in enumerate(sort_crude_forecast):
                forecast[i] = ling_method(prow_arr, el.fulldata)

        # add row in DataFrame
        df_predict.loc[len(df_predict.index)] = _create_frame_row(
            forecast, indices)

    return df_predict


def ling_corr_frame(prows, frows, indices, corrcoef=0.5, top=99):
    '''
    Create DataFrame with forecast use prior information (refine forecast use expert reviews).
    Use only median in function. Prediction only on one step.

    prows: array of arrays with rows to be refined, last value in every row is expert value.

    frows: array of arrays with which forecast specify is made.

    indices: indexes to compute medians. Each index describes column in result table.

    corrcoef: correlation coefficient. 

    top: how many forecast set to hmethod (sort by correlation). Use in addition to corrcoef.
    '''
    assert not np.isscalar(frows) or not np.isscalar(
        frows[0]), "frows - array of arrays!"

    assert not np.isscalar(prows) or not np.isscalar(
        prows[0]), "prows - array of arrays!"

    top = int(top)

    assert -1 <= corrcoef <= 1, "Incorrect corrcoef!"
    assert 1 <= top, "Incorrect top!"

    df_refine = pd.DataFrame(columns=indices)

    for prow_arr in prows:

        # last value is expert value
        # method need specify it
        specify_val = prow_arr[-1]
        # row without expert value
        corr_row = prow_arr[:-1]

        # collect all the best subrows by correlation
        refine_forecast = TopForecast(top)
        for el in find_all_rows(prow_arr, frows, corrcoef=corrcoef, horizon=1):
            refine_forecast.append(el)

        # sort values. At the beginning there will be elements with the maximum correlation coefficient
        # when calculating the correlation, the last element is removed, because it is an expert value
        sort_refine_forecast = sorted(
            refine_forecast.get_result(), reverse=True, key=lambda val: row_corr(val.corrdata[:-1], corr_row))

        arr = [val.corrcoef for val in sort_refine_forecast]
        print(np.min(arr))

        forecast = np.zeros(len(sort_refine_forecast))

        for i, el in enumerate(sort_refine_forecast):
            forecast[i] = ling_method(corr_row, el.corrdata)

        # add row in DataFrame
        df_refine.loc[len(df_refine.index)] = _create_frame_row(
            forecast, indices)

    return df_refine


# neurolinguistic methods


def _create_frame_row(forecast, slices):
    '''
    Create row for frame use median.

    Parametrs:
        forecast: 1D-array with elements range by corr coef.
        slices: 1D-array with indices. Each index use to take slice
            by forecast (from 0 to index) and then compute median.
    '''
    result = []
    for i in slices:
        result.append(np.median(forecast[:i]))
    return result


def axis_reflection(arr):
    '''
    This method augments the data for forecasting.

    Return three arrays:
    - reversed array, in back order
    - inverted array (arr * (-1) + min + max), array are mirrored by x-axis
    - reversed inverted array, inverted array in back order
    '''
    if not isinstance(arr, (np.ndarray)):
        arr = np.array(arr)
    # reversed array
    rev_arr = arr[::-1]
    # inverted array
    inv_arr = arr * (-1) + arr.min() + arr.max()
    # reversed inverted array
    rev_inv_arr = inv_arr[::-1]

    return rev_arr, inv_arr, rev_inv_arr


def find_all_rows(crow, srows, corrcoef=0.95, horizon=1):
    '''
    On each step return `horizon` elements if previous elements
    have correlation coefficient with `crow` great then `corrcoef`.

    Parameters
    ---------

    srows: array of arrays
    in this row we need to find all subrows
        with correlation coefficient of (subrow and `crow`) >= `corrcoef`

    crow: row to calculation corrcoef coef

    corrcoef: correlation coefficient

    horizon - count of forecasts (on one step, on two steps, ...)
    '''
    assert -1 <= corrcoef <= 1, "Incorrect corrcoef!"

    assert not np.isscalar(srows[0]), "srows - array of arrays!"

    if not len(crow) or not len(srows):
        return

    # len subrow without horizon elements
    sub_len = len(crow)
    # len subrow with horizon elements
    full_len = sub_len + horizon

    # return correlation row with horizon elements at the end on each step
    for row in srows:
        for i in range(len(row) - full_len + 1):
            # get full subrow (with horizon elements)
            sub_row = row[i:i+full_len]
            # get only correlation row
            corr_row = row[i:i+sub_len]

            # calculate correlation coef
            coef = row_corr(corr_row, crow)
            # if corr_row is suitable
            if coef >= corrcoef:
                # subrow[-horizon:] - [horizon elements]
                yield CrudeForecast(corr_row, sub_row[-horizon:], sub_row, coef)


def _normalize(row_x, Mx, Dx):
    '''
    Normalize row with Mx and Dx: (x - Mx) / Dx.
    '''
    if isinstance(row_x, (np.ndarray)):
        return (row_x - Mx) / math.sqrt(Dx)
    return (np.array(row_x) - Mx) / math.sqrt(Dx)


def ling_method(prow: Sequence, frow: Sequence, horizon: int = None) -> float:
    '''
    Predict one number and return it.
    WARNING: len(prow) < len(frow)

    Parameters:
    ---------

    prow: array like
    Row to be predicted.

    frow: array like
    Row with which forecast is made.

    horizon: int (standart None)
    How many predicted dots will return.

    Returns: 
    -------
    array[float]
    '''

    if horizon == None:
        horizon = len(frow) - len(prow)

    assert len(frow) > len(prow)
    assert horizon >= 1 and horizon <= len(frow) - len(prow)

    # result
    forecast = np.zeros(horizon)

    row_x = np.array(prow)
    row_y = np.array(frow[:len(prow)])

    Mx, My = row_x.mean(), row_y.mean()
    Dx, Dy = row_x.var(), row_y.var()

    for index in range(horizon):
        y = frow[len(prow) + index]
        # Formula: (x - Mx) / sqrt(Dx) =  (y - My) / sqrt(Dy)
        # Find x
        x = (y - My) * math.sqrt(Dx / Dy) + Mx
        forecast[index] = x

    return forecast


def row_corr(row1: Sequence, row2: Sequence) -> float:
    '''
    Return rows correlation number in interval [-1; 1]
    '''
    return np.corrcoef(row1, row2)[0, 1]


if __name__ == "__main__":
    print('row_corr', row_corr([0, 1, 4, 9], [9, 16, 25, 36]))
    res = ling_method(
        [0, 1, 4, 9], [9, 16, 25, 36, 49, 64], horizon=2)
    print(res)
    print(row_corr([0, 1, 4, 9] + list(res), [9, 16, 25, 36, 49, 64]))
