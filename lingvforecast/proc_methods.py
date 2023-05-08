'''
Modules provide methods for processing forecast.

Contains class HForecast to be inherit handler classes.
'''
import statistics
import numpy as np
from abc import ABC, abstractmethod


class HForecast(ABC):

    @abstractmethod
    def handle(self, arr):
        '''
        Takes array of Forecast classes.
        Handle result forecast.
        '''
        pass

class Median(HForecast):

    def handle(self, arr):

        if len(arr[0].info.fordata) == 1:
            # just one step
            for ft in arr:
                # get all information
                forecast = ft.forecast
                info = ft.info

                
        else:
            # many steps
            # ....
            pass


class MedianMean10(HForecast):

    def handle(self, arr):
        pass

def max_corr(arr):
    '''

    '''
    pass


def median(arr):
    '''
    Return median element of array
    '''
    arr = np.sort(np.array(arr))
    return statistics.median(arr)


def median_mean_10(arr):
    '''
    Takes the 10 elements closest to the median and calculate mean.
    arr - array with forecasts
    '''

    arr = np.sort(np.array(arr))

    if len(arr) == 0:
        return 0

    median_len = 0

    while len(arr) - 2*median_len > 10:
        if len(arr) == 1:
            return arr[0]
        if len(arr) == 2:
            return (arr[0] + arr[1]) / 2
        median_len += 1

    return np.mean(arr)


def median_mean(arr, coef):
    '''
    First - calculating median of array while array not lose `len(array) * coef` 
    element. 
    Second - calculate average from remaining element.

    0 <= coef <= 1

    Return average
    '''

    assert 0 <= coef <= 1

    if len(arr) == 0:
        return 0

    remain_count = int((1 - coef) * len(arr))

    while len(arr) > remain_count:
        if len(arr) == 1:
            return arr[0]
        if len(arr) == 2:
            return (arr[0] + arr[1]) / 2
        del arr[0]
        del arr[-1]

    return np.mean(arr)


def p_velocity(x: float, y: float):
    '''
    Return velocity of value grows in *percents*.
    Velocity compute by formula:
    (y / x - 1) * 100.
    '''
    return (y / x - 1) * 100


def s_velocity(x: float, y: float):
    '''
    Return velocity of value grows.
    Velocity compute by formula:
    (x - y) / ( (x + y) / 2).
    '''
    return (x - y) / ((x + y) / 2)


if __name__ == "__main__":
    print(median(list(range(1, 21))))

    print(median_mean(list(range(1, 21)), 0.7))
