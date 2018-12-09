import os
import sys
import pdb
import numpy as np


PRICE_HISTORY = "Raw_Data_Technical"
FORMATTED_DATA_DIR = "Formatted_Data"
TRAIN_PRICES_TECHNICAL_FILE = "technical_train_prices.npy"
TRAIN_VOLUMES_TECHNICAL_FILE = "technical_train_volumes.npy"
TRAIN_LABELS_TECHNICAL_FILE = "technical_train_labels.npy"
TRAIN_DATES_TECHNICAL_FILE = "technical_train_dates.npy"
TRAIN_TICKERS_TECHNICAL_FILE = "technical_train_tickers.npy"


class MissingYearException(Exception):
    pass


def is_numeric(val):
    try:
        float(val.strip())
        return True
    except ValueError:
        return False


def get_year_string(year):
    """
    arg: year: string with year month and maybe other chars
    return: four-digit year as a string
    """
    yr = year.strip()[0:4]
    if not yr.isdigit():
        raise Exception("Year %s is not a digit!" % yr)
    assert(int(yr) > 1950 and int(yr) < 2020)
    return yr


def get_years_dict(ticker, years):
    """
    Get annual price changes for the given ticker
    arg: ticker: string ticker symbol
    arg: years: list of years as strings
    return: list or annual price changes for the given years
      in the same order as the years list
      calculated as (late Dec price - early Jan price) / (early Jan price)
    """
    file_name = ticker.strip().lower() + '.us.txt'
    file_path = os.path.join(os.getcwd(), PRICE_HISTORY, file_name)
    if not os.path.isfile(file_path):
        raise Exception("File %s does not exist!" % file_path)
    with open(file_path, 'r') as fh:
        lines = fh.readlines()
    years_dict = {}  # key = year string, value = price change that year
    # first line is column names
    del(lines[0])
    for line in lines:
        parts = line.split(',')
        assert(7 == len(parts))
        yr = get_year_string(parts[0])
        price = parts[1]
        if not is_numeric(price):
            raise Exception("Price %s for ticker %s is not a digit!" % (price, ticker))
        price = float(price)
        assert(price > 0)
        assert(price < 1000000)
        if yr in years_dict:
            years_dict[yr].append(price)  # chronological order
        else:
            years_dict[yr] = [price]

    return years_dict


def get_annual_price_changes(ticker, years):
    """
    Get annual price changes for the given ticker
    arg: ticker: string ticker symbol
    arg: years: list of years as strings
    return: list or annual price changes for the given years
      in the same order as the years list
      calculated as (late Dec price - early Jan price) / (early Jan price)
    """
    years_dict = get_years_dict(ticker, years)

    # remove years with too few prices
    for key in years_dict.keys():
        if len(years_dict[key]) < 100:
            del(years_dict[key])

    # convert price lists to price change for each year
    for key in years_dict.keys():
        prices = years_dict[key]
        prc_change = (prices[-1] - prices[0]) / prices[0] * 100
        years_dict[key] = round(prc_change, 3)

    price_changes = []
    for yr in years:
        if yr not in years_dict:
            msg = "Year %s is not in years_dict for ticker %s" % (yr, ticker)
            raise MissingYearException(msg)
        price_changes.append(years_dict[yr])
    assert(len(price_changes) == len(years))
    return price_changes

def get_train_and_test_data(file_path, days_window, num_windows):
    """
    Read in the given file_path and return a tuple with five numpy arrays:
      train_prices, train_volumes, train_labels, tickers, and train_dates
      for the given number of time windows
      of days_windows number of days each
    """
    assert(os.path.exists(file_path))
    with open(file_path, 'r') as fh:
        lines = fh.readlines()
    assert('Date' in lines[0])
    del(lines[0])  # remove headers line
   
    PRICE_CHANGE_DAYS = 261  # work days, so ~1 year total
 
    # num days for one set of training data
    iter_learn_days = days_window * num_windows  # period used for prediction
    # both learning and price change periods
    iter_total_days = iter_learn_days + PRICE_CHANGE_DAYS

    #print("Len lines = %s" % len(lines))
    #print("iter_learn_days = %s" % iter_learn_days)
    #print("iter_total_days = %s" % iter_total_days)

    if len(lines) < iter_total_days:
        raise Exception("Skipping %s due to too few days of data." % file_path)

    train_prices = []  # list of 1+ lists of price averages
    train_volumes = []  # list of 1+ lists of volume averages
    train_labels = []  # list of price change labels
    train_tickers = []  # ticker corresponding to each label
    train_dates = []  # list of date strings for each days_window start

    # how many iterations fit into lines
    num_iter = int( len(lines) / iter_total_days )
    iter_start = 0
    for _ in range(num_iter):
        #print("iter start: %s" % iter_start)
        win_price_avgs = []  # current iteration's price averages for each window
        win_vol_avgs = []  # current iteration's volume averages for each window
        win_start_dates = []  # date strings for first date of window

        # iterate over days_windows
        iter_stop = iter_start + iter_learn_days
        for win_start in range(iter_start, iter_stop, days_window):

            # iterate over days in window
            vol_win_sum = 0
            price_win_sum = 0
            win_start_date = None
            win_stop = win_start + days_window
            #print("      win start - stop: %s - %s" % (win_start, win_stop))
            for indx in range(win_start, win_stop, 1):
                #print("        indx = %s" % indx)
                cur_date, cur_price, cur_vol = get_date_price_vol(lines[indx])
                cur_price = round(cur_price, 2)
                if not win_start_date:
                    win_start_date = cur_date
                price_win_sum += cur_price
                vol_win_sum += cur_vol
            win_price_avgs.append(price_win_sum / days_window)
            win_vol_avgs.append(vol_win_sum / days_window)
            win_start_dates.append(win_start_date)

        # calculate following 12-months' price change
        nxt_yr_strt_indx = iter_start + iter_learn_days
        nxt_yr_stop_indx = nxt_yr_strt_indx + PRICE_CHANGE_DAYS - 1
        nxt_yr_strt_dt, nxt_yr_strt_price, _ = get_date_price_vol(lines[nxt_yr_strt_indx])
        nxt_yr_stop_dt, nxt_yr_stop_price, _ = get_date_price_vol(lines[nxt_yr_stop_indx])
        
        # append new training data
        train_volumes.append(win_vol_avgs)
        train_prices.append(win_price_avgs)
        train_dates.append(win_start_dates)
        nxt_yr_label = get_label(nxt_yr_strt_price, nxt_yr_stop_price)
        #print("Next year price change from indx %s to %s" % (nxt_yr_strt_indx, nxt_yr_stop_indx))
        #print("Label for price change from %s to %s is %s" %\
        #      (nxt_yr_strt_dt, nxt_yr_stop_dt, nxt_yr_label))
        train_labels.append(nxt_yr_label)
        train_tickers.append(file_path.split('/')[-1])

        iter_start += iter_total_days

    # toggle between vol and price training data here
    return (train_prices, train_volumes, train_labels, train_tickers, train_dates)


def get_label(price_start, price_end):
    """ Return a label for the price change """
    if price_end > price_start:
        return 1
    return 0


def get_date_price_vol(line):
    """ Return the date, open price, and volume for this line """
    parts = line.split(',')
    if len(parts) != 7:
        raise Exception("Line parts != 7!  Line: '%s'" % line)
    date_str = parts[0]
    price_str = parts[1]
    vol_str = parts[5]
    if not is_numeric(price_str) or not is_numeric(vol_str):
        raise Exception("Price %s or vol %s not numeric!" % (price_str, vol_str))
    return (date_str, float(price_str), float(vol_str))


def store_full_data_set(formatted_data_dir, days_window, num_windows, num_examples):
    """
    Create a data set for the given number of examples and time variables
    arg: formatted_data_dir: directory to write numpy arrays of
      training and test data and labels
    arg: days_window: number of days to average into a single data point
      e.g. 7 would mean weekly price averages, 30 would be ~monthly
    arg: num_windows: number of windows to use for predicting next 12 months'
      price change, e.g. for days_window = 7, num_windows of 52 would mean
      use 1 year of 7-day averages to predict price change over next 12 months
    arg: num_examples: how many training examples to return
    return: 5-item tuple: (train_prices, train_volumes, train_labels, tickers, train_dates)
      train_prices: 2-D list, each row = days_window averages for num_windows
      train_volumes: 2-D list, each row = days_window averages for num_windows
      train_labels: 1-D list, Jan-Dec price change identifier for the year
      tickers: 1-D list, ticker symbol for each label
      train_dates: 2-D list, start date for each price / vol window
        following the last of the num_windows        
    """
    price_hist_dir = os.path.join(os.getcwd(), PRICE_HISTORY)
    assert(os.path.exists(price_hist_dir))
    train_prices = []
    train_volumes = []
    train_labels = []
    train_tickers = []
    train_dates = []

    count = 0
    for file_name in os.listdir(price_hist_dir):
        file_path = os.path.join(price_hist_dir, file_name)
        if count >= num_examples:
            break
        #print("Processing file %s" % file_name)
        print("File %s -- %s examples out of %s" % (file_name, count, num_examples))
        try:
            prices, volumes, labels, tickers, dates = \
              get_train_and_test_data(file_path, days_window, num_windows)
        except Exception as exc:
            #print("Skipping %s due to error %s" % (file_name, exc))
            continue
        train_prices += prices
        train_volumes += volumes
        train_labels += labels
        train_tickers += tickers
        train_dates += dates
        count += 1

    train_prices = np.array(train_prices)
    train_volumes = np.array(train_volumes)
    train_labels = np.array(train_labels)
    train_tickers = np.array(train_tickers)
    train_dates = np.array(train_dates)

    train_prices_path = os.path.join(formatted_data_dir, TRAIN_PRICES_TECHNICAL_FILE)
    train_volumes_path = os.path.join(formatted_data_dir, TRAIN_VOLUMES_TECHNICAL_FILE)
    train_labels_path = os.path.join(formatted_data_dir, TRAIN_LABELS_TECHNICAL_FILE)
    train_tickers_path = os.path.join(formatted_data_dir, TRAIN_TICKERS_TECHNICAL_FILE)
    train_dates_path = os.path.join(formatted_data_dir, TRAIN_DATES_TECHNICAL_FILE)

    np.save(train_prices_path, train_prices)
    np.save(train_volumes_path, train_volumes)
    np.save(train_labels_path, train_labels)
    np.save(train_tickers_path, train_tickers)
    np.save(train_dates_path, train_dates)


def load_full_data_set():
    """ return tuple (train_prices, train_volumes, train_labels, tickers, train_dates) """
    formatted_data_dir = os.path.join(os.getcwd(), FORMATTED_DATA_DIR)
    assert(os.path.exists(formatted_data_dir))
    prices_path = os.path.join(formatted_data_dir, TRAIN_PRICES_TECHNICAL_FILE)
    volumes_path = os.path.join(formatted_data_dir, TRAIN_VOLUMES_TECHNICAL_FILE)
    labels_path = os.path.join(formatted_data_dir, TRAIN_LABELS_TECHNICAL_FILE)
    tickers_path = os.path.join(formatted_data_dir, TRAIN_TICKERS_TECHNICAL_FILE)
    dates_path = os.path.join(formatted_data_dir, TRAIN_DATES_TECHNICAL_FILE)
    assert(os.path.exists(prices_path))
    assert(os.path.exists(volumes_path))
    assert(os.path.exists(labels_path))
    assert(os.path.exists(tickers_path))
    assert(os.path.exists(dates_path))
    prices = np.load(prices_path)
    volumes = np.load(volumes_path)
    labels = np.load(labels_path)
    tickers = np.load(tickers_path)
    dates = np.load(dates_path)
    return(prices, volumes, labels, tickers, dates)


if __name__ == '__main__':
    formatted_data_dir = os.path.join(os.getcwd(), FORMATTED_DATA_DIR)
    assert(os.path.exists(formatted_data_dir))
    days_window = 5  # one week (2 weekend days are not in the csv files)
    num_windows = 52 * 7  # years is 52 * <num> 
    num_examples = 5000 
    store_full_data_set(formatted_data_dir, days_window, num_windows, num_examples)

