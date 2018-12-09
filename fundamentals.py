# Methods to get and normalize data from Morningstar fundamentals csv files

import os
import re
import numpy as np
import pdb
import logging
import traceback

from technical import get_annual_price_changes, get_year_string,\
  is_numeric, MissingYearException, FORMATTED_DATA_DIR


logging.basicConfig(filename='fundamentals.log', level=logging.DEBUG)


# Source directories
MORNINGSTAR_FUNDAMENTALS = "Raw_Data_Fundamentals"

# File names for formatted data
TRAIN_DATA_FUNDAMENTALS_FILE = "fundamentals_train_data.npy"
TRAIN_LABELS_FUNDAMENTALS_FILE = "fundamentals_train_labels.npy"
YEAR_RANGES_FUNDAMENTALS_FILE = "fundamentals_year_ranges.npy"

METRIC_NAMES = ['Gross Margin %', 'Operating Margin %', 'Payout Ratio %', 'Shares Mil', 'Book Value Per Share', 'Earnings Per Share', 'Dividends', 'Operating Cash Flow', 'Cap Spending', 'Free Cash Flow Per Share', 'Working Capital', 'Revenue', 'Operating Margin', 'Net Int Inc & Other', 'EBT Margin', 'Tax Rate %', 'Net Margin %', 'Asset Turnover (Average)', 'Return on Assets %', 'Financial Leverage (Average)', 'Return on Equity %', 'Return on Invested Capital %', 'Interest Coverage', 'Operating Cash Flow Growth % YOY', 'Free Cash Flow Growth % YOY', 'Cap Ex as a % of Sales', 'Free Cash Flow/Sales %', 'Free Cash Flow/Net Income', 'Cash & Short-Term Investments', 'Inventory', 'Accounts Receivable', 'Total Current Assets', 'Net PP&E', 'Intangibles', 'Accounts Payable', 'Short-Term Debt', 'Total Current Liabilities', 'Long-Term Debt', 'Total Liabilities', "Total Stockholders' Equity", 'Current Ratio', 'Quick Ratio', 'Financial Leverage', 'Debt/Equity', 'Days Sales Outstanding', 'Days Inventory', 'Payables Period', 'Cash Conversion Cycle', 'Receivables Turnover', 'Inventory Turnover', 'Fixed Assets Turnover', 'Asset Turnover']


def unquote_and_uncomma(line):
    """
    Remove numbers with commas inside double-quotes
    e.g. '"-12,345", stuff' becomes '-12345, stuff'
    """
    SEARCH_STR = r'("[^"]*")'
    match_obj = re.search(SEARCH_STR, line)
    while match_obj:
        old_txt = match_obj.group(1)
        new_txt = old_txt.replace('"', '')
        new_txt = new_txt.replace(',', '')
        line = line.replace(old_txt, new_txt)
        match_obj = re.search(SEARCH_STR, line)
    return line


def remove_quotes_and_commas(lines):
    for indx in range(len(lines)):
        lines[indx] = unquote_and_uncomma(lines[indx])


def delete_blank_lines(lines, file_name):
    indx = 0
    while indx < len(lines):
        if not lines[indx].strip():
            del(lines[indx])
            continue
        indx += 1


def delete_bad_lines(lines, file_name):
    indx = 0
    while indx < len(lines):
        line_parts = lines[indx]
        assert isinstance(line_parts, list), "File %s, line %s not list" % (file_name, indx)
        to_delete = False
        if 12 != len(line_parts):
            msg = "Line deletion: not 12 comma-separated items:\n"
            msg += "%s\n" % line_parts
            msg += "Only see %s items.\n" % len(line_parts)
            logging.warning(msg)
            to_delete = True    
        else:
            for val in line_parts[1:]:
                val = val.strip()
                if val and not is_numeric(val):
                    msg = "Line deletion: values not numeric: %s" % line_parts
                    msg += ". Item '%s' is not a number." % val
                    logging.warning(msg)
                    to_delete = True
                    break
        if to_delete:
            logging.warning("File: %s, deleting line: %s" % (file_name, lines[indx]))
            del(lines[indx])
            continue
        indx += 1


def get_years(lines, file_name):
    yrs_indx = None
    year_search_str = 'TTM.*2016.*2015.*2014.*2013.*2012.*2011.*2010.*2009'
    for line_indx in range(len(lines)):
        if re.search(year_search_str, lines[line_indx], flags=re.IGNORECASE):
            yrs_indx = line_indx
    if yrs_indx is None:
        raise Exception("File %s has no years line!" % file_name)
    years = lines[yrs_indx].split(',')  # [None, TTM, 2017, ..., 2008]
    if 'TTM' not in years[1]:
        raise Exception("Expecting TTM in first year: %s" % years)
    del(years[0])  # delete first None
    del(years[0])  # delete TTM because price history only goes to 2017
    years = map(get_year_string, years)
    # [2017, 2016, ..., 2008]
    assert 10 == len(years), "File %s years %s not len 10" % (file_name, years)
    return years


def get_metrics(file_name):
    """
    arg: file_name: name of csv file to open
    return: tuple with two lists (years, metrics)
      years: 11 items [TTM, 2017, ..., 2008]
      price_averages: 11 items with average stock prices for each year
      metrics: dictionary where key = fundamental metric name (e.g. 'P/E')
        and value = list of metric values for years 2017 to 2008
    """
    file_path = os.path.join(os.getcwd(), MORNINGSTAR_FUNDAMENTALS, file_name)
    lines = None
    with open(file_path, 'r') as fh: 
        lines = fh.readlines()
    delete_blank_lines(lines, file_name)
    remove_quotes_and_commas(lines)
    years = get_years(lines, file_name)

    for indx in range(len(lines)):  # convert each line string into a list of strings
        lines[indx] = lines[indx].split(',')
    delete_bad_lines(lines, file_name)  # remove malformatted lines

    # get the fundamentals for each year
    metrics = {}  # key = one of METRIC_NAMES, value = list of vals for years 2017-2008
    for line_parts in lines:
        # line format: ['P/E', TTM-value, 2017-value, ..., 2008-value]
        for mtrc_nm in METRIC_NAMES:
            if mtrc_nm.lower() in line_parts[0].lower() and mtrc_nm not in metrics:
                metrics[mtrc_nm] = line_parts[2:]  # 10 items: values for 2017 to 2008

    for mtrc_nm in METRIC_NAMES:
        if mtrc_nm not in metrics:
            raise Exception("%s not found in %s" % (mtrc_nm, file_name))
        else:
            # convert all values to numbers
            for jjx in range(len(metrics[mtrc_nm])):
                val = metrics[mtrc_nm][jjx].strip()
                if not val:
                    metrics[mtrc_nm][jjx] = None
                else:
                    metrics[mtrc_nm][jjx] = round(float(val), 3)
    return (years, metrics)


def get_metric_slopes(metrics, start, end):
    """
    Calculate slopes of metrics between given indexes
    Start index will be > end index becuase latest year is on left
    If the start:end window is missing data, that metric slope will be None
    arg: metrics: dictionary where key = fundamental metric name (e.g. 'P/E')
         and value = list of 10 metric values for years 2017 to 2008
    arg: start: start index (inclusive)
    arg: end: end index (inclusive)
    return: dictionary with key = metric name, value = slope of three year's
      e.g. {'P/E Ratio': -0.09, 'Market Cap': 0.24, ...}
    """
    metric_slopes = {}
    # years increase right-to-left
    assert start > end, "Start %s should be > end %s" % (start, end)
    # expect 3 points
    assert (start-end) == 2, "Start %s to end %s should be 2" % (start, end)
    for key in metrics.keys():
        if None in metrics[key][end:(start+1)]:
            metric_slopes[key] = None
        else:
            # slopes of start to end and start to mid
            slope1 = (metrics[key][end] - metrics[key][start]) / 2
            slope2 = metrics[key][end+1] - metrics[key][start]
            # average the above two slopes
            metric_slopes[key] = (slope1 + slope2) / 2

    return metric_slopes
    

def years_metrics_checker(years, metrics):
    """ Integrity check """
    # make sure years are decreasing
    for indx in range(len(years)-1):
        assert years[indx] > years[indx+1], "Years not decreasing %s !> %s" %\
          (years[indx], years[indx+1])
    # make sure same number of years and metric values
    num_yrs = len(years)
    for ky in metrics.keys():
        assert len(metrics[ky]) == num_yrs, "len(metrics[%s] %s != num_yrs %s" %\
          (ky, len(metrics[ky]), num_yrs)
    

def cut_off_2018_and_newer(years, metrics):
    """ Price history only has data up to 2017 """
    while int(years[0]) >= 2018:
        del(years[0])
        for ky in metrics.keys():
            del(metrics[ky][0])


def get_train_and_test_data(file_name):
    """
    Process a morningstar fundamentals csv file
    arg: file_name: name of csv file
    return: 3-item tuple: (train_data, train_labels, year_ranges)
      train_data: 2-D list, each row = slopes of fundamental metrics
      train_labels: 1-D list, price changes during 4th year
      year_ranges: 1-D list of year strings, the first three years are over
        which the fundamentals slope is measured, the fourth year is during
        which the Jan-Dec price change is measured
    """
    # years ~ [2017, 2016, ..., 2008] -- 10 entries
    # annual_price_changes ~ [1, 2, ..., 11]  -- 10 entries
    # metrics ~ {'P/E': [1, 2, ..., 10], 'Market_Cap': [1, ..., 10], ...}
    #   for years 2017 to 2008 -- 11 entries total
    (years, metrics) = get_metrics(file_name)
    years_metrics_checker(years, metrics)
    cut_off_2018_and_newer(years, metrics)

    ticker = file_name.split()[0]
    try:
        annual_price_changes = get_annual_price_changes(ticker, years)
    except MissingYearException as exc:
        logging.warning("File %s does is missing a year: %s" % (file_name, exc))
        #print("Warning: ignoring file %s" % file_name)
        #print("    It is missing a year: %s" % exc)
        return ([], [], [])

    len_yrs = len(years)
    for k in metrics.keys():
        assert len(metrics[k]) == len_yrs, "File %s, len(metrics[%s]) %s != len_yrs %s" %\
          (file_name, k, len(metrics[k]), len_yrs)

    # use a 4-year sliding window to create training / test data
    # left-most index is the latest, right-most is the oldest
    train_data = []  # fundamentals' slopes over 3-year windows, 2-D array
    train_labels = []  # price changes for 4th year, 1-D array
    year_ranges = []  # strings of years (for debugging), 1-D array
    start_indx = len(years) - 1

    # iterate over the years: 2008 to 2016, sliding window of 3 years
    for indx in range(start_indx, 2, -1):
        train_data_part = []
        train_labels_part = []
        year_ranges_part = []
        try:
            years_str = "%s-%s" % (years[indx], years[indx-3])
            year_ranges_part.append(years_str)

            # dict, key: metric name, value: slope of three years
            # e.g. {'P/E Ratio': -0.09, 'Market Cap': 0.24, ...}
            # Remember that years increase right-to-left
            metric_slopes = get_metric_slopes(metrics, indx, indx-2)

            # iterate over the fundamentals metric names
            # for given year window, e.g. 2008-2010
            # [P/E_slope, debt_slope, ...]
            slopes_row = [metric_slopes[key] for key in METRIC_NAMES]
            train_data_part.append(slopes_row)

            # add the label: stock price change for the 4th year
            fourth_yr_price_change = annual_price_changes[indx-3]  # single float num
            train_labels_part.append(fourth_yr_price_change)
            # Note: slope over first 3 years, 4th year is the price change
        except Exception as exc:
            print("Skipping years %s due to error %s" % (years_str, exc))
            continue

        train_data += train_data_part
        train_labels += train_labels_part
        year_ranges += year_ranges_part

    assert len(train_data) == len(year_ranges), "Len train_data != year_ranges"
    assert len(train_labels) == len(year_ranges), "Len train_lbls != year_ranges"
    assert len(train_data[0]) == len(METRIC_NAMES), "Len train_data != metric names"
    return (train_data, train_labels, year_ranges)


def store_full_data_set(formatted_data_dir):
    """
    Create a full data set based on all available Morningstar
      fundamentals data as csv files and store the 
    arg: formatted_data_dir: directory to write numpy arrays of
      training and test data and labels
    return: 3-item tuple: (train_data, train_labels, year_ranges)
      train_data: 2-D list, each row = slopes of fundamental metrics
      train_labels: 1-D list, price changes during 4th year
      year_ranges: 1-D list of year strings, the first three years are over
        which the fundamentals slope is measured, the fourth year is during
        which the Jan-Dec price change is measured
    """
    fundamentals_dir = os.path.join(os.getcwd(), MORNINGSTAR_FUNDAMENTALS)
    assert os.path.exists(fundamentals_dir), "Path no exist %s" % fundamentals_dir
    train_data = []
    train_labels = []
    year_ranges = []
    all_files = os.listdir(fundamentals_dir)
    num_files = len(all_files)
    for indx in range(num_files):
        file_name = all_files[indx]
        #print("Processing file %s" % file_name)
        print("File %s of %s" % (indx, num_files))
        try:
            train_data_part, train_labels_part, year_ranges_part = \
              get_train_and_test_data(file_name)
        except Exception as exc:
            stack_trace = traceback.format_stack()
            print("Skipping %s due to error %s" % (file_name, exc))
            continue

        train_data += train_data_part
        train_labels += train_labels_part
        year_ranges += year_ranges_part

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    year_ranges = np.array(year_ranges)

    train_data_path = os.path.join(formatted_data_dir, TRAIN_DATA_FUNDAMENTALS_FILE)
    train_labels_path = os.path.join(formatted_data_dir, TRAIN_LABELS_FUNDAMENTALS_FILE)
    year_ranges_path = os.path.join(formatted_data_dir, YEAR_RANGES_FUNDAMENTALS_FILE)

    np.save(train_data_path, train_data)
    np.save(train_labels_path, train_labels)
    np.save(year_ranges_path, year_ranges)


def load_full_data_set():
    """ return tuple (train_data, train_labels, year_ranges) """
    formatted_data_dir = os.path.join(os.getcwd(), FORMATTED_DATA_DIR)
    assert os.path.exists(formatted_data_dir), "Path not exist %s" % formatted_data_dir
    train_data_path = os.path.join(formatted_data_dir, TRAIN_DATA_FUNDAMENTALS_FILE)
    train_labels_path = os.path.join(formatted_data_dir, TRAIN_LABELS_FUNDAMENTALS_FILE)
    year_ranges_path = os.path.join(formatted_data_dir, YEAR_RANGES_FUNDAMENTALS_FILE)
    assert os.path.exists(train_data_path), "bad data path"
    assert os.path.exists(train_labels_path), "bad labels path"
    assert os.path.exists(year_ranges_path), "bad years path"
    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)
    year_ranges = np.load(year_ranges_path)
    return (train_data, train_labels, year_ranges)


if __name__ == '__main__':
    formatted_data_path = os.path.join(os.getcwd(), FORMATTED_DATA_DIR)
    assert os.path.exists(formatted_data_path), "bad formatted data path"
    store_full_data_set(formatted_data_path)

