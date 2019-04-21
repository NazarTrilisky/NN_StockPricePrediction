  Abstract
  This project searched for patterns in technical data (prices, volumes) and fundamentals data (54 metrics) that will predict future stock prices.  Neural networks (NN) were used as models.
  The technical data input was a list of weekly price and volume averages over a 3 to 8-year period.  The fundamentals data input was a list of fundamental metric slopes over 3-year windows.  The output was a predicted price change over the next 12 months.
  The fundamentals approach showed a small (~1%) improvement in prediction compared to random guessing.  The technical approach did not show any improvement over random guessing.
  From a presonal investment standpoint, I am sticking to old school value investing.  The NN model may work for large investment institutions that have high trading volumes and the small edge will show up in the average.  My number of trades is too small to justify using the NN as a key decision factor.
See "StockPrediction_Technical_vs_Fundamentals.odt" for more details.

Getting Fundamentals Data
1. Populate "tickers.txt" with ticker symbols: one on each line
2. Run "python pull_morningstar_data.py" for the csv files to be placed into
   '<your_working_dir>/Morningstar_Data'

Running Fundamentals Data
1. Run "python fundamentals.py" to generate train and test data,
   which will be placed into the "Formatted_Data" folder.
2. Run "python nn_regression.py" to create and test a model.

Running Technical Data
1. Run "python technical.py" to generate train and test data,
   which will be placed into the "Formatted_Data" folder.
2. Run "python nn_classification.py" to create and test a model.
   Note: developed using Python 2.7.15.
