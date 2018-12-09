
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
