import os
import urllib2

with open('tickers.txt', 'r') as fh:
    lines = fh.readlines()

url = 'http://financials.morningstar.com/ajax/exportKR2CSV.html?t=TICKERHERE&culture=en-CA&region=USA&order=desc'

folder_path = '/home/user1/Desktop/Morningstar_Data'
file_base = 'TICKERHERE Key Ratios.csv'

tickers = []
for line in lines:
    tickers.append(line.strip())

for indx in range(len(tickers)):
    ticker = tickers[indx]
    print("%s of %s: processing %s" % (indx, len(tickers), ticker))
    temp_url = url.replace('TICKERHERE', ticker)
    req_obj = urllib2.urlopen(temp_url)
    content = req_obj.read()

    if len(content) < 1000:
        print("Skipping %s: content length too short: %s" % (ticker, len(content)))
        continue

    file_name = file_base.replace('TICKERHERE', ticker)
    file_name = os.path.join(folder_path, file_name)
    with open(file_name, 'w') as fh:
        fh.write(content)



