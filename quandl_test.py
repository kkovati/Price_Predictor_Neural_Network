# https://www.quandl.com/tools/python
# registered to this site
# API key:HNG5Ky6fcw6SsTQZuxNS

# https://docs.quandl.com/docs/python-time-series
# https://github.com/quandl/quandl-python

# https://www.quandl.com/search

# https://www.youtube.com/watch?v=EYnC4ACIt2g&t=373s

import quandl
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = 'HNG5Ky6fcw6SsTQZuxNS'
# Get the data for Coca-cola
dataframe = quandl.get("WIKI/FB", start_date="2016-01-01", end_date="2018-01-01", api_key="HNG5Ky6fcw6SsTQZuxNS")

print(dataframe.head())

# Plot the prices
dataframe.plot()
plt.show()
