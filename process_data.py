import sqlite3
import json
import pandas as pd
import numpy as np
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import statistics as stat
from math import sqrt

def plot_and_RMS(predictions, target, n, title):
	idx = 0
	RMSs = []
	for i in predictions:
		plt.figure(figsize=(20,6))
		plt.plot(i[:500])
		plt.plot(target[:500])
		RMS = sqrt(mean_squared_error(target,i))
		RMSs.append(RMS)
		mean = stat.mean(target)
		stddev = stat.stdev(target)
		plt.title('%s n = %d, RMS = %f, mean = %f, stdev = %f' % (title, n[idx],RMS, mean, stddev))
		print 'title: %s, mean: %s' % (title, mean)
		print 'title: %s, stdev: %s' % (title, stddev)
		print 'title: %s, RMS: %s' % (title, RMS)


		plt.show()
		idx += 1

	return

def pandafy_data(bill_from_db):
	'''
		input:
			bill_from_db - an array from the database

		output:
			to_forecast - numpy array with the y values from bill
			kWseries - Series object with dates as index
	'''

	dataY = [] # kW interval data
	dates = [] # dates for plotting 

	for i in bill_from_db:
		dataY.append(float(i[1]))
		dates.append(time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(float(i[0]))))

	# make dates into pandas-native format for plotting
	dateTimes = pd.to_datetime(dates)
	kW = pd.Series.to_frame(pd.Series(dataY))

	kWseries = pd.Series(dataY, index = dateTimes)
	to_forecast = kWseries.values
	dates = kWseries.index

	return to_forecast

def organize_data(to_forecast, window, horizon):
	"""
	input:
		to_forecast - univariate time series as numpy array
		window - number of items to use in forecast window
		horizon - horizon of the forecast
	output:
		X - a matrix where each row contains a forecast window
		y, the target values for each row of X
	"""
	hop_length = 1 # move window 1 entry each time
	num_frames = 1 + (len(to_forecast) - window) / hop_length
	row_stride = to_forecast.itemsize * hop_length
	col_stride = to_forecast.itemsize

	X = stride_tricks.as_strided(to_forecast,
								shape = (num_frames, window),
								strides = (row_stride, col_stride))


	y = np.zeros(len(X)-horizon)
	for i in range(len(X)-horizon):
			y[i] = X[i+horizon][-1]

	return X[:-horizon], y

def make_predictions(X,y,n):
	'''
		input:
			m - how many datapoints to train on
			X - strided array of hourly use, use as input to model
			y - hourly use at t + 1, target of model
			n - # neighbors to use

		output:
			models - array of models, same size as m, trained on m datapoints
			predictions - predicted use

	'''

	# make a buncha models
	models = [neighbors.KNeighborsRegressor(n[i], weights = 'distance') for i in range(len(n))]
	idx = 0
	predictions = []
	# train len(n) models with m[i] number of points
	for i in models:
		i.fit(X, y)
		predictions.append(i.predict(X))
		idx += 1
	# test accuracy of each one and plot results
	return models, predictions

def get_training_bills():
	# open database connection
	conn = sqlite3.connect('db/development.sqlite3')
	print "Database Opened Successfully"
	print "..."

	# get first 3 from bills table
	cursor = conn.execute("SELECT * from bills ORDER BY user_id ASC LIMIT 3")

	data = [] # will hold id and time series data
	for row in cursor:
		data += [row[0], json.loads(row[1])]

	# close connection because manners
	conn.close()

	# get most recent bill from db
	bills = []
	bills.append(data[-1])
	bills.append(data[-3])
	bills.append(data[-5])
	return bills

def update_db():
	con = sqlite3.connect('db/development.sqlite3')
	print "Database reopened"
	cur = con.cursor()
	# get most recent bill
	cur.execute('SELECT * FROM bills ORDER BY user_id DESC LIMIT 1')
	row = cur.fetchone()
	data = [row[0], json.loads(row[1])]
	forecast = pandafy_data(data[1])
	X,y = organize_data(forecast, 3, 1)
	models, prediction = make_predictions(X, y, [20])
#	nowlist = np.ndarray.tolist(prediction)
	fit_string = json.dumps(prediction[0].tolist())
	print(cur.execute('UPDATE bills SET data_fit = ? WHERE user_id = ?', (fit_string, data[0])))
	con.commit()
	con.close()
	return



bills = get_training_bills()

# make 3 forecastable datasets
forecast1 = pandafy_data(bills[0])
forecast2 = pandafy_data(bills[1])
forecast3 = pandafy_data(bills[2])


window = 3 # sets number of hours to look backward
horizon = 1 # sets number of hours to look forward

# organize datasets into strided arrays X and targets y
X1,y1 = organize_data(forecast1, window, horizon)
X2,y2 = organize_data(forecast2, window, horizon)
X3,y3 = organize_data(forecast3, window, horizon)

n = [2,10,20]
models, predictions = make_predictions(X1, y1, n)

out_of_sample1 = []
out_of_sample2 = []
for i in models:
	out_of_sample1.append(i.predict(X2))
	out_of_sample2.append(i.predict(X3))



plot_and_RMS(predictions, y1, n, "Test Data")
plot_and_RMS(out_of_sample1, y2, n, "Bill One")
plot_and_RMS(out_of_sample2, y3, n, "Bill Two")


update_db()


# TODO rigorous testing for length of window, n value, training data
# TODO split up data based on moving average
# TODO look at data based on day
# include points from previous day in forecast? ie average or max,min



