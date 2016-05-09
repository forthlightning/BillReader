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
import pprint as pp

def get_training_bills():
	# open database connection
	conn = sqlite3.connect('db/development.sqlite3')
	print "Database Opened Successfully"
	print "..."
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

def pandafy_data(bill):
	'''
		input:
			bill - an array from the database

		output:
			to_forecast - numpy array with the y values from bill
			kWseries - Series object with dates as index
	'''

	dataY = [] # kW interval data
	dates = [] # dates for plotting 

	for i in bill:
		dataY.append(float(i[1]))
		dates.append(time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(float(i[0]))))

	# make dates into pandas-native format for plotting
	dateTimes = pd.to_datetime(dates)
	kW = pd.Series.to_frame(pd.Series(dataY))
	kWseries = pd.Series(dataY, index = dateTimes)

	# calculates exponentially weighted moving average because smooth like butta
	mov_avg = pd.DataFrame.ewm(kWseries.to_frame(), halflife = .1)
	to_forecast = kWseries.values
	dates = kWseries.index


	return to_forecast, kWseries, mov_avg, dates

def organize_data(kWseries, window, horizon):
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
	num_frames = 1 + (len(kWseries) - window) / hop_length
	row_stride = 8 * hop_length
	col_stride = 8

	window = 12
	X = np.zeros([len(kWseries)-window, window+24])
	y = np.zeros([len(kWseries)-window, 1])

	for i in range(len(kWseries)-window):
		x_of_i, y_of_i = big_x_small_y(kWseries, i, window)
	
		X[i] = x_of_i
		y[i] = y_of_i

	X_array = np.asarray(X)
	y_array = np.asarray(y)

	# TODO return y for different horizon
	return X_array, y_array

def fit_data(X,y,n):

	models = [neighbors.KNeighborsRegressor(n[i], weights = 'distance') for i in range(len(n))]
	for i in models:
		i.fit(X,y)

	return models

def make_predictions(X,y,models):

	predictions = []
	for i in models:
		predictions.append(i.predict(X))

	# pick one model
	model = models[-1]

	# return array of predictions, one for each model 
	return predictions

def plot_and_RMS(predictions, target, n, title):
	idx = 0
	RMSs = []
	for i in predictions:
		plt.figure(figsize=(20,6))
		plt.plot(i[-500:])
		plt.plot(target[-500:])
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

def update_db(models, window):
	con = sqlite3.connect('db/development.sqlite3')
	print "Database reopened"
	print '...'
	print "..."
	cur = con.cursor()
	# get most recent bill
	cur.execute('SELECT * FROM bills ORDER BY user_id DESC LIMIT 1')
	row = cur.fetchone()
	data = [row[0], json.loads(row[1])]
	forecast, kWseries, MA0, dates = pandafy_data(data[1])
	# TODO clip datevector
	X,y = organize_data(kWseries, window, 1)
	prediction = make_predictions(X, y, models)
#	nowlist = np.ndarray.tolist(prediction)
	fit_string = json.dumps(prediction[0].tolist())
#	plot_and_RMS(prediction, y, [20], "Test Data")
	cur.execute('UPDATE bills SET data_fit = ? WHERE user_id = ?', (fit_string, data[0]))
	con.commit()
	con.close()
	return

def main(window):

	# pull first 3 bills from database
	bills = get_training_bills()

	# make 3 forecastable datasets
	y_array1, kWseries1, MA1, dates1 = pandafy_data(bills[0])
	y_array2, kWseries2, MA2, dates2 = pandafy_data(bills[1])
	y_array3, kWseries3, MA3, dates3 = pandafy_data(bills[2])

	# concatenate EWMA for 3 seed bills from database
	big_data = pd.concat([MA1.mean(), MA2.mean(), MA3.mean()])
	big_series = pd.concat([kWseries1, kWseries2, kWseries3])
	big_MA = pd.DataFrame.ewm(big_series.to_frame(), halflife = .1)

	X0,y0 = organize_data(big_series, window, 1)
	big_model = fit_data(X0,y0.reshape(-1,1),[20]) # this 20 is number of neighbors
	pred = make_predictions(X0,y0,big_model)

	update_db(big_model, window)

	return big_model[0], y0, X0, big_series

def next_prediction_generator(model, X, steps, series):

	i = 0
	current_window = X[-1]

	while i < steps:
		n_plus_one = model.predict(X[i].reshape(1,-1))
		print n_plus_one
		yield n_plus_one
		i += 1

	# print "current window",current_window
	# while i < steps:
	# 	n_plus_one = model.predict(current_window.reshape(1,-1))
	# 	yield n_plus_one
	# 	trim_window = np.delete(current_window, 0)
	# 	new_window = np.concatenate([trim_window, n_plus_one[0]])
	# 	i += 1
	# 	current_window = new_window

def big_x_small_y(series, index, window):
 	
	bin_vector = np.zeros((1,24))[0]
	hour = series.index[index].hour
	bin_vector[hour] = 1

	prev_couple_hours = series[index:index+window].values

	x_of_i = np.concatenate((prev_couple_hours, bin_vector))
	y_of_i = series.values[index]

	return x_of_i, y_of_i


window = 12 # sets size of window for striding array
big_model, y0, X0, big_series = main(window)

		
# sets number of steps to look backward
steps = 96

gentest = next_prediction_generator(big_model, X0, steps, big_series)

predictions = np.zeros([steps, 1])
idx = 0
for i in gentest:
	predictions[idx] = i
	idx += 1


compare = []
for i in range(steps):
	compare.append(X0[-(steps+i)][-25])

plt.plot(compare, color = 'b', label = "Real Data")
plt.plot(predictions, color = 'r', label = "Predicted Data")
plt.legend()
plt.ylabel('kW')
plt.xlabel('Hourly Interval Data')
plt.title('Forecast Horizon: 1 Day')
plt.show()

# TODO rigorous testing for length of window, n value, training data
# TODO split up data based on moving average
# TODO look at data based on day
# TODO include points from previous day in forecast? ie average or max,min



