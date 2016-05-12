import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import statistics as stat
from math import sqrt
import pprint as pp

def get_training_bills():
	# open database connection
	conn = sqlite3.connect('db/development.sqlite3')

	# get first 3 from bills table
	cursor = conn.execute("SELECT * from bills ORDER BY user_id ASC LIMIT 3")

	data = [] # will hold id and time series data
	for row in cursor:
		data += [row[0], json.loads(row[1])]

	# close connection because manners
	conn.close()

	# get first three bills from db
	bills = []
	bills.append(data[-1])
	bills.append(data[-3])
	bills.append(data[-5])

	return bills

def pandafy_data(bill):

	# build lists of time-series values and date indices
	usage_list = []
	date_list = []
	for i in bill:
		usage_list.append(float(i[1]))
		date_list.append(time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(float(i[0]))))

	# make dates into pandas-native format for plotting
	dateTimes = pd.to_datetime(date_list)
	series = pd.Series(usage_list, index = dateTimes)

	return series

def organize_data(series):
	"""
	input:
		to_forecast - univariate time series as numpy array
	output:
		X - a matrix where each row contains a forecast window
		y, the target values for each row of X
	"""

	prev_loads_list = np.zeros([len(series)-window, window])
	times_list = np.zeros([len(series)-window, 24])
	y = np.zeros([len(series)-window, 1])
	X = np.zeros([len(series)-window, window + 24])
	dates = pd.Series(series.index[window:len(series)])

	for i in range(window,len(series)):

		time_vector = np.zeros((1,24))[0]
		hour = series.index[i-window].hour
		
		time_vector[hour] = 1

		prev_loads_list[i-window] = series[i-window:i:1].values
		X[i-window] = np.concatenate((prev_loads_list[i-window], time_vector))
		y[i-window] = series.values[i-1] # TODO should this actually be -1???

		dates[i-window] = series.index[i-window]

		times_list[i-window] = time_vector

	return y, X, prev_loads_list, times_list, dates

def update_db(model):
	con = sqlite3.connect('db/development.sqlite3')

	# create cursor
	cur = con.cursor()

	# get most recent bill
	cur.execute('SELECT * FROM bills ORDER BY user_id DESC LIMIT 1')
	row = cur.fetchone()

	# data[0] is name, data [1] is bill data from database
	data = [row[0], json.loads(row[1])]

	# standard pandafy, organize, predict chain
	series = pandafy_data(data[1])
	y, X, prev_loads_list, times_list, dates = organize_data(series)


	AR_predictor = do_autoregressive_prediction(model, steps, times_list, prev_loads_list)

	temp = []
	for i in AR_predictor:
		temp.append(i)

	AR_prediction = np.concatenate(temp)
	AR_pred_list = AR_prediction.tolist()

	# save data as string and write to DB
	fit_string = json.dumps(AR_pred_list)
	cur.execute('UPDATE bills SET data_fit = ? WHERE user_id = ?', (fit_string, data[0]))
	con.commit()
	con.close()

	return X, y, AR_prediction, series

def do_autoregressive_prediction(model, steps, time_vector_list, load_history):

	recent_load = load_history[-(start_point+steps)].reshape((1, -1)) # SOMETIMES THIS WORKS BETTER WITH "start_point+steps+1"
	recent_time = time_vector_list[-(start_point+steps+1)].reshape((1, -1))
	most_recent_x = np.concatenate((recent_load, recent_time), axis = 1)
	t_plus_one = model.predict(most_recent_x[0].reshape(1,-1))

	for i in range(steps):

		yield t_plus_one

		clip_array = np.delete(recent_load, (0,1)).reshape(-1,1)
		print load_history[i]
		real_point = load_history[i][-(window-1)]
		new_x = np.concatenate((real_point.reshape(1,1), clip_array, t_plus_one, time_vector_list[i+1].reshape(-1,1)))

		t_plus_one = model.predict(new_x.reshape(1,-1))


def main():

	print "Time Started @ %f" % (time.clock())

	# pull first 3 bills from database
	bills = get_training_bills()

	print "Bills from DB @ %f" % (time.clock())

	# make 3 forecastable datasets
	kWseries1 = pandafy_data(bills[0])
	kWseries2 = pandafy_data(bills[1])
	kWseries3 = pandafy_data(bills[2])

	print "Bills Pandafied from DB @ %f" % (time.clock())

	# make one large training set
	training_data = pd.concat([kWseries1, kWseries2, kWseries3])
	big_MA = pd.DataFrame.ewm(training_data.to_frame(), halflife = .1)

	# organize, fit, predict
	y0, X0, prev_load_list, times_list, dates = organize_data(training_data)
	main_model = neighbors.KNeighborsRegressor(num_neighbors, weights = 'distance')

	print "Model Made @ %f" % (time.clock())

	main_model.fit(X0,y0.reshape(-1,1))

	print "Model Fit @ %f" % (time.clock())

	pred = main_model.predict(X0)

	'''

	this does a prediction on training set

	# make AR prediction generator
	AR_predictor = do_autoregressive_prediction(main_model, steps, times_list, prev_load_list)

	# cycle through, add each to array
	temp = []
	for i in AR_predictor:
		temp.append(i)

	# squish all into array
	AR_prediction = np.concatenate(temp)

	'''

	print "Update DB Start @ %f" % (time.clock())

	# save db stuff for plotting
	X_db, y_db, pred_db, series_db = update_db(main_model)

	print "DB Updated %f" % (time.clock())

	T1 = series_db[-(steps+start_point):]
	real_data = T1[:steps]

	all_at_once = main_model.predict(X_db[-(start_point+steps):])

	# print "Real Data, AR_Prediction, All At Once"
	# for i in range(len(real_data.values)):
	# 	print real_data.values[i], AR_prediction[i][0], all_at_once[i][0]

	plt.plot(real_data.values, color = 'b', label = "Real Data" )
	plt.plot(pred_db, color = 'r', label = "Predicted Data, RMS: %f" % (sqrt(mean_squared_error(real_data, pred_db))))
	plt.plot(all_at_once, color = 'g', label = "All At Once, RMS: %f" % (sqrt(mean_squared_error(real_data, all_at_once))))
	plt.legend()
	plt.xlim(0,steps)
	plt.ylabel('kW')
	plt.xlabel('Hour')
	plt.title('Power Draw - Window: %d, Neighbors: %d' % (window, num_neighbors))

	print "Plots Plotted @ %f" % (time.clock())

	plt.show()

	return main_model, y0, X0, training_data

window = 12
steps = 96
num_neighbors = 10
start_point = 0

main_model, y0, X0, training_data = main()

		


# TODO rigorous testing for length of window, n value, training data
# TODO split up data based on moving average
# TODO look at data based on day
# TODO include points from previous day in forecast? ie average or max,min



