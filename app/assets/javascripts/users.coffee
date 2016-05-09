# Place all the behaviors and hooks related to the matching controller here.
# All this logic will automatically be available in application.js.
# You can use CoffeeScript in this file: http://coffeescript.org/

jQuery ->
	Morris.Line({

  element: 'graph',
  data: out,
  xkey: 'date_mils',
  ykeys: ['use', 'fitted'],
  labels: ['kWh', 'prediction']
  lineColors: ['blue','green']
  pointSize: [1,1]
  lineWidth: [2,2]

}) if (typeof out != 'undefined');
