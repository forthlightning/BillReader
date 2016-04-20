# Place all the behaviors and hooks related to the matching controller here.
# All this logic will automatically be available in application.js.
# You can use CoffeeScript in this file: http://coffeescript.org/

processData = (data) ->
  out = []
  dataArray = JSON.parse(data.data)
  i = 0
  while i < dataArray.length - 1
    dataArray[i][0] = parseInt(dataArray[i][0]) * 1000
    # convert to mils
    dataArray[i][1] = parseFloat(dataArray[i][1])
    out.push
      date_mils: dataArray[i][0]
      use: dataArray[i][1]
    i++
  return