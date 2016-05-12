class User < ActiveRecord::Base
	has_many :bills

	require 'csv'

	def self.import(file, user_id)

		#TODO use iter to return lines until the end
		#TODO make interpretation of csv more robust to weird inputs
  		


  		begin 
			rows = CSV.read(file.path)
			i = 6
			data_arr = []

			puts Date.parse(rows[i][1])
			date = Date.strptime(rows[i][1], "%Y-%m-%d")

		rescue
			date = Date.strptime(rows[i][1], "%m/%d/%Y")
		ensure


			while i < rows.size-7 do

				date_sec = date.strftime("%s").to_i

				hour = rows[i][2].split(":").first
				hour_sec = hour.to_i * 3600

				tot_sec = (date_sec + hour_sec).to_s

				data_arr << [tot_sec, rows[i][4]]

				i += 1

			end

			data_str = JSON.generate(data_arr)

			@hash = {:data => data_str, :user_id => user_id}

	#		puts @user.id

			Bill.create!(@hash)

		end
	end
end
