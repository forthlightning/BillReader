class User < ActiveRecord::Base
	has_many :bills

	require 'csv'

	def self.import(file, user_id)
  	
		rows = CSV.read(file.path)
		i = 6
		data_arr = []

		while i < rows.size-2 do
			puts rows[i][1]
			date = Time.strptime(rows[i][1],"%D")
			date_sec = date.to_i

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

	%x'python process_data.py'
end
