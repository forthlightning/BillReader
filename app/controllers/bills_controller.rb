class BillsController < ApplicationController

	def import
		puts @user

		Bill.import(params[:file])

		redirect_to "users#show", notice: "Products imported."
	end

	
	private

		def set_bill
			@bill = Bill.find(params[:id])
		end

		def bill_params
			params.require(:bill).permit(:id, :data, :user_id, :bill)
		end
end