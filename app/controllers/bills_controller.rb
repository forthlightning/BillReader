class BillsController < ApplicationController

	private

		def set_bill
			@bill = Bill.find(params[:id])
		end

		def bill_params
			params.require(:bill).permit(:id, :data, :user_id, :bill)
		end
end