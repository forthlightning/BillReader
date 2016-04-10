class UsersController < ApplicationController

	def show
		@user = User.find(params[:id])
		begin
			@bill = Bill.find(params[:id])
		rescue

		ensure

		end


	end


	def new

	end


	def index
		@users = User.all
	end


	def create

		#render plain: params[user_params].inspect # use to see incoming parameters


		@user = User.new(user_params)
		if @user.save
			redirect_to @user
		else
			render 'new'
		end
	end


	def import

		#puts request.original_url.last

		User.import(params[:file],params[:id])

		redirect_to root_url, notice: "Products imported."

	end


	private

		def set_user
			@user = User.find(params[:id])
		end

		def user_params
			params.require(:user).permit(:id, :name, :survey, :user)
		end


end
