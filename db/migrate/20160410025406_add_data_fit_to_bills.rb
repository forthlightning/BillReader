class AddDataFitToBills < ActiveRecord::Migration
  def change
    add_column :bills, :data_fit, :text
  end
end
