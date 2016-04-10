class CreateBills < ActiveRecord::Migration
  def change
    create_table :bills do |t|
      t.text :data
      t.references :user, index: true, foreign_key: true

      t.timestamps null: false
    end
  end
end
