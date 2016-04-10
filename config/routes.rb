MarchFourteenth::Application.routes.draw do
  root to: "users#index"

  get '/users/new', to: 'users#new', as: 'new'

  get '/users/create', to: 'users#create', as: 'create'

  get '/users/:id', to: 'users#show', as: 'show'

  resources :bills

  resources :users do
  	collection  {post :import}
  end

end