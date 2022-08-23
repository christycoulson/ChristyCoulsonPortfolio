# Capstone - MovieLens Project.
# For this project, you will be creating a movie recommendation system using the MovieLens data set.
# You will be creating your own recommendation system using all the tools we have shown you throughout the courses in this series. 
# We will use the 10M version of the MovieLens data set to make the computation a little easier.

# In order to boost the accuracy of my recommendation system, I will need to account for a number of different biases.
# I will also need to apply regularisation to the biases that I'm able to, in order to penalise small sample sizes. 
# The three biases I will account for are movie, user & genre bias. 

# The first is the starting point - the overall average of movie ratings.
mu_hat <- mean(edx$rating) # This is our starting point, the average of all ratings which minimises our residual mean squared.
naive_rmse <- RMSE(validation$rating, mu_hat) # RMSE on test set relative to average is over 1 (more than one star off, on average). 
# This is naive as possible for a prediction whilst still being informed by the data, measuring ratings versus the average and computing the RMSE. 

# First, I need to separate edx (original train data) into a train & test set as I can only use the validation set for RMSE calculation. 
# Next, I need to add the MOVIE effect,
# After that, I need to account for USER effect
# Following the user effect, I need to account for GENRE effect.
# I need to build a function with the above 3, and apply regularisation to the biases that I'm able to, in order to penalise small sample sizes. 
# In order to do so, I will need to find the optimal lambda and introduce it to my function. 
# Then, I need to apply this function to the test set to see what the RMSE is
# After I'm satisfied that my function is as expected, I'll apply to VALIDATION set (final) and measure overall RMSE. 

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index,] # TRAINING SET W/ 80% of data
test <- edx[test_index,] # TEST SET W/ 20% of data

# I've now got a train & test set. 

mu_hat <- mean(train$rating) # This is our starting point, the average of all ratings which minimises our residual mean squared
naive_rmse <- RMSE(test$rating, mu_hat) # RMSE on test set relative to average is over 1 (more than one star off, on average). 
# This is naive as possible for a prediction whilst still being informed by the data, measuring ratings versus the average and computing the RMSE. 

# We know from experience that some movies are just generally rated higher than others.
# We can see this by simply making a plot of the average rating that each movie got.
# So our intuition that different movies are rated differently is confirmed by data. We can add term b-i to represent the average rating for movie i.
# We will incorporate this movie-by-movie variation into our recommendation by determining the residual of the specific movie's average rating
# versus the average rating of ALL movies combined. 

movie_avgs <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat)) # Remember, the overall average is about 3.5. So a b i of 1.5 implies a perfect five-star rating.

# Now, use the test set.

predicted_ratings_1_test <- mu_hat + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i 

model_1_rmse_test <- RMSE(predicted_ratings_1_test, test$rating, na.rm = TRUE) # 0.9437

# Adding b_i which is the MOVIE effect.

predicted_ratings_1 <- mu_hat + train %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i # make predictions using average value of movie rating + movie_avg deviance

# Can we make it better - what about users? are different users different in terms of how they rate movies? Are some harsh, and some not?
# Now, i need to add b-u, the user effect! 

user_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i)) # user effect here

predicted_ratings_2 <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred # 

# Now, use the test set. 

predicted_ratings_2_test <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred #

model_2_rmse_test <- RMSE(predicted_ratings_2_test, test$rating, na.rm=TRUE) # 0.8659
rm(predicted_ratings_2)

# Now I have Yu,i = mu + b-i + b-u + epsilon-u, i.
# In laymans, Prediction = avg of all movies + residual of avg of specific movie + residual of avg of specific user + error.
# It's time to add the genre effect. 

genre_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u)) # genre effect here

predicted_ratings_3 <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred # 

# Now, use the test set

predicted_ratings_3_test <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred # 

model_3_rmse_test <- RMSE(predicted_ratings_3_test, test$rating, na.rm = TRUE) # 0.8655
# model_3_rmse_test signifies the RMSE for the movie, user & genre effect without regularisation. We need to perform regularisation so that large variability due 
# to small sample sizes for specific users & movies is penalised. 

# Now I have Yu,i = mu + b-i + b-u + b_g + epsilon-u, i.
# In laymans, Prediction = avg of all movies + residual of avg of specific movie + residual of avg of specific user + residual of avg of specific genre combo + error.

# Now we use regularisation to penalise the small sample sizes associated with low number of rankings per movie and per user. 
# We need to tune for hyperparameter (lambda) in order to optimise regularisation.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% # pulling pred which is average + movie effect + user effect
    pull(pred)
  return(RMSE(predicted_ratings, test$rating, na.rm=TRUE)) # Measure predicted ratings versus observed outcomes and return RMSEs for each value of lambda
})

head(test$rating)

qplot(lambdas, rmses)  
rmses

optimal_lambda <- lambdas[which.min(rmses)] # for full model that includes movie and user effects, the optimal lambda is 4.75
optimal_lambda # optimal lambda = 4.75
model_4_rmse_test <- min(rmses) # RMSE of 0.8652421 where lambda is  4.75 BUT genre effect is excluded. 

######## Now to combine regularised movie and user effects with Genre effect for final model

  mu <- mean(train$rating)
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+optimal_lambda))
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+optimal_lambda))
  b_g <- genre_avgs
  
  predicted_ratings_5 <- 
    train %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) 
  
  predicted_ratings_5_test <- 
    test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) 
  
model_5_rmse_test <- RMSE(test$rating,predicted_ratings_5_test$pred, na.rm=TRUE) # RMSE on test of 0.8649495

# Now I have an algo that has movie, user & genre effect and both movies and users have been regularised. 
# Now I want to apply it to the validation data. 

predicted_ratings_validation <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) 

model_6_validation <- RMSE(validation$rating, predicted_ratings_validation$pred, na.rm=TRUE)

# Naive Bayes predicting average

mu_validation <- mean(validation$rating)

# Adding Movie effect

b_i_validation <- validation %>%
  group_by(movieId) %>%
  summarize(b_i_validation = sum(rating - mu)/(n()+optimal_lambda))

# Adding User Effect

b_u_validation <- validation %>% 
  left_join(b_i_validation, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_validation = sum(rating - b_i_validation - mu_validation)/(n()+optimal_lambda))

# Adding Genre Effect

movie_avgs_validation <- validation %>% 
  group_by(movieId) %>% 
  summarize(b_i_val = mean(rating - mu_validation)) 

user_avgs_validation <- validation %>% 
  left_join(movie_avgs_validation, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_validation - b_i)) # user effect here

b_g_validation <- validation %>% 
  left_join(movie_avgs_validation, by='movieId') %>%
  left_join(user_avgs_validation, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g_validation = mean(rating - mu_validation - b_i - b_u)) 

# Generating Predicted Ratings

predicted_ratings_validation <- 
  validation %>% 
  left_join(b_i_validation, by = "movieId") %>%
  left_join(b_u_validation, by = "userId") %>%
  left_join(b_g_validation, by = "genreas") %>%
  mutate(pred = mu_validation + b_i_validation + b_u_validation + b_g_validation) 

# Measure RMSE for final model

model_final_RMSE <- RMSE(validation$rating, predicted_ratings_validation$pred, na.rm=TRUE) #  RMSE of 0.8394.

# Clean Script. Next steps?




