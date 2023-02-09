Simple Machine Learning exercise.

step 1

Read input data from csv file to Panda Dataframe.
See function: read_csv_file_to_dataframe in data_prep.py

step 2

Clean the data.
We will use: 'Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs' as predictors.
These fields can only contain numerical values.
From the file we got mostly strings, so we will convert that to float.
See function: remove_illegal_chars in data_prep.py

step 3

Now we need to check the categories.
These must be converted to numbers.
We also need to combine a few categories in this dataset.
See function: simplify_categories in data_prep.py

step 4

We might now use all the rows (categories) in the dataset.
Remove the rows you don't need.
See function: extract_rows_we_need in data_prep.py

step 5

Normalize the data.

step 6

Split the data in a train-set and a test-set.

step 7

Convert panda dataframe to numpy

step 5,6,7

Now split the data in a train-set and a test-set.
You can use: from sklearn.model_selection import train_test_split
Or make your own split method
In this example I have combined the creation of train- and test-dataset
with normalizing the data en converting the panda dataframe to numpy.
See data_prep.py for all the details.

step 8

Now finally we have our data ready and we can start to machine-learn.
See ml_modules.py for all the details.
With most models this is very straightforward.
1. Create an empty classifier (model).
2. Train your model with the "fit" method.
3. Use your trained model to predict the test-set. (with the "predict" method).
4. Display the results (accuracy score) to see how good the model is.
