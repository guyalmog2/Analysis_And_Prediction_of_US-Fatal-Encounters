# Analysis_And_Prediction_of_US-Fatal-Encounters
# Overview:
Since the death of George Floyd on May May 25, 2020, wide protests against the police are taking place all over the United States as well as other countries on a daily basis.
The main demand of the protesters is to stop the racial bias in the police system.
Following these events, it seems like there's a need for deeper understanding of the way things are going on in terms of law enforcement, mainly in the US.
This work tryies to provide some basic information of the data in the last few years.

# goals:
### There are to main goals to this project:
#### 1) Explore the relationships between certain variables and the chance of being involved in a fatal encounter
#### 2) Trying to predict those chances using different ML classification techniques

## Data Sources:
#### The data was gathered from different places:
* Fatal Police Shootings in the US (2015-2020) - https://www.kaggle.com/andrewmvd/police-deadly-force-usage-us
* Fatal Police Shootings in the US - https://www.kaggle.com/kwullum/fatal-police-shootings-in-the-us
* Crime rate in the United States in 2018, by state - https://www.statista.com/statistics/301549/us-crimes-committed-state/

## Coding resources:
python version - 3

packages: pandas,datetime, numpy, sklearn, scipy, matplotlib, seaborn.

# Process

## data cleaning:

* Converted numerical null values based on the median while relying on other factors

## Exploratory Data Analysis:
### First Question: What is the relationship between the victims and their proportion of the total population in terms of race?

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/deaths_share_pop.png?raw=true)


* the data suggests that the correlations between blacks and whites are in the opposite directions: while the percent of white victims is *lower* then their proportion of the population, black victims percentage is *bigger*.
* It's important to note though - we obviously cannot conclude if there's a bias or racism based on mere statistics, but getting to know the data is important for further investigation. 

# Question Number Two
## Is the number of victims has been on the rise for the past few years like many of the protesters and the media suggest?

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/deaths_over_time_race.png?raw=true)

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/deaths_over_time_months.png?raw=true)

### Big lesson here - when diving deeper and scaling down to months instead of years, we cannot see any difference in the total number of victims. the somewhat good news are that although the numbers didn't decrease, they didn't increase either, as oppposed to what many people think or say 

# Question number three
## How other categorial factors in the data are distributed

Insights:
Most of the victims were not fleeing
More then 70 percent of the victims didn't have signs of mental illness
Important one - more then 60 percent of the victims are labeled as "attackers"
More than 90 percent are males

# Question Number Four
## Is their any difference between armed and unarmed victims?
#### To answer this question:
* First, I reduced the number of values for the feature 'armed' to the most common ones
* Second, I created a column which specifies if the victim was armed or not
* Lastly, I created a dataframe for armed and unarmed victims and compared them

#### Results:

* Race - both armed and unarmed data seems to be similar to each other.
* Gender - since the absolute majority of the victims are men i didn't expect to see any different pattern
* Age - both armed and unarmed victims' age distribution are peaked between late teens and mid 30s
* Weapon - most of the victims who were armed had a gun, while most of the unarmed labeled as 'Unknown' - which could be gun as well but the data doesn't tell us much more


# Question Number Five
## Is there any difference between the number of victims per 100K people for each state?

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/by_state_map.png?raw=true)


* Seems like Alaske and New mexico are leading with the number of victims via police shootings.

* We can further ask what's the reason for that if there is any, let's look deeper into each state

## Heatmap of the top 15 correlators with the number of victims per 100K citizens:

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/correlations.png?raw=true)

## Few Insights Here:
1) *Crime rate* is a correlation coefficient of 0.61 which is very high and not surprising.

2) *Native* population is positively associated with the number of victims, this could be for a number of reasons, one of them is the correlation to crime rate which is 0.34.

3) A large *private work* sector is negatively correlated to the number of victims. while not correlated with Poverty, it is negatively correlated with crime rate, which maybe can tell us something about the importance of the private work sector.

4) *Education* - Again, im not surprised to see a negative correlation between the share of people who finished highschool and number of victims as well as crime rate. Education is important.

5) Some features are highly correlated to one another as they represent the same thing basically(child poverty - poverty, private work - public work etc..), it's important to mention that because it would be necessary to handle those features later when we perform regressions.

### Some scatterplots of the highly correlated variables:

![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/scatter_crime.png?raw=true)


![alt text](https://github.com/guyalmog2/Analysis_And_Prediction_of_US-Fatal-Encounters/blob/master/scatter_poverty.png?raw=true)


# Alright, i hope you got some visual insights about our variables. Now moving on to the modelling
### The purpose of this stage is to create  a model which can predict with high accuracy the number of victims per 100K citizens using our features from the data. In order to do that:
1) I had a look at our target variable, and noticed it is slightly skewed, therefore i transformed it's values using log, and got an output of a nice normal distributed variable

2) I created an x variable by selecting features who seemed relevant for our predictions, while having in mind the correlations between each of them to one another and the variance they have - for example, i didn't include Men/Women, as well as child poverty becuase, as we've seen earlier, the absolute majority of victims are men, and child poverty and poverty are basically the same thing

3) After createing x and y, i performed a scaling transformation on x values in order to avoid biases towards some of the features

3) I then splitted the data into test and train while using 0.7 of the data to be for training

4) After all the feature engineering has done and the data is splitted and scaled, i performed multiple regressions  simultaneously using cross validation, and presented the results with a dataframe and box plots.

5) I repeated stage 2 with some different features to see if the accuracy improved(the mean squared error)

6) At this point, I chose the best model from the x who showed the best results, and used randomized search grid in order to improve the accuracy even more

7) Finally, i presented the features' coefficients of the model after tuning it, Enjoy!

## Results from the third x, which had the best results:
Model |	neg_mean_squared_error(CV) |	neg_mean_squared_error(Test_Data) |	Std	| difference
------------ | ------------- | ------------- | ------------- | -------------
Lasso |	0.128302	| 0.103680 | 0.074093 |	0.024622
RandomForest	| 0.062395 |	0.034825 |	0.038366 |	0.027570
XGB |	0.073877	| 0.037443	| 0.038406	| 0.036434
LR	| 2.102318	| 0.172650 |	3.394712 |	1.929667
SVR|	0.052020	| 0.053931 |	0.028861	| -0.001911
Enet |	0.128302	| 0.103680	| 0.074093 |	0.024622
LightGBM |	0.128302	| 0.103680 |	0.074093 |	0.024622
Bayes |	0.044148	| 0.033433 |	0.014932 |	0.010716
GB	| 0.087083	| 0.031071 |	0.053879 |	0.056012



## Gradient Booster  had the best results with the third x! After tuning it, the final squared mean error was 0.036170820501416
## the features' coefficients:

Coefficient | value
------------ | -------------
crime_rate|	0.196895
PrivateWork	|0.138418
Native|	0.135286
Carpool|	0.126850
Construction|	0.081782
Pacific|	0.044664
HS_over_25|	0.040108
Hispanic|	0.030082
Service|	0.028256
SelfEmployed|	0.025869
Asian|	0.024317
OtherTransp|	0.017444
Men_ratio|	0.015049
Production|	0.014802
Unemployment|	0.009930
FamilyWork|	0.009629
MeanCommute	|0.008947
Income Per Capita|0.007187
WorkAtHome|	0.006952
Office|	0.006305
Professional|	0.005433
Poverty|	0.005290
Drive|	0.004730
White|	0.004552
Black|	0.004203
Transit|	0.003784
Walk	|0.003235

# Thank You!
