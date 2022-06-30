# set working directory and load UKL data
setwd("/Users/candice/Desktop/Dissertation/project1")
load("UKL.rda")

# overview UKL data and read the data type of each variable
head(UKL)
str(UKL)

# there is no missing value in the data
sum(is.na(UKL))

# transform UKL data
# convert MW to GW (1GW=1000MW) for clear view
UKL$load <- UKL$load/1000
UKL$load48 <- UKL$load48/1000

# change the level of categorical variable, dow, for graphical use
UKL$dow <- factor(UKL$dow,levels=c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"))

## Exploring data analysis
library(dplyr)
library(ggplot2)
library(mgcv)

# look at correlation plot for numerical features
library(psych)
UKL_num <- select(UKL,-c('date','dow'))
corPlot(UKL_num, cex = 0.7)

# yearly pattern, colored in different years
# overall descending trend
ggplot(data = UKL, aes(date, load, group = year, colour = year)) +
  geom_line(size = 0.8) +
  theme_bw() +
  labs(x = "Year", y = "UK Half Hourly Load (GW)",
       title = "Yearly Seasonality and Descending Pattern")

# temperature pattern
# high load in cold weather and relative high load in hot weather
ggplot(data=UKL,aes(temp,load)) + geom_line() +
  labs(title = "Relationahsip Between Temprature and Load",
       x = "Temprature", y = 'UK Half Hourly Load (GW)')

# look at weekly and daily pattern in both summer and winter
# inspired by https://bar.rady.ucsd.edu/Viz1.html
# weekly-daily pattern in winter, using mean load of 48 time intervals
# different color represents the different daily pattern in the week
winter <- filter(UKL,month==11)
winter %>%
  group_by(dow,tod) %>%
  summarize(mean.load=mean(load)) %>%   
  ggplot(aes(x=tod,y=mean.load, group=dow, color=dow)) + 
  geom_point(size=3) + 
  geom_line(size=0.5) + 
  facet_wrap(~dow,nrow=1) + 
  theme(legend.position="none") +
  labs(title = "Weekly and Daily Pattern - Winter",
       x = "Half-hourly Time of Day",
       y = "UK Load (GW)")

# Weekly-daily pattern in summer
summer <- filter(UKL,month==6)
summer %>%
  group_by(dow,tod) %>%
  summarize(mean.load=mean(load)) %>%
  ggplot(aes(x=tod,y=mean.load, group=dow, color=dow)) + 
  geom_point(size=3) + 
  geom_line(size=0.5) + 
  facet_wrap(~dow,nrow=1) + 
  theme(legend.position="none") +
  labs(title = "Weekly and Daily Pattern - Summer",
       x = "Half-hourly Time of Day",
       y = "UK Load (GW)")

## Modeling
# separate training and test dataset
# make 2016 load data to be test set
ind <- which(UKL$year!=2016)
train  <- UKL[ind,]
test   <- UKL[-ind, ]

# fit model
# compare models using RMSE and MAPE
# the following two packages are used for calculating the two statistics above
library(MLmetrics)
library(Metrics)

# M1: the simplest model
# mainly used to test if the interaction between temperature and lagged
# temperature is significant
m1 <- bam(load~s(load48)
              +s(tod)
              +s(temp,k=10)
              +s(temp95,k=10)
              +ti(temp,temp95,k=30),
          data=train, discrete=TRUE)
# check if k is large enough and residual check
gam.check(m1)
# provide model result
summary(m1)

# predict load for 2016 using M1
test.pred <- predict(m1,test)

# compute MAPE and RMSE for test data
MAPE_1 <- MAPE(test.pred,test$load)
RMSE_1 <- rmse(test$load,test.pred)

# compute RMSE for training data in order to test overfit/underfit
RMSE_1_tr <- rmse(train$load,m1$fitted.values)

# M2: smooth of one-day lagged load depends on level of dow
m2 <- bam(load~s(load48,by=dow,k=20)
              +dow
              +s(tod)
              +s(temp,temp95,k=20),
          data=train, discrete=TRUE)
# check if k is large enough and residual check
gam.check(m2)
# provide model result
summary(m2)

# predict load for 2016 using M2
test.pred <- predict(m2,test)

# compute MAPE and RMSE for test data
MAPE_2 <- MAPE(test.pred,test$load)
RMSE_2 <- rmse(test$load,test.pred)

# compute RMSE for training data in order to test overfit/underfit
RMSE_2_tr <- rmse(train$load,m2$fitted.values)

# M3:  daily pattern of one-day lagged load varies in the week
m3 <- bam(load~te(load48,tod,by=dow,k=20)
          +dow
          +s(temp,temp95,k=20),
          data=train, discrete=TRUE)
# check if k is large enough and residual check
# residuals do not randomly distributed, auto-correlation exists
gam.check(m3)
# provide model result
summary(m3)

# predict load for 2016 using M3
test.pred <- predict(m3,test)

# compute MAPE and RMSE for test data
MAPE_3 <- MAPE(test.pred,test$load)
RMSE_3 <- rmse(test$load,test.pred)

# compute RMSE for training data in order to test overfit/underfit
RMSE_3_tr <- rmse(train$load,m3$fitted.values)

# further check the acf plot of residuals of M3
# there is auto-correlation
acf(m3$residuals,main="")
  
# to see if adding auto-correlation to residuals
# choose the rho which minimises AIC score and REML score
aic <- fREML <- as.numeric()
rho <- seq(0.9,0.99,by=.01)
for (i in 1:length(rho)) {
   b <- bam(load~te(load48,tod,by=dow,k=20)
                +dow
                +te(temp,temp95,k=20),
            data=train, discrete=TRUE,
            rho=rho[i])
   aic[i] <- AIC(b)
   fREML[i] <- b$gcv.ubre
}

# the smallest rho is 0.97
# which means eps~AR(0.97)
aic; fREML

# M4:add auto-correlation error to M3
m4 <- bam(load~te(load48,tod,by=dow,k=20)
              +dow
              +te(temp,temp95,k=20),
              data=train, discrete=TRUE,
              rho=0.97)

# check the acf plot of standardized residuals
# now, the residuals seem to be random
acf(m4$std.rsd,main="")

# check if adding auto-correlation part will improve the model accuracy
# however, model performs worse on test data
# we still want M3 to be our final model
test.pred <- predict(m4,test)
MAPE_4 <- MAPE(test.pred,test$load)
RMSE_4 <- rmse(test$load,test.pred)
RMSE_4_tr <- rmse(train$load,m4$fitted.values)

# AIC values to compare three models, confirming M3 is the best
AIC(m1,m2,m3)

# plot predicted vs actual load on test set (first 75 days)
test.pred <- predict(m3,test)
test_pred <- data.frame(date=test$date[1:3600],load=test.pred[1:3600])
data_pred <- bind_rows(test[1:3600,][c('date','load')],test_pred)
data_pred$value <- c(rep('Actual',3600),rep('Predicted',3600))

# grey line is the observed load, black line is the predicted load
ggplot(data = data_pred, aes(date, load, group = value, colour = value)) +
  geom_line(size = 0.8) +
  theme_bw() +
  labs(x = "Time", y = "Load (GW)",
       title = "Fit from GAM Model") +
  scale_color_manual(values = c("grey","black"))


