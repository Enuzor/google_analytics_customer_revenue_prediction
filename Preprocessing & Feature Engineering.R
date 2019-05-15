library(plyr)
library(lubridate)
library(tidyverse)
library(gridExtra)


#1. Read in the Data
ctypes <- cols(trafficSource_adwordsClickInfo.isVideoAd = col_logical(),
               trafficSource_isTrueDirect = col_logical(),
               device_isMobile = col_logical(),
               fullVisitorId = col_character(),
               channelGrouping = col_factor(),
               date = col_datetime())

train <- read_csv("/Users/peterfagan/Desktop/train.csv",na = c("not available in demo dataset", "(not provided)","(not set)", "<NA>", "unknown.unknown",  "(none)","","NA"),col_types = ctypes)
test <- read_csv("/Users/peterfagan/Desktop/test.csv",na = c("not available in demo dataset", "(not provided)","(not set)", "<NA>", "unknown.unknown",  "(none)","","NA"),col_types = ctypes)


good_cols = c("channelGrouping","date","fullVisitorId","visitNumber","visitStartTime","device_browser","device_deviceCategory",
              "device_isMobile","device_operatingSystem","geoNetwork_city","geoNetwork_continent","geoNetwork_country","geoNetwork_metro","geoNetwork_networkDomain","geoNetwork_region","geoNetwork_subContinent","totals_bounces",                              
              "totals_hits","totals_newVisits","totals_pageviews","totals_sessionQualityDim","totals_timeOnSite","totals_totalTransactionRevenue","totals_transactionRevenue",
              "totals_transactions","trafficSource_adContent","trafficSource_adwordsClickInfo.adNetworkType","trafficSource_adwordsClickInfo.isVideoAd",
              "trafficSource_adwordsClickInfo.page", "trafficSource_adwordsClickInfo.slot","trafficSource_campaign","trafficSource_isTrueDirect","trafficSource_keyword","trafficSource_medium","trafficSource_referralPath","trafficSource_source")

train <- train %>% select(c(good_cols))
test <- test %>% select(c(good_cols))
train <- as.data.frame(train)
test <- as.data.frame(test)


#Exploratory Data Analysis

#Plot of NA values of non-numeric variables
notnum <- function(x){
  if(is.numeric(x)){
    return(FALSE)
  }
  else{
    return(TRUE)
  }
}

sub <- Filter(notnum,train)
nrows = dim(sub)[1]
ncols = dim(sub)[2]
na_count = numeric(ncols)
for(i in 1:ncols){
  na_count[i] = sum(is.na(sub[,i]))
}
df_navals = as.data.frame(names(sub))
df_navals$na_count = na_count
df_navals$na_percentage = ((df_navals$na_count/nrows)*100)
colnames(df_navals)<-c("Variables","Na_count","Na_percent")
ggplot(data=df_navals,aes(x=reorder(Variables,Na_percent),y=Na_percent))+geom_bar(stat='identity',colour='black',fill='royalblue')+coord_flip()+ggtitle("Missing Values by Feature")+xlab("Features")+ylab("Missing Values (%)")


#(***Optional***) Drop non-numeric variables with more than 95% NA values
drop_list = as.character(df_navals[df_navals$na_percentage>0.95,][,1])
train = train[,!(names(train) %in% drop_list)]


# Target variable visualisations

#Target variable behaviour over time
train$totals_totalTransactionRevenue <- ifelse(is.na(train$totals_totalTransactionRevenue),0,train$totals_totalTransactionRevenue)
test$totals_totalTransactionRevenue <- ifelse(is.na(test$totals_totalTransactionRevenue),0,test$totals_totalTransactionRevenue)

target_agg <- aggregate((train$totals_totalTransactionRevenue*1e-06), by = list(train$date),sum)
plot(target_agg,t='l',main="Transaction revenue time series",xlab= "Year",ylab="Total transaction revenue")

#Target variable distribution
target = train$totals_totalTransactionRevenue
target_spent = log(target[target>0] +1)
target_spent = as.data.frame(target_spent)
colnames(target_spent)<-c("log_target")
ggplot(target_spent,aes(log_target,stat(density))) + geom_histogram(colour='black',fill='royalblue')+ggtitle("Transaction Revenue")+xlab("log(transaction_revenue +1)")+ylab("%")

#Distribution of expenditure
no_spend_percent = (length(target)-(dim(target_spent)[1]))/length(target)
spend_percent = (dim(target_spent)[1])/length(target)
df_spend = as.data.frame(rbind(spend_percent,no_spend_percent))
df_spend$V2 = c("Revenue","No Revenue")
colnames(df_spend) <- c("Percent","Groups")
ggplot(df_spend,aes(x=Groups,y=Percent))+geom_bar(stat='identity',colour='black',fill='royalblue')+ggtitle('Barplot of Site Visits Generating Revenue')

#Checking for features that remain constant or show no variability
unique_vals = apply(train,2,n_distinct)
unique_vals[unique_vals==1]


#Checking for categorical features with a large number of levels
cat_levels <- sapply(train,n_distinct)
for(i in 1:length(cat_levels)){
  if(cat_levels[i] > 32){
    print(names(cat_levels[i]))
  }
}

#Visualisations of categorical variables levels with repect to target variable
train <- as_tibble(train)

p1 <- train %>% 
  group_by(channelGrouping) %>%
  summarise(TransactionRevenue = sum(totals_totalTransactionRevenue),visits = n()) 
P11 <- ggplot(p1, aes(x=channelGrouping,y=visits)) + geom_bar(stat='identity',colour='black',fill='lightgreen')+ggtitle("Visits by Channel grouping") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
P12 <- ggplot(p1, aes(x=channelGrouping,y=TransactionRevenue/visits)) + geom_bar(stat='identity',colour='black',fill='lightgreen')+ggtitle("Mean Transaction Revenue by Channel grouping") + theme(axis.text.x = element_text(angle = 60, hjust = 1))



p2 <- train %>% 
  group_by(device_operatingSystem) %>%
  summarise(TransactionRevenue = sum(totals_totalTransactionRevenue),visits = n()) 
P21 <- ggplot(p2, aes(x=device_operatingSystem,y=visits)) + geom_bar(stat='identity',colour='black',fill='orange')+ggtitle("Visits by operating system") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
P22 <- ggplot(p2, aes(x=device_operatingSystem,y=TransactionRevenue/visits)) + geom_bar(stat='identity',colour='black',fill='orange')+ggtitle("Mean Transaction Revenue by operating system") + theme(axis.text.x = element_text(angle = 60, hjust = 1))


p3 <- train %>% 
  group_by(geoNetwork_continent) %>%
  summarise(TransactionRevenue = sum(totals_totalTransactionRevenue),visits = n()) 
P31 <- ggplot(p3, aes(x=geoNetwork_continent,y=visits)) + geom_bar(stat='identity',colour='black',fill='yellow')+ggtitle("Visits by Continent") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
P32 <- ggplot(p3, aes(x=geoNetwork_continent,y=TransactionRevenue/visits)) + geom_bar(stat='identity',colour='black',fill='yellow')+ggtitle("Mean Transaction Revenue by Continent") + theme(axis.text.x = element_text(angle = 60, hjust = 1))

p4 <- train %>% 
  group_by(device_deviceCategory) %>%
  summarise(TransactionRevenue = sum(totals_totalTransactionRevenue),visits = n()) 
P41 <- ggplot(p4, aes(x=device_deviceCategory,y=visits)) + geom_bar(stat='identity',colour='black',fill='lightblue')+ggtitle("Visits by device category") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
P42 <- ggplot(p4, aes(x=device_deviceCategory,y=TransactionRevenue/visits)) + geom_bar(stat='identity',colour='black',fill='lightblue')+ggtitle("Mean Transaction Revenue by device category") + theme(axis.text.x = element_text(angle = 60, hjust = 1))


grid.arrange(P11,P12,P21,P22,P31,P32,P41,P42, ncol = 2, nrow = 4)


g1 <- ggplot(train, aes(x=train$channelGrouping,y=train$totals_totalTransactionRevenue)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 60, hjust = 1))
g2 <- ggplot(train, aes(x=train$device_operatingSystem,y=train$totals_totalTransactionRevenue)) + geom_boxplot()+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
g3 <- ggplot(train, aes(x=train$geoNetwork_continent,y=train$totals_totalTransactionRevenue)) + geom_boxplot()+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
g4 <- ggplot(train, aes(x=train$device_deviceCategory,y=train$totals_totalTransactionRevenue)) + geom_boxplot()+ theme(axis.text.x = element_text(angle = 60, hjust = 1))

grid.arrange(g1,g2,g3,g4, ncol = 1, nrow = 4)

train <- as.data.frame(train)

Feature_engineering <- function(data)
{
  #Preprocessing features some more...
  for(i in 4:dim(data)[2]){
    if(is.character(data[,i])){
      data[,i] = as.factor(data[,i])
    }
  }
  data$visitStartTime = as_datetime(data$visitStartTime)
  data$device_deviceCategory = as.factor(data$device_deviceCategory)
  data$device_isMobile <- as.numeric(as.factor(data$device_isMobile))
  data$trafficSource_isTrueDirect = as.numeric(ifelse(is.na(data$trafficSource_isTrueDirect),FALSE,data$trafficSource_isTrueDirect))
  data$trafficSource_adwordsClickInfo.isVideoAd <- (ifelse(is.na(data$trafficSource_adwordsClickInfo.isVideoAd),0,1))
  
  
  # Feature Engineering
  #Reducing the dimension of categorical features through grouping levels with few occurences into the one level
  data$device_browser = fct_lump(data$device_browser,n=10)
  data$device_operatingSystem = fct_lump(data$device_operatingSystem,n=10)
  
  #Make NA values a factor level
  for(i in 1:dim(data)[2]){
    if(is.factor(data[,i])){
      data[,i] = fct_explicit_na(data[,i])
    }
  }
  
  #Make dates into hour,day,week,month as separate variables
  data$month_day <- as.factor(as.integer(mday(data$date)))
  data$month <- as.factor(as.character(months(data$date)))
  data$hour <- as.factor(as.POSIXlt(data$visitStartTime,origin='1970-01-01')$hour)
  data$weekday <- as.factor(wday(as.POSIXlt(data$visitStartTime, origin='1970-01-01')))
  
  
  #Target encoding geographical features
  wealth_continent <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_continent, data, mean)
  wealth_subContinent <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_subContinent, data, mean)
  wealth_country <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_country, data, mean)
  wealth_city <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_city, data, mean)
  wealth_metro <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_metro, data, mean)
  wealth_region <- aggregate(totals_totalTransactionRevenue ~ geoNetwork_region, data, mean)
  
  wealth_continent[,2] <- scale(wealth_continent[,2],center=min(wealth_continent[,2]),scale=max(wealth_continent[,2])-min(wealth_continent[,2]))
  wealth_subContinent[,2] <- scale(wealth_subContinent[,2],center=min(wealth_subContinent[,2]),scale=max(wealth_subContinent[,2])-min(wealth_subContinent[,2]))
  wealth_country[,2] <- scale(wealth_country[,2],center=min(wealth_country[,2]),scale=max(wealth_country[,2])-min(wealth_country[,2]))
  wealth_city[,2] <- scale(wealth_city[,2],center=min(wealth_city[,2]),scale=max(wealth_city[,2])-min(wealth_city[,2]))
  wealth_metro[,2] <- scale(wealth_metro[,2],center=min(wealth_metro[,2]),scale=max(wealth_metro[,2])-min(wealth_metro[,2]))
  wealth_region[,2] <- scale(wealth_region[,2],center=min(wealth_region[,2]),scale=max(wealth_region[,2])-min(wealth_region[,2]))
  
  
  
  colnames(wealth_continent) <- c("geoNetwork_continent","wealth_continent")
  colnames(wealth_subContinent) <- c("geoNetwork_subContinent","wealth_subContinent")
  colnames(wealth_country) <- c("geoNetwork_country","wealth_country")
  colnames(wealth_city) <- c("geoNetwork_city","wealth_city")
  colnames(wealth_metro) <- c("geoNetwork_metro","wealth_metro")
  colnames(wealth_region) <- c("geoNetwork_region","wealth_region")
  
  
  data <- join(data,wealth_continent,by="geoNetwork_continent",type='left')
  data <- join(data,wealth_subContinent,by="geoNetwork_subContinent",type='left')
  data <- join(data,wealth_country,by="geoNetwork_country",type='left')
  data <- join(data,wealth_city,by="geoNetwork_city",type='left')
  data <- join(data,wealth_metro,by="geoNetwork_metro",type='left')
  data <- join(data,wealth_region,by="geoNetwork_region",type='left')
  
  data$wealth_continent <- as.numeric(data$wealth_continent)
  data$wealth_subContinent <- as.numeric(data$wealth_subContinent)
  data$wealth_country <- as.numeric(data$wealth_country)
  data$wealth_city <- as.numeric(data$wealth_city)
  data$wealth_metro <- as.numeric(data$wealth_metro)
  data$wealth_region <- as.numeric(data$wealth_region)
  
  #Dropping columns that are no longer needed
  data$geoNetwork_continent <- NULL
  data$geoNetwork_subContinent <- NULL
  data$geoNetwork_country <-NULL
  data$geoNetwork_city <- NULL
  data$geoNetwork_networkDomain <- NULL
  data$geoNetwork_metro <-NULL
  data$geoNetwork_region <- NULL
  data$trafficSource_adContent  <- NULL                 
  data$trafficSource_adwordsClickInfo.adNetworkType<- NULL
  data$trafficSource_adwordsClickInfo.page    <- NULL 
  data$trafficSource_adwordsClickInfo.slot  <- NULL      
  data$trafficSource_campaign    <- NULL                 
  data$trafficSource_keyword      <- NULL                
  data$trafficSource_medium    <- NULL                   
  data$trafficSource_referralPath   <- NULL              
  data$trafficSource_source   <- NULL  
  
  #Making features numeric
  for(i in 1:dim(data)[2]){
    if(is.factor(data[,i])){
      data[,i] <- as.numeric(data[,i])
    }
  }
  return(data)
}

train <- Feature_engineering(train)
test <- Feature_engineering(test)
#Formatting original training set to reflect the prediction problem.
#Data for 168 days is combined with target variable calulated over a subsequent
#period of 62 days taking place 46 days in advance of initial period.
train$date <- ymd(train$date)
test$date <- ymd(test$date)
(max(test$date) - min(test$date))
(max(train$date) - min(train$date))

Generate_training_set <- function(data,t)
{

  #Split into windows corresponding to the traffic data and the period used to calculate
  #the target variable.
  traffic_dat_window <- data[data$date >= min(data$date)+(168*t) & data$date <= min(data$date)+(168*(t+1)),]
  target_calc_window <- data[data$date >= min(data$date)+(168*(t+1))+46 & data$date <= min(data$date)+(168*(t+1))+46+62,]
  
  
  #Returning and non-returning customers ids
  target_fvid = unique(target_calc_window$fullVisitorId)
  ret_fvid = traffic_dat_window[(traffic_dat_window$fullVisitorId %in% target_fvid),]$fullVisitorId
  nret_fvid = traffic_dat_window[!(traffic_dat_window$fullVisitorId %in% target_fvid),]$fullVisitorId
  
  #target window filtered by returned customers
  target_calc_window_ret = target_calc_window[target_calc_window$fullVisitorId %in% ret_fvid, ]
  
  #Create a dataframe for training
  dtrain = aggregate(totals_totalTransactionRevenue ~ fullVisitorId, target_calc_window_ret, function(x){log(1+sum(x))})
  dtrain$return = 1
  colnames(dtrain) = c("fullVisitorId","target","return")
  dtrain_nret = data.frame(fullVisitorId = nret_fvid, target = 0, return = 0)
  dtrain = rbind(dtrain, dtrain_nret)
  dtrain_maxdate = max(traffic_dat_window$date)
  dtrain_mindate = min(traffic_dat_window$date)
  
  
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  features <- traffic_dat_window %>%
    group_by(fullVisitorId) %>%
    summarise(
      channelGrouping = getmode(ifelse(is.na(channelGrouping),-9999,channelGrouping)),
      first_ses_from_the_period_start = min(date) - dtrain_mindate,
      last_ses_from_the_period_end = dtrain_maxdate - max(date),
      interval_dates = max(date) - min(date),
      unique_date_num = length(unique(date)),
      maxVisitNum = max(visitNumber,na.rm=TRUE),
      browser = getmode(ifelse(is.na(device_browser),-9999,device_browser)),
      operatingSystem =  getmode(ifelse(is.na(device_operatingSystem),-9999,device_operatingSystem)),
      deviceCategory =  getmode(ifelse(is.na(device_deviceCategory),-9999,device_deviceCategory)),
      wealth_continent = mean(wealth_continent),
      wealth_subContinent = mean(wealth_subContinent),
      wealth_country = mean(wealth_country),
      wealth_region = mean(wealth_region),
      wealth_metro = mean(wealth_metro) ,
      wealth_city = mean(wealth_city),
      isVideoAd_mean = mean(trafficSource_adwordsClickInfo.isVideoAd),
      isMobile = mean(ifelse(device_isMobile == TRUE, 1 , 0)),
      isTrueDirect = mean(ifelse(is.na(trafficSource_isTrueDirect) == TRUE, 1, 0)),
      bounce_sessions = sum(ifelse(is.na(totals_bounces),0,totals_bounces)),
      hits_sum = sum(totals_hits),
      hits_mean = mean(totals_hits),
      hits_min = min(totals_hits),
      hits_max = max(totals_hits),
      hits_median = median(totals_hits),
      pageviews_sum = sum(totals_pageviews, na.rm = TRUE),
      pageviews_mean = mean(totals_pageviews, na.rm = TRUE),
      pageviews_min = min(totals_pageviews, na.rm = TRUE),
      pageviews_max = max(totals_pageviews, na.rm = TRUE),
      pageviews_median = median(totals_pageviews, na.rm = TRUE),
      session_cnt = NROW(visitStartTime),
      transactions  = sum(totals_transactions,na.rm = TRUE)
    )
  
  features <- as.data.frame(features)
  dtrain = join(dtrain, features, by = "fullVisitorId")
  glimpse(dtrain)
  
  #Setting found NA values to zero (Only 10 occurences)
  dtrain$pageviews_mean = ifelse(is.na(dtrain$pageviews_mean),0,dtrain$pageviews_mean)
  dtrain$pageviews_median = ifelse(is.na(dtrain$pageviews_median),0,dtrain$pageviews_median)
  dtrain$pageviews_max <- ifelse(is.infinite(dtrain$pageviews_max),0,dtrain$pageviews_max)
  dtrain$pageviews_min <- ifelse(is.infinite(dtrain$pageviews_min),0,dtrain$pageviews_min)
  
  return(dtrain)
}

tr0 <- Generate_training_set(train,0)
tr1 <- Generate_training_set(train,1)
tr2 <- Generate_training_set(train,2)
training_set <- rbind(tr0,tr1,tr2)





Generate_testing_set <- function(data)
{
  dtrain_maxdate = max(data$date)
  dtrain_mindate = min(data$date)
  
  
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  features <- data %>%
    group_by(fullVisitorId) %>%
    summarise(
      channelGrouping = getmode(ifelse(is.na(channelGrouping),-9999,channelGrouping)),
      first_ses_from_the_period_start = min(date) - dtrain_mindate,
      last_ses_from_the_period_end = dtrain_maxdate - max(date),
      interval_dates = max(date) - min(date),
      unique_date_num = length(unique(date)),
      maxVisitNum = max(visitNumber,na.rm=TRUE),
      browser = getmode(ifelse(is.na(device_browser),-9999,device_browser)),
      operatingSystem =  getmode(ifelse(is.na(device_operatingSystem),-9999,device_operatingSystem)),
      deviceCategory =  getmode(ifelse(is.na(device_deviceCategory),-9999,device_deviceCategory)),
      wealth_continent = mean(wealth_continent),
      wealth_subContinent = mean(wealth_subContinent),
      wealth_country = mean(wealth_country),
      wealth_region = mean(wealth_region),
      wealth_metro = mean(wealth_metro) ,
      wealth_city = mean(wealth_city),
      isVideoAd_mean = mean(trafficSource_adwordsClickInfo.isVideoAd),
      isMobile = mean(ifelse(device_isMobile == TRUE, 1 , 0)),
      isTrueDirect = mean(ifelse(is.na(trafficSource_isTrueDirect) == TRUE, 1, 0)),
      bounce_sessions = sum(ifelse(is.na(totals_bounces),0,totals_bounces)),
      hits_sum = sum(totals_hits),
      hits_mean = mean(totals_hits),
      hits_min = min(totals_hits),
      hits_max = max(totals_hits),
      hits_median = median(totals_hits),
      pageviews_sum = sum(totals_pageviews, na.rm = TRUE),
      pageviews_mean = mean(totals_pageviews, na.rm = TRUE),
      pageviews_min = min(totals_pageviews, na.rm = TRUE),
      pageviews_max = max(totals_pageviews, na.rm = TRUE),
      pageviews_median = median(totals_pageviews, na.rm = TRUE),
      session_cnt = NROW(visitStartTime),
      transactions  = sum(totals_transactions,na.rm = TRUE)
    )
  
  features <- as.data.frame(features)
  #Setting found NA values to zero (Only 10 occurences)
  features$pageviews_mean = ifelse(is.na(features$pageviews_mean),0,features$pageviews_mean)
  features$pageviews_median = ifelse(is.na(features$pageviews_median),0,features$pageviews_median)
  features$pageviews_max <- ifelse(is.infinite(features$pageviews_max),0,features$pageviews_max)
  features$pageviews_min <- ifelse(is.infinite(features$pageviews_min),0,features$pageviews_min)
  
  
  return(features)
}

testing_set <- Generate_testing_set(test)


saveRDS(training_set,file="/Users/peterfagan/Desktop/training_set")
saveRDS(testing_set,file="/Users/peterfagan/Desktop/testing_set")
