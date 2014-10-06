## This clears the console window 
cat("\014")
rm(list=ls()) #will remove ALL objects 

## Read the data
data <-read.table("/home/christian/workspace_python/MasterThesis/LegoBricks/TextFile.txt")

##Remove the first column
data = data[,1]
## Show histogram
hist(data)

#Mean of data
mean_data = mean(data)
mean_data

#Standard deviation
sd_data = sd(data)
sd_data

#Data is in range within 3 sd
numberOfSd = 3
lowBound_data = mean_data - numberOfSd*sd_data
lowBound_data
highBound_data = mean_data + numberOfSd*sd_data
highBound_data

#Length of data si:
length(data)