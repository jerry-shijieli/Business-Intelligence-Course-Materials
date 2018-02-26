# Load the libraries 
library(vars)
library(urca)
library(pcalg)
library(tseries)
library(stats)

# Read the input data 
filepath = "./Input Data/data.csv" # please set the R working directory to current project folder
inputData <- read.csv(filepath)

# Build a VAR model 
varModel <- VAR(inputData, ic="SC")

# Extract the residuals from the VAR model 
varResiduals <- residuals(varModel)

# Check for stationarity using the Augmented Dickey-Fuller test 
adf.test(varResiduals[,1], k=1) # test QTY
adf.test(varResiduals[,2], k=1) # test RPRICE
adf.test(varResiduals[,3], k=1) # test MPRICE

# Check whether the variables follow a Gaussian distribution  
ks.test(varResiduals[,1], "pnorm")
ks.test(varResiduals[,2], "pnorm")
ks.test(varResiduals[,3], "pnorm")

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(varResiduals[,1:3], file="varResiduals.csv", row.names = FALSE)
# PC algorithm
suffStat=list(C=cor(varResiduals), n=1000)
pc_fit <- pc(suffStat, indepTest=gaussCItest, alpha=0.05, labels=colnames(varResiduals), skel.method="original")
plot(pc_fit, main="PC Output")

# LiNGAM algorithm
lingam_fit <- LINGAM(varResiduals)
show(lingam_fit)

