library(rpart) #Popular decision tree algorithm
 library(rattle) #Fancy tree plot, nice graphical interface
 library(rpart.plot) #Enhanced tree plots
 library(RColorBrewer) #Color selection for fancy tree plot
 library(party) #Alternative decision tree algorithm
 library(partykit) #Convert rpart object to BinaryTree
 library(RWeka) #Weka decision tree J48
 library(evtree) #Evolutionary Algorithm, builds the tree from the bottom up
 library(randomForest)
 library(doParallel)
 library(CHAID) #Chi-squared automatic interaction detection tree
 library(tree)
 library(caret) 

ls(package:party) #list functions in package party

#Data Prep
 data(weather)
 dsname <- “weather”
target <- “RainTomorrow”
risk <- “RISK_MM”
ds <- get(dsname)
 vars <- colnames(ds)
 (ignore <- vars[c(1, 2, if (exists("risk")) which(risk==vars))])

vars <- setdiff(vars, ignore)
 (inputs <- setdiff(vars, target))

(nobs <- nrow(ds))
 dim(ds[vars])

(form <- formula(paste(target, “~ .”)))
set.seed(1426)
 length(train <- sample(nobs, 0.7*nobs))
 length(test <- setdiff(seq_len(nobs), train))

dim(ds)
 head(ds)
 tail(ds)
 summary(ds)
 str(ds)

#——————————————————————-

# Basic Scatterplot Matrix
 pairs(paste(“~”, paste(vars, collapse=’+'), sep=”),data=ds,
 main=”Simple Scatterplot Matrix”)

pairs(~MinTemp+MaxTemp+Rainfall+Evaporation, data =ds,
 main=”Simple Scatterplot Matrix”)

histogram(ds$MinTemp, breaks=20, col=”blue”)

#——————————————————————-

#Rpart Tree
 library(rpart)
 model <- rpart(formula=form, data=ds[train, vars])
 model
 summary(model)
 printcp(model) #printcp for rpart objects
 plotcp(model)
 plot(model)
 text(model)
 fancyRpartPlot(model)
 prp(model)
 prp(model, type=2, extra=104, nn=TRUE, fallen.leaves=TRUE,
 faclen=0, varlen=0, shadow.col=”grey”, branch.lty=3)

pred <- predict(model, newdata=ds[test, vars], type=”class”) #na.action = na.pass
 pred.prob <- predict(model, newdata=ds[test, vars], type=”prob”) 

#Check for na in the data, remove rows, if there are NA’s, rpart will use surrogate splits.
 table(is.na(ds))
 ds.complete <- ds[complete.cases(ds),]
 (nobs <- nrow(ds.complete))
 set.seed(1426)
 length(train.complete <- sample(nobs, 0.7*nobs))
 length(test.complete <- setdiff(seq_len(nobs), train.complete))

#Prune tree
 model$cptable[which.min(model$cptable[,"xerror"]),”CP”] #want the first minimum
 model <- rpart(formula=form, data=ds[train.complete, vars], cp=0)
 printcp(model)
 prune <- prune(model, cp=.01)
 printcp(prune) 

#——————————————————————-

#Party Tree
 install.packages(“partykit”, repos=”http://R-Forge.R-project.org”)
library(partykit)

class(model)
 plot(as.party(model))

#——————————————————————-

#tree
 model <- tree(formula=form, data=ds[train, vars])
 summary(model)

#——————————————————————-

#Conditional Inference Tree
 model <- ctree(formula=form, data=ds[train, vars])
 model
 plot(model)
 pred <- predict(model, newdata=ds[test, vars])
 pred.prob <- predict(model, newdata=ds[test, vars], type=”prob”) 

#Try this for class predictions:
 library(caret)
 confusionMatrix(pred, ds[test, target])
 mc <- table(pred, ds[test, target])
 err <- 1.0 – (mc[1,1] + mc[2,2]) / sum(mc) #resubstitution error rate 

#For class probabilities:
 probs <- treeresponse(model, newdata=test)
 pred <- do.call(rbind, as.list(pred))
 summary(pred)

#For a roc curve:
 library(ROCR)
 roc <- prediction(pred[,1], ds[test, target]) #noquote(paste(“test$”, target, sep=”))
plot(performance(roc, measure=”tpr”, x.measure=”fpr”), colorize=TRUE)

#For a lift curve:
 plot(performance(roc, measure=”lift”, x.measure=”rpp”), colorize=TRUE)

#Sensitivity/specificity curve and precision/recall curve:
 #sensitivity(i.e True Positives/Actual Positives) and specifcity(i.e True Negatives/Actual Negatives)
 plot(performance(roc, measure=”sens”, x.measure=”spec”), colorize=TRUE)
 plot(performance(roc, measure=”prec”, x.measure=”rec”), colorize=TRUE)

#Here’s an example of using 10-fold cross-validation to evaluation your model
 library(doParallel)
 registerDoParallel(cores=2)
 model <- train(ds[, vars], ds[,target], method=’rpart’, tuneLength=10)

#cross validation
 #example
 n <- nrow(ds) #nobs
 K <- 10 #for 10 validation cross sections
 taille <- n%/%K
 set.seed(5)
 alea <- runif(n)
 rang <- rank(alea)
 bloc <- (rang-1)%/%taille +1
 bloc <- as.factor(bloc)
 print(summary(bloc))

all.err <- numeric(0)
 for(k in 1:K){
 model <- rpart(formula=form, data = ds[train,vars], method=”class”)
pred <- predict(model, newdata=ds[test,vars], type=”class”)
mc <- table(ds[test,target],pred)
 err <- 1.0 – (mc[1,1] +mc[2,2]) / sum(mc)
 all.err <- rbind(all.err,err)
 }
 print(all.err)
 (err.cv <- mean(all.err))

#——————————————————————-

#Weka Decision Tree
 model <- J48(formula=form, data=ds[train, vars])
 model
 predict <- predict(model, newdata=ds[test, vars])
 predict.prob <- predict(model, newdata=ds[test, vars], type=”prob”) 

#——————————————————————-

#Evolutionary Trees
 target <- “RainTomorrow”
model <- evtree(formula=form, data=ds[train, vars])
 model
 plot(model)
 predict <- predict(model, newdata=ds[test, vars])
 predict.prob <- predict(model, newdata=ds[test, vars], type=”prob”) 

#——————————————————————-

#Random Forest from library(randomForest)
 table(is.na(ds))
 table(is.na(ds.complete))
 setnum <- colnames(ds.complete)[16:19] #subset(ds, select=-c(Humidity3pm, Humidity9am, Cloud9am, Cloud3pm))
 ds.complete[,setnum] <- lapply(ds.complete[,setnum] , function(x) as.numeric(x))

ds.complete$Humidity3pm <- as.numeric(ds.complete$Humidity3pm)
 ds.complete$Humidity9am <- as.numeric(ds.complete$Humidity9am)

begTime <- Sys.time()
 set.seed(1426)
 model <- randomForest(formula=form, data=ds.complete[train.complete, vars])
 runTime <- Sys.time()-begTime
 runTime
 #Time difference of 0.3833725 secs

begTime <- Sys.time()
 set.seed(1426)
 model <- randomForest(formula=form, data=ds.complete[train, vars], ntree=500, replace = FALSE, sampsize = .632*.7*nrow(ds), na.action=na.omit)
 runTime <- Sys.time()-begTime
 runTime
 #Time difference of 0.2392061 secs

model
 str(model)
 pred <- predict(model, newdata=ds.complete[test.complete, vars])

#Random Forest in parallel
 library(doParallel)
 ntree = 500
 numCore = 4
 rep <- 125 # tree / numCore
 registerDoParallel(cores=numCore)
 begTime <- Sys.time()
 set.seed(1426)
 rf <- foreach(ntree=rep(rep, numCore), .combine=combine, .packages=’randomForest’) %dopar%
 randomForest(formula=form, data=ds.complete[train.complete, vars],
 ntree=ntree,
 mtry=6,
 importance=TRUE,
 na.action=na.roughfix, #can also use na.action = na.omit
 replace=FALSE)
 runTime <- Sys.time()-begTime
 runTime
 #Time difference of 0.1990662 secs
 importance(model)
 importance(rf)

pred <- predict(rf, newdata=ds.complete[test.complete, vars])
 confusionMatrix(pred, ds.complete[test.complete, target])

#Random Forest from library(party)
 model <- cforest(formula=form, data=ds.complete[train.complete, vars])

#Factor Levels
 id <- which(!(ds$var.name %in% levels(ds$var.name)))
 ds$var.name[id] <- NA

#——————————————————————-

#Regression Trees – changing target and vars
 target <- “RISK_MM”
vars <- c(inputs, target)
 form <- formula(paste(target, “~ .”))
 (model <- rpart(formula=form, data=ds[train, vars]))
 plot(model)
 text(model)
 prp(model, type=2, extra=101, nn=TRUE, fallen.leaves=TRUE,
 faclen=0, varlen=0, shadow.col=”grey”, branch.lty=3)

rsq.rpart(model)
 library(Metrics)
 pred <- predict(model, newdata=ds[test, vars])
 err <- rmsle(ds[test, target], pred) #compare probabilities not class 

#——————————————————————-

#Chaid Tree – new data set
 data(“BreastCancer”, package = “mlbench”)
sapply(BreastCancer, function(x) is.factor(x))
 b_chaid <- chaid(Class ~ Cl.thickness + Cell.size + Cell.shape +
 + Marg.adhesion + Epith.c.size + Bare.nuclei +
 + Bl.cromatin + Normal.nucleoli + Mitoses,
 data = BreastCancer)
 plot(b_chaid)

#——————————————————————-

#List functions from a package
 ls(package:rpart)

#save plots as pdf
 pdf(“plot.pdf”)
fancyRpartPlot(model)
 dev.off()

#———————————————————————————————–
#———————————————————————————————–
#———————————————————————————————–
