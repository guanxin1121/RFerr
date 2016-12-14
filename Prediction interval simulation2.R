require(randomForest)
require(VSURF)
require(quantregForest)

##simulation for prediction interval, variable to predict variability
sim<-function(){
x1<-rnorm(500)
x2<-rnorm(500)
x3<-rnorm(500)
x4<-rnorm(500)
x5<-rnorm(500)
x6<-rnorm(500)
e<-rnorm(500)
b<-c(0.2, 0.5, 0.6, 0.7, 0.8, 0.9)
y=b[1]+b[2]*x1+b[3]*x2+b[4]*x3+b[5]*x4+b[6]*x5+x6*e
data<-data.frame(y, x1, x2, x3, x4, x5, x6)
}
################################functions ################################
extract.nodes <- function(rf,train.data,errs,newdata, oob="T"){
  
  pred.train <- predict(rf,newdata=train.data,nodes=T)
  nodes.train <- attr(pred.train,"nodes")
  pred.new <- predict(rf,newdata,nodes=T)
  nodes.new <- attr(pred.new,"nodes")
  n <- dim(newdata)[1]
  ntree <- rf$ntree
  res <- list()[rep(1,n)]
  for(i in 1:n){
    res[[i]] <- list()[rep(1,ntree)]
    for(j in 1:ntree){
      if(oob=="T"){
        tmp <- errs[nodes.train[,j]==nodes.new[i,j]&rf$inbag[,j]==0]  ##new data in the same node with train data, AND train data is an OOB sample for that tree
        res[[i]][[j]] <- tmp}
      if(oob=="F"){
        tmp <- errs[nodes.train[,j]==nodes.new[i,j]&rf$inbag[,j]!=0]  ##new data in the same node with train data, AND train data is an inbag sample for that tree
        res[[i]][[j]] <- tmp}
      if(oob=="A"){
        tmp <- errs[nodes.train[,j]==nodes.new[i,j]]  ##new data in the same node with train data, AND train data is all sample for that tree
        res[[i]][[j]] <- tmp}
    }
  }
  res
}

##5-fold CV QRF
cv5qrf<-function(dataX, dataY,n=10){
  len<-c()
  m<-c()
  for (rep in 1:n){
    folds<-sample(1:5, nrow(dataX), replace=T)
    ci5<-matrix(NA, nrow(dataX), 2)
    for (i in 1:5){
      index <- which(folds==i,arr.ind=TRUE)
      qrf<-quantregForest(x=dataX[-index,], y=dataY[-index])
      ci5[index,]<-predict(qrf,dataX[index,], quantiles=c(0.025, 0.975) )
    }
    len[rep]=length(which(dataY<ci5[,1]))+length(which(dataY>ci5[,2])) 
    m[rep]=mean(ci5[,2]-ci5[,1])
  }
  list(len/length(dataY), m) ##uncovered rate and mean of length
}



##5-fold CV error+QRF
cv5err<-function(dataX, dataY, n=10, nodesize=10, oob="T"){
  len<-c()
  m<-c()
  for (rep in 1:n){
    prediction<-c()
    ci<-matrix(NA, nrow(dataX), 2)
    folds<-sample(1:5, nrow(dataX), replace=T)
    for (i in 1:5){
      index <- which(folds==i,arr.ind=TRUE)
      rf<-randomForest(dataX[-index,], dataY[-index])
      error<-rf$predicted-dataY[-index]
      mod.err<-randomForest(dataX[-index,], error, nodesize=nodesize, keep.inbag = T)
      node<-extract.nodes(mod.err, dataX[-index,], error, dataX[index,], oob)
      prediction[index]<-predict(rf,dataX[index,])
      for (j in 1:length(index)){
        all<-prediction[index[j]]-unlist(node[[j]])
        ci[index[j],]<-quantile(all, c(0.025, 0.975))}
    }
    len[rep]=length(which(dataY<ci[,1]))+length(which(dataY>ci[,2])) 
    m[rep]=mean(ci[,2]-ci[,1])  
  }
  list(len/nrow(dataX), m)
}

##feature selection for random forest model, using Y as response
var<-VSURF(y~., data)
names(data[-1])[var$varselect.pred]
##feature selection for error, using OOB error as response
py<-randomForest(y~x5+x4+x3+x2+x1, data)$predicted
err<-py-data$y
var.err<-VSURF(err~., data.frame(err, data[-1]))
names(data[-1])[var.err$varselect.pred]

##5-fold QRF
r<-cv5qrf(dataX=data[names(data[-1])[var$varselect.pred]], dataY=data$y, n=100)
lapply(r, mean)  ##mean of accuracy and interval length of n replicates
lapply(r, sd)    ##standard deviation of accuracy and interval length of n replicates
##5-fold QRF+error
r2<-cv5err(dataX=data[names(data[-1])], dataY=data$y, n=100)
lapply(r2, mean)  ##mean of accuracy and interval length
lapply(r2, sd)    ##standard deviation of accuracy and interval length

##no feature selection
##simulate 100 datasets and run each method on each dataset just once
qrf.err<-c()
qrf.length<-c()
rferr.err<-c()
rferr.length<-c()
for (i in 1:100){  
data<-sim()
qrf<-cv5qrf(data[,-1], data$y, n=1)
rferr<-cv5err(data[,-1], data$y, n=1, nodesize=10)
qrf.err[i]<-qrf[[1]]    ##qrf error rate
qrf.length[i]<-qrf[[2]]   ##qrf interval length
rferr.err[i]<-rferr[[1]]   ##RFerr error rate
rferr.length[i]<-rferr[[2]]   ##RFerr interval length
}
r<-list(qrf.err, qrf.length)
r2<-list(rferr.err, rferr.length)
lapply(r, mean) 
lapply(r, sd) 
lapply(r2, mean)
lapply(r2, sd)


############plotting################
tiff("Figure 3.tif", res=300, height=3.5, width=7.5,units="in")
par(mar=c(3, 5, 3, 2))
par(mfrow=c(1,2))
boxplot(r2[[1]]*100, r[[1]]*100, names=c("RFerr", "QRF"), ylim=c(0,15),ylab="Miscoverage rate (%)")
abline(a=5, b=0)
boxplot(r2[[2]], r[[2]], names=c("RFerr", "QRF"), ylab="Interval length")
mtext("Simulation dataset", side = 3, line = -2, outer = TRUE, font=2)
dev.off()