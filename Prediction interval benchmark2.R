##prediction interval on benchmark dataset
require(mlbench)
require(alr3)
require(randomForest)
require(quantregForest)
require(Hmisc)
data("BostonHousing")
data("Ozone")
data("BigMac2003")
data("fuel2001")

##Boston Housing dataset


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

##1) error model for computing PI (use all samples to predict)

error<-c()
prediction<-c()
for (i in 1:dim(BostonHousing)[1]){
  rf<-randomForest(medv~., BostonHousing[-i,])
  prediction[i]<-predict(rf, BostonHousing[i,])
  error[i]<-prediction[i]-BostonHousing$medv[i]
}

ci<-matrix(NA, dim(BostonHousing)[1], 2)
for (i in 1:dim(BostonHousing)[1]){
mod.err<-randomForest(x=BostonHousing[-i,1:13], y=error[-i], nodesize=10, keep.inbag = T)
node<-extract.nodes(mod.err, BostonHousing[-i, ], error[-i], BostonHousing[i,])
all<-unlist(node)+prediction[i]
ci[i,]<-quantile(all, c(0.025, 0.975))
}

mean(ci[,2]-ci[,1])
length(which(BostonHousing$medv<ci[,1]))+length(which(BostonHousing$medv>ci[,2]))

##2) error model for computing PI (use inbag only samples to predict)
##mode.err$inbag contains info about inbag samples and oob samples. However in prediction, the training data would be different for each tree. Should we extract all trees and make predictions using individual trees



##quantile regression
ci2<-matrix(NA, dim(BostonHousing)[1], 3)
for (i in 1:dim(BostonHousing)[1]){
  qrf<-quantregForest(x=BostonHousing[-i,1:13], y=BostonHousing[-i,14])
  ci2[i,]<-predict(qrf,BostonHousing[i,1:13], quantiles=c(0.025, 0.5, 0.975) )
}
length(which(BostonHousing$medv<ci2[,1]))+length(which(BostonHousing$medv>ci2[,3]))
mean(ci2[,2]-ci2[,1])

##5-fold CV
folds <- cut(seq(1,nrow(BostonHousing)),breaks=5,labels=FALSE)
ci3<-matrix(NA, dim(BostonHousing)[1], 3)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  qrf<-quantregForest(x=BostonHousing[-index,1:13], y=BostonHousing[-index,14])
  ci3[index,]<-predict(qrf,BostonHousing[index,1:13], quantiles=c(0.025, 0.5, 0.975) )
}
length(which(BostonHousing$medv<ci3[,1]))+length(which(BostonHousing$medv>ci3[,3]))
mean(ci3[,2]-ci3[,1])


##Ozone dataset
oz<-Ozone[complete.cases(Ozone),]
##predict V4
##5-fold CV
folds<-sample(1:5, nrow(Ozone), replace=T)
ci4<-matrix(NA, nrow(oz), 2)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  qrf<-quantregForest(x=oz[-index,-4], y=oz[-index,4])
  ci4[index,]<-predict(qrf,oz[index,-4], quantiles=c(0.025, 0.975) )
}
length(which(oz$V4<ci4[,1]))+length(which(oz$V4>ci4[,2]))
mean(ci4[,2]-ci4[,1])


##prediction error
error.oz<-c()
prediction.oz<-c()
for (i in 1:nrow(oz)){
  rf<-randomForest(V4~., oz[-i,])
  prediction.oz[i]<-predict(rf, oz[i,])
  error.oz[i]<-prediction.oz[i]-oz$V4[i]
}

##1) error model for computing PI (use all samples to predict)
ci<-matrix(NA, nrow(oz), 2)
for (i in 1:nrow(oz)){
  mod.err<-randomForest(x=oz[-i,-4], y=error.oz[-i], nodesize=10, keep.inbag = T)
  node<-extract.nodes(mod.err, oz[-i, ], error.oz[-i], oz[i,])
  all<-unlist(node)+prediction.oz[i]
  ci[i,]<-quantile(all, c(0.025, 0.975))
}

mean(ci[,2]-ci[,1])
length(which(oz$V4<ci[,1]))+length(which(oz$V4>ci[,2]))

##bigmac dataset
##predict bigMac
##5-fold CV
folds<-sample(1:5, nrow(BigMac2003), replace=T)
ci5<-matrix(NA, nrow(BigMac2003), 2)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  qrf<-quantregForest(x=BigMac2003[-index,-1], y=BigMac2003[-index,1])
  ci5[index,]<-predict(qrf,BigMac2003[index,-1], quantiles=c(0.025, 0.975) )
}
length(which(BigMac2003$BigMac<ci5[,1]))+length(which(BigMac2003$BigMac>ci5[,2]))
mean(ci5[,2]-ci5[,1])

##prediction error

error<-c()
prediction<-c()
for (i in 1:nrow(BigMac2003)){
  rf<-randomForest(BigMac~., BigMac2003[-i,])
  prediction[i]<-predict(rf,BigMac2003[i,])
  error[i]<-prediction[i]-BigMac2003$BigMac[i]
}

##1) error model for computing PI (use all samples to predict)
ci<-matrix(NA, nrow(BigMac2003), 2)
for (i in 1:nrow(BigMac2003)){
  mod.err<-randomForest(x=BigMac2003[-i,-1], y=error[-i], nodesize=10, keep.inbag = T)
  node<-extract.nodes(mod.err, BigMac2003[-i, ], error[-i], BigMac2003[i,])
  all<-unlist(node)+prediction[i]
  ci[i,]<-quantile(all, c(0.025, 0.975))
}

mean(ci[,2]-ci[,1])
length(which(BigMac2003$BigMac<ci[,1]))+length(which(BigMac2003$BigMac>ci[,2]))

##############################################################################################################################
##dataset fuel2001
fuel2001$target=fuel2001$FuelC/fuel2001$MPC
fuel2001<-fuel2001[,c(1, 3,4,6:8)]
##predict target
##5-fold CV, QRF, repeat 10 times for mean and std
len<-c()
m<-c()
for (rep in 1:10){
folds<-sample(1:5, nrow(fuel2001), replace=T)
ci5<-matrix(NA, nrow(fuel2001), 2)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  qrf<-quantregForest(x=fuel2001[-index,1:5], y=fuel2001[-index,6])
  ci5[index,]<-predict(qrf,fuel2001[index,1:5], quantiles=c(0.025, 0.975) )
}
len[rep]=length(which(fuel2001$target<ci5[,1]))+length(which(fuel2001$target>ci5[,2]))  ##2
m[rep]=mean(ci5[,2]-ci5[,1])
}

##LOOCV, QRF
for (rep in 1:10){
  ci5<-matrix(NA, nrow(fuel2001), 2)
  for (i in 1:nrow(fuel2001)){
    qrf<-quantregForest(x=fuel2001[-i,1:5], y=fuel2001[-i,6])
    ci5[i,]<-predict(qrf,fuel2001[i,1:5], quantiles=c(0.025, 0.975) )
  }
  len[rep]=length(which(fuel2001$target<ci5[,1]))+length(which(fuel2001$target>ci5[,2]))  
  m[rep]=mean(ci5[,2]-ci5[,1])
}



##using only out of bag samples for PI
#"T" for oob, "F" for in bag, "A" for all samples

##OOB error for error model, LOOCV
prediction<-c()
for (rep in 1:10){
ci<-matrix(NA, nrow(fuel2001), 2)
for (i in 1:nrow(fuel2001)){
  rf<-randomForest(target~., fuel2001[-i,1:6])
  error<-rf$predicted-fuel2001$target[-i]
  mod.err<-randomForest(x=fuel2001[-i,1:5], y=error, nodesize=20, keep.inbag = T)
  node<-extract.nodes(mod.err, fuel2001[-i, ], error, fuel2001[i,], oob="T")
  
  prediction[i]<-predict(rf,fuel2001[i,])  
  all<-prediction[i]-unlist(node)
  ci[i,]<-quantile(all, c(0.025, 0.975))
}
len[rep]=length(which(fuel2001$target<ci[,1]))+length(which(fuel2001$target>ci[,2]))  
m[rep]=mean(ci[,2]-ci[,1])
}


##5-fold CV, error model
for (rep in 1:10){
prediction<-c()
ci<-matrix(NA, nrow(fuel2001), 2)
folds<-sample(1:5, nrow(fuel2001), replace=T)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  rf<-randomForest(target~., fuel2001[-index,1:6])
  error<-rf$predicted-fuel2001$target[-index]
  mod.err<-randomForest(x=fuel2001[-index,1:5], y=error, nodesize=20, keep.inbag = T)
  node<-extract.nodes(mod.err, fuel2001[-index, ], error, fuel2001[index,], "F")
  prediction[index]<-predict(rf,fuel2001[index,1:5])
  for (j in 1:length(index)){
    all<-prediction[index[j]]-unlist(node[[j]])
    ci[index[j],]<-quantile(all, c(0.025, 0.975))}
}
len[rep]=length(which(fuel2001$target<ci[,1]))+length(which(fuel2001$target>ci[,2]))  ##2
m[rep]=mean(ci[,2]-ci[,1])  
}


##discretize fuel2001$target
fuel2001$target2<-as.numeric(cut(fuel2001$target, seq(0, 2000, 100)))
##OOB error for error model, LOOCV
prediction<-c()
ci<-matrix(NA, nrow(fuel2001), 2)
for (i in 1:nrow(fuel2001)){
  
  rf<-randomForest(target2~., fuel2001[-i,-6])
  error<-rf$predicted-fuel2001$target2[-i]
  mod.err<-randomForest(x=fuel2001[-i,1:5], y=error, nodesize=20, keep.inbag = T)
  node<-extract.nodes(mod.err, fuel2001[-i, ], error, fuel2001[i,])
  
  prediction[i]<-predict(rf,fuel2001[i,])  
  all<-unlist(node)+prediction[i]
  ci[i,]<-quantile(all, c(0.025, 0.975))
}
length(which(fuel2001$target2<ci[,1]))+length(which(fuel2001$target2>ci[,2]))


##OOB error for error model, discretize response to discrete variable, 5-fold CV
for (rep in 1:10){
prediction<-c()
ci<-matrix(NA, nrow(fuel2001), 2)
folds<-sample(1:5, nrow(fuel2001), replace=T)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  rf<-randomForest(target2~., fuel2001[-index,-6])
  error<-rf$predicted-fuel2001$target2[-index]
  mod.err<-randomForest(x=fuel2001[-index,1:5], y=error, nodesize=20, keep.inbag = T)
 node<-extract.nodes(mod.err, fuel2001[-index, ], error, fuel2001[index,], "T")
   prediction[index]<-predict(rf,fuel2001[index,])
  for (j in 1:length(index)){
  all<-prediction[index[j]]-unlist(node[[j]])
   ci[index[j],]<-quantile(all, c(0.025, 0.975))}
}
len[rep]=length(which(fuel2001$target2<ci[,1]))+length(which(fuel2001$target2>ci[,2]))  
m[rep]=mean(ci[,2]-ci[,1])  
}


##5-fold CV QRF
for (rep in 1:10){
folds<-sample(1:5, nrow(fuel2001), replace=T)
ci5<-matrix(NA, nrow(fuel2001), 2)
for (i in 1:5){
  index <- which(folds==i,arr.ind=TRUE)
  qrf<-quantregForest(x=fuel2001[-index,1:5], y=fuel2001[-index,7])
  ci5[index,]<-predict(qrf,fuel2001[index,1:5], quantiles=c(0.025, 0.975) )
}
len[rep]=length(which(fuel2001$target2<ci5[,1]))+length(which(fuel2001$target2>ci5[,2]))  
m[rep]=mean(ci5[,2]-ci5[,1])  
}


################################functions#############################################

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
    ci5[index,]<-predict(qrf,dataX[index,], what=c(0.025, 0.975))
  }
  len[rep]=length(which(dataY<ci5[,1]))+length(which(dataY>ci5[,2])) 
  m[rep]=mean(ci5[,2]-ci5[,1])
}
  list(len/length(dataY), m)
}

r<-cv5qrf(fuel2001[,1:5], fuel2001$target2, 100)
lapply(r, mean)
lapply(r, sd)



##5-fold CV error+QRF
cv5err<-function(dataX, dataY, n=10, nodesize=20, oob="T"){
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
##fuel2001
f.r<-cv5err(fuel2001[,1:5], fuel2001$target, nodesize=10, n=100, oob="T")
f.r2<-cv5qrf(fuel2001[,1:5], fuel2001$target, n=100)
f.r3<-cv5err(fuel2001[,1:5], fuel2001$target, nodesize=10, n=100, oob="F")
f.r4<-cv5err(fuel2001[,1:5], fuel2001$target, nodesize=10, n=100, oob="A")
f.r<-cv5err.weight(fuel2001[,1:5], fuel2001$target, nodesize=10, n=100, oob="T")   ##weighted version
lapply(f.r, mean)
lapply(f.r, sd)
lapply(f.r2, mean)
lapply(f.r2, sd)
lapply(f.r3, mean)
lapply(f.r3, sd)
lapply(f.r4, mean)
lapply(f.r4, sd)
boxplot(f.r[[1]], f.r2[[1]],f.r3[[1]],f.r4[[1]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="Fuel2001")
abline(a=0.05, b=0)
boxplot(f.r[[2]], f.r2[[2]],f.r3[[2]],f.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="Fuel2001")

##BigMac2003
b.r<-cv5err(BigMac2003[,-1], BigMac2003[,1], nodesize=10, n=100, oob="T")
b.r2<-cv5qrf(BigMac2003[,-1], BigMac2003[,1], n=100)
b.r3<-cv5err(BigMac2003[,-1], BigMac2003[,1], nodesize=10, n=100, oob="F")
b.r4<-cv5err(BigMac2003[,-1], BigMac2003[,1], nodesize=10, n=100, oob="A")
b.r<-cv5err.weight(BigMac2003[,-1], BigMac2003[,1], nodesize=10, n=100, oob="T")  ##weighted version
logb.r<-cv5err.weight(BigMac2003[,-1], log(BigMac2003[,1]), nodesize=10, n=100, oob="T")  ##log transformation of the response variable
logb.r2<-cv5qrf(BigMac2003[,-1], log(BigMac2003[,1]), n=100)
lapply(b.r, mean)
lapply(b.r, sd)
lapply(b.r2, mean)
lapply(b.r2, sd)
lapply(b.r3, mean)
lapply(b.r3, sd)
lapply(b.r4, mean)
lapply(b.r4, sd)
boxplot(b.r[[1]], b.r2[[1]],b.r3[[1]],b.r4[[1]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="BigMac2003")
abline(a=0.05, b=0)
boxplot(b.r[[2]], b.r2[[2]],b.r3[[2]],b.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="BigMac2003")


##Ozone
o.r<-cv5err(oz[,-4], oz[,4], nodesize=10, n=100, oob="T")
o.r2<-cv5qrf(oz[,-4], oz[,4], n=100)
o.r3<-cv5err(oz[,-4], oz[,4], nodesize=10, n=100, oob="F")
o.r4<-cv5err(oz[,-4], oz[,4], nodesize=10, n=100, oob="A")
o.r<-cv5err.weight(oz[,-4], oz[,4], nodesize=10, n=100, oob="T") ###weighted
lapply(o.r, mean)
lapply(o.r, sd)
lapply(o.r2, mean)
lapply(o.r2, sd)
lapply(o.r3, mean)
lapply(o.r3, sd)
lapply(o.r4, mean)
lapply(o.r4, sd)
boxplot(o.r[[1]], o.r2[[1]],o.r3[[1]],o.r4[[1]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="Ozone")
abline(a=0.05, b=0)
boxplot(o.r[[2]], o.r2[[2]],o.r3[[2]],o.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="Ozone")


##BostonHousing
h.r<-cv5err(BostonHousing[,-14], BostonHousing[,14], nodesize=10, n=100, oob="T")
h.r2<-cv5qrf(BostonHousing[,-14], BostonHousing[,14], n=100)
h.r3<-cv5err(BostonHousing[,-14], BostonHousing[,14], nodesize=10, n=100, oob="F")
h.r4<-cv5err(BostonHousing[,-14], BostonHousing[,14], nodesize=10, n=100, oob="A")
# r3<-cv5err.sample(BostonHousing[,-14], BostonHousing[,14], nodesize=10, n=10, oob="T")
h.r<-cv5err.weight(BostonHousing[,-14], BostonHousing[,14], nodesize=10, n=100, oob="T")  ##weighted by the nodesize
lapply(h.r, mean)
lapply(h.r, sd)
lapply(h.r2, mean)
lapply(h.r2, sd)
lapply(h.r3, mean)
lapply(h.r3, sd)
lapply(h.r4, mean)
lapply(h.r4, sd)
boxplot(h.r[[1]], h.r2[[1]],h.r3[[1]],h.r4[[1]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="BostonHousing")
abline(a=0.05, b=0)
boxplot(h.r[[2]], h.r2[[2]],h.r3[[2]],h.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), main="BostonHousing")



##
c.r<-cv5err(combined1[,unique(unlist(day1.genes))], combined1$dose.n, nodesize=25, n=1, oob="T")
c.r2<-cv5qrf(combined1[,unique(unlist(day1.genes))], combined1$dose.n, n=1)
lapply(c.r, mean)
lapply(c.r, sd)
lapply(c.r2, mean)
lapply(c.r2, sd)


################################################################################
##5-fold CV error+QRF, sampling with replacement based on the size of terminal node, for OOB only
cv5err.sample<-function(dataX, dataY, n=10, nodesize=20, oob="T"){
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
        sample.list=list()
        l=max(sapply(node[[j]], length))
        ##ntree with at least one oob sample
        for (ntree in 1:500){
          if(sapply(node[[j]], length)[ntree]>0){
            sample.list[[ntree]]=sample(unlist(node[[j]][ntree]), l, replace=T)  ##sample each tree to be the same important
          }
        }
        all<-prediction[index[j]]-unlist(sample.list)
        ci[index[j],]<-quantile(all, c(0.025, 0.975))}
    }
    len[rep]=length(which(dataY<ci[,1]))+length(which(dataY>ci[,2])) 
    m[rep]=mean(ci[,2]-ci[,1])  
  }
  list(len/nrow(dataX), m)
}



##5-fold CV error+QRF, weighting based on the size of terminal node, for OOB only
cv5err.weight<-function(dataX, dataY, n=10, nodesize=20, oob="T"){
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
        weight=list()
        ##ntree with at least one oob sample
        for (ntree in 1:500){
          if (sapply(node[[j]], length)[ntree]>0)
          weight[[ntree]]<-rep(1/sapply(node[[j]], length)[ntree], sapply(node[[j]], length)[ntree])
          }
        all<-prediction[index[j]]-unlist(node[[j]])
        ci[index[j],]<-wtd.quantile(all, unlist(weight), c(0.025, 0.975))
        }
    }
    len[rep]=length(which(dataY<ci[,1]))+length(which(dataY>ci[,2])) 
    m[rep]=mean(ci[,2]-ci[,1])  
  }
  list(len/nrow(dataX), m)
}


###########plotting##################
tiff("Figure 2.tif", res=300, height=9, width=7, units="in")
par(mar=c(3, 5, 3, 2))
par(mfrow=c(4,2))
##BostonHousing
boxplot(h.r[[1]]*100, h.r2[[1]]*100, h.r3[[1]]*100,h.r4[[1]]*100, names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylim=c(0,15), ylab="Miscoverage rate (%)")
abline(a=5, b=0)
boxplot(h.r[[2]], h.r2[[2]],h.r3[[2]],h.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylab="Interval length")
mtext("BostonHousing", side = 3, line = -2, outer = TRUE)


##Ozone
boxplot(o.r[[1]]*100, o.r2[[1]]*100,o.r3[[1]]*100,o.r4[[1]]*100, names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylim=c(0,15), ylab="Miscoverage rate (%)")
abline(a=5, b=0)
boxplot(o.r[[2]], o.r2[[2]],o.r3[[2]],o.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylab="Interval length")
mtext("Ozone", side = 3, line = -18, outer = TRUE)


##BigMac
boxplot(b.r[[1]]*100, b.r2[[1]]*100,b.r3[[1]]*100,b.r4[[1]]*100, names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylim=c(0,15), ylab="Miscoverage rate (%)")
abline(a=5, b=0)
boxplot(b.r[[2]], b.r2[[2]],b.r3[[2]],b.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylab="Interval length")
mtext("BigMac2003", side = 3, line = -36, outer = TRUE)

##Fuel
boxplot(f.r[[1]]*100, f.r2[[1]]*100,f.r3[[1]]*100,f.r4[[1]]*100, names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylim=c(0,15), ylab="Miscoverage rate (%)")
abline(5, b=0)
boxplot(f.r[[2]], f.r2[[2]],f.r3[[2]],f.r4[[2]], names=c("RFerr(out)", "QRF", "RFerr(in)", "RFerr(all)"), ylab="Interval length")
mtext("Fuel2001", side = 3, line = -53, outer = TRUE)

dev.off()



  
  
  
  
  

