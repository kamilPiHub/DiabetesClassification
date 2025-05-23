setwd("C:\\Users\\kvmil\\Desktop\\projekt2mad")
framingham<-read.table("dane.txt",header=T,sep=",")
head(framingham)
dim(framingham)
names(framingham)
summary(framingham)


install.packages("ggplot2")
library(ggplot2)

ggplot(framingham) +
  geom_point(aes(x=totChol, y=sysBP, color=as.factor(TenYearCHD)))

ggplot(framingham) +
  geom_point(aes(x=BMI, y=heartRate, color=as.factor(TenYearCHD)))

table(framingham$male)


set.seed(23)
split=sample(1:nrow(framingham), round(0.7*nrow(framingham)))
train=framingham[split,]
test=framingham[-split,]

mod1<-glm(TenYearCHD~totChol+sysBP, data = train, family = "binomial")
summary(mod1)

exp(0.021877) # OR dla sysBP
exp(0.002002) # OR dla totChol

mod2<-glm(TenYearCHD~BMI+heartRate, data = train, family = "binomial")
summary(mod2)

mod3<-glm(TenYearCHD~BMI+sysBP, data = train, family = "binomial")
summary(mod3)

ggplot(framingham) +
  geom_point(aes(x=BMI, y=sysBP, color=as.factor(TenYearCHD)))



full_model=glm(TenYearCHD~., data = train, family = "binomial")
summary(full_model)

exp(0.439804) # OR dla prevalentHyp
table(framingham$prevalentHyp)
prop.table(table(framingham$prevalentHyp))


table(full_model$y, full_model$fitted.values>0.5)

(2165+31)/(2165+31+370+8)
#[1] 0.8531469 accuracy

predictTest=predict(full_model, newdata = test, type = "response")
table(test$TenYearCHD, predictTest>0.5)

(923+12)/(144+5+923+12)
## [1] 0.8625461
#Z modelu pelnego 
czulosc
12/(12+144)
[1] 0.07692308
specyficznosc
923/(923+5)
[1] 0.9946121
# uswuwanie brakow danych
train1<-na.omit(train)
full_model1=glm(TenYearCHD~., data = train1, family = "binomial")
summary(full_model1)

step(full_model1)

best_model<-glm(formula = TenYearCHD ~ male + age + cigsPerDay + prevalentHyp + 
                  totChol + sysBP + diaBP + glucose, family = "binomial", data = train1)
summary(best_model)
table(best_model$y, best_model$fitted.values>0.5)

(2153+26)/(2153+9+359+26)
[1] 0.8555163
(2168+29)/(2168+29+372+5)
0.8535354

predictTest=predict(best_model, newdata = test, type = "response")
table(test$TenYearCHD, predictTest>0.5)
(967+13)/(967+13+169+5)
[1] 0.8492201
(975+10)/(975+10+154+4)
0.8617673
###
czulosc
10/(10+154)
[1] 0.06097561
specyficznosc
975/(975+4)
[1] 0.9959142

exp(0.463056) # OR dla male
exp(0.453646) # OR dla prevalentHyp