library(rpart)
library(ggplot2)
library(lattice)
library(rattle)
library(gplots)
library(ROCR)
library(randomForest)

require(mice)
require(dummies)  # 轉換虛擬變數的套件
require(rpart)
require(rpart.plot) 
require(caret)
require(e1071)
require(neuralnet) # for neuralnet(), nn model
require(nnet)      # for class.ind()

# 讀取 csv
promotion <- read.csv("D:/DM/final/promotion/promotion.csv")
# 補NA
promotion$previous_year_rating<-as.numeric(as.character(promotion$previous_year_rating))
promotion$previous_year_rating[is.na(promotion$previous_year_rating)]<-0
# 確認column name 合理
colnames(promotion) <- make.names(colnames(promotion))

# 根據下面的重要變數篩選出新的資料集
promotion0 <- promotion[,-1]
promotion1 <- promotion0[,-2]
promotion2 <- promotion1[,-3]
promotion_importantArg <- promotion2[,-3]
str(promotion_importantArg)

# dummy
# alldummy_data <- dummy.data.frame(promotion[,-10])
alldummy_data <- dummy.data.frame(promotion)

# 確認column name 合理 
colnames(alldummy_data) <- make.names(colnames(alldummy_data))

View(alldummy_data)
summary(alldummy_data)

# alldummy_data<-cbind(alldummy_data , is_promoted=promotion[,10])
str(alldummy_data)
# ==============視覺化====================
# 盒鬚
boxplot(formula = avg_training_score ~ no_of_trainings, # Y ~ X (代表X和Y軸要放的數值) 
        data = promotion,       # 資料
        xlab = "no_of_training",# X軸名稱
        ylab = "avg_training_score",    # Y軸名稱
        col ="gray")$out

# is_promoted 長條圖
hist(x=promotion$is_promoted, 
     main="Histogram of is_promoted",         # 圖片的名稱
     xlab="is_promoted",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱

# id 長條圖
hist(x=promotion$employee_id, 
     main="Histogram of employee_id",         # 圖片的名稱
     xlab="employee_id",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱

# no_of_training  長條圖
hist(x=promotion$employee_id, 
     main="Histogram of employee_id",         # 圖片的名稱
     xlab="employee_id",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱

# age 長條圖
hist(x=promotion$age, 
     main="Histogram of age ",         # 圖片的名稱
     xlab="age ",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱
# previous_year_rating 長條圖
hist(x=promotion$previous_year_rating, 
     main="Histogram of previous_year_rating",         # 圖片的名稱
     xlab="previous_year_rating",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱
# length_of_service 長條圖
hist(x=promotion$length_of_service, 
     main="Histogram of length_of_service",         # 圖片的名稱
     xlab="length_of_service",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱
# KPIs_met..80. 長條圖
hist(x=promotion$KPIs_met..80., 
     main="Histogram of KPIs_met..80.",         # 圖片的名稱
     xlab="KPIs_met..80.",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱
# awards_won. 長條圖
hist(x=promotion$awards_won., 
     main="Histogram of awards_won.",         # 圖片的名稱
     xlab="awards_won.",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱

# avg_training_score 長條圖
hist(x=promotion$avg_training_score, 
     main="Histogram of avg_training_score",         # 圖片的名稱
     xlab="avg_training_score",                      # X軸的名稱
     ylab="Number")                  # Y軸的名稱# 升遷和性別
Probs_1 <- as.data.frame(prop.table(table(promotion$is_promoted, promotion$gender), 1))
ggplot(Probs_1, aes(x = Var2, y = Freq, fill = Var1)) + geom_bar(stat = "identity", position = "fill", color = "black") + theme_bw() +
  scale_fill_brewer(palette = "Dark2") + labs( x = "gender", y = "promotion", fill = "promotion", title = "Relationship between promotion and gender")
# 升遷和獎項
Probs_2 <- as.data.frame(prop.table(table(promotion$awards_won., promotion$is_promoted), 1))
ggplot(Probs_2, aes(x = Var1, y = Freq, fill = Var2)) + geom_bar(stat = "identity", position = "fill", color = "black") + theme_bw() +
  scale_fill_brewer(palette = "Dark2") + labs( x = "awards won", y = "promotion", fill = "promotion", title = "Relationship between promotion and winning awards")
# ==============================================================================
# ======================================分割訓練與測試資料集====================================
write.csv(promotion,file="D:/DM/final/promotion/promotion_complete.csv",row.names = FALSE)


set.seed(77)
Index <- createDataPartition(promotion_importantArg$is_promoted, p = 0.75, list = F)
Train <- promotion[Index,]
Test <- promotion[-Index,]


set.seed(87)
Index <- createDataPartition(alldummy_data$is_promoted, p = 0.75, list = F)
Train_dummy <- alldummy_data[Index,]
Test_dummy <- alldummy_data[-Index,]

set.seed(97)
Index <- createDataPartition(promotion_importantArg$is_promoted, p = 0.75, list = F)
Train_new <- promotion_importantArg[Index,]
Test_new <- promotion_importantArg[-Index,]


#====================================重要變數===================================


null = lm(is_promoted ~ 1, data = alldummy_data)  
full = lm(is_promoted ~ . , data = alldummy_data)
# 漸進
forward.lm = step(null, 
                  # 從空模型開始，一個一個丟變數，
                  scope=list(lower=null, upper=full), 
                  direction="forward")
# 倒退
backward.lm = step(full, 
                   scope = list(upper=full), 
                   direction="backward")
summary(forward.lm)
summary(backward.lm)

#===============================================================================
# ================================== 羅吉斯 =====================================
Train$is_promoted <- factor(Train$is_promoted, levels= c(0,1), labels = c(0,1))
Test$is_promoted <- factor(Test$is_promoted, levels= c(0,1), labels = c(0,1))

model_1 <- glm(is_promoted~., data= Train, family= "binomial")# 全部變數
summary(model_1)

# 只留重要變數
model_2 <- glm(is_promoted~. - employee_id- region- gender- recruitment_channel,
               data= Train,
               family= "binomial")
summary(model_2)


predict_result <- predict(model_2, newdata = Test, type = "response")# 預測
pr_class <- factor(ifelse(predict_result > 0.5, "1", "0"))
levels(Test$is_promoted)
confusionMatrix(pr_class, Test$is_promoted, positive = "1")# 混淆矩陣

# ROC 曲線
p_test <- prediction(predict_result, Test$is_promoted)
performance <- performance(p_test, "tpr", "fpr")
plot(performance, colorize = TRUE)
performance(p_test, "auc")@y.values
#===============================================================================

#==================================決策樹==========================================
Train_new$is_promoted <- factor(Train_new$is_promoted, levels= c(0,1), labels = c(0,1))
Test_new$is_promoted <- factor(Test_new$is_promoted, levels= c(0,1), labels = c(0,1))

cart.model<- rpart(is_promoted ~ . , 
                   data=Train_new,
                   method = "class")

prp(cart.model,         # 模型
    faclen=1,           # 呈現的變數不要縮寫
    shadow.col="gray",  # 最下面的節點塗上陰影
    extra = 1
) 
result <- predict(cart.model, newdata = Test_new, type = "class")
cm <- table(Test_new$is_promoted, result, dnn = c("實際", "預測"))
cm

#正確率
#計算猜升值正確率
cm[2,2] / sum(cm[, 2])

#計算猜不升值正確率
cm[1] / sum(cm[, 1])

#整體準確率(取出對角/總數)
accuracy <- sum(diag(cm)) / sum(cm)
accuracy # 0.9364326

prunetree_cart.model <- prune(cart.model, cp = cart.model$cptable[which.min(cart.model$cptable[,"xerror"]),"CP"]) 

prunetree_pred <- predict(prunetree_cart.model, newdata=Test_new, type="class")

# 用table看預測的情況
cm2<-table(Test_new$is_promoted, prunetree_pred, dnn = c("實際", "預測"))

printcp(prunetree_cart.model)
plotcp(prunetree_cart.model)
prp(prunetree_cart.model,         # 模型
    faclen=1,           # 呈現的變數不要縮寫
    shadow.col="gray",  # 最下面的節點塗上陰影
    extra = 1
) 

#正確率
#計算猜升值正確率
cm2[2,2] / sum(cm2[, 2])

#計算猜不升值正確率
cm2[1] / sum(cm2[, 1])

#整體準確率(取出對角/總數)
accuracy2 <- sum(diag(cm2)) / sum(cm2)
accuracy2 # 0.9364326
#==================================================================================================
#=================================================k fold ===============================================
promotion_importantArg$is_promoted <- factor(promotion_importantArg$is_promoted, levels= c(0,1), labels = c(0,1))
train_control <- trainControl(method="cv", number=10)
train_control.model <- train(is_promoted ~ . ,
                             data=promotion_importantArg,
                             method="rpart",
                             trControl=train_control)
train_control.model
# Acc =  0.9229857 
#==============================================================================================
#=================================隨機森林======================================================
set.seed(123)
# model
m1 <- randomForest(
  formula = is_promoted ~ .,
  data    = Train_new
)
m1 # Acc = 0.9399

plot(m1)

prediction <- predict(m1, subset(Test_new, select = - is_promoted))#預測

cmRF <- table(Test_new$is_promoted, prediction, dnn = c("實際", "預測"))
cmRF
#正確率
#計算猜升值正確率
cmRF[2,2] / sum(cmRF[, 2])

#計算猜不升值正確率
cmRF[1] / sum(cmRF[, 1])

#整體準確率(取出對角/總數)
accuracyRF <- sum(diag(cmRF)) / sum(cmRF)
accuracyRF # 0.9434389
#===================================================================================================
#============================================NN=====================================================
promotion_importantArg$is_promoted<-as.integer(as.character(promotion_importantArg$is_promoted))
promotion_importantArg$previous_year_rating<-as.integer(as.character(promotion_importantArg$previous_year_rating))
alldummy_important <- dummy.data.frame(promotion_importantArg)
set.seed(107)
Index <- createDataPartition(alldummy_important$is_promoted, p = 0.75, list = F)
Train_dummy_imp <- alldummy_important[Index,]
Test_dummy_imp <- alldummy_important[-Index,]

# 確認column name 合理
colnames(Train_dummy_imp) <- make.names(colnames(Train_dummy_imp))
# 確認column name 合理
colnames(Test_dummy_imp) <- make.names(colnames(Test_dummy_imp))

Train_dummy_imp <- scale(Train_dummy_imp , center=T,scale=T)
Test_dummy_imp <- scale(Test_dummy_imp, center=T,scale=T)


str(Train_dummy_imp)

bpn <- neuralnet(formula = is_promoted ~ ., 
                 data = Train_dummy_imp,
                 hidden = c(4,3),     # 第一隱藏層1個node，第二隱藏層2個nodes
                 learningrate = 0.005, # learning rate
                 threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                 stepmax = 5e5,        # 最大的ieration數 = 500000(5*10^5)
                 linear.output = F
)
plot(bpn)
