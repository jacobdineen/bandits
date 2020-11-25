import numpy as np

class UCBLinearBanditStruct:
    def __init__(self, featureDimension, lambda_, epsilon, alpha):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d) #d-dim identity matrix
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.alpha = alpha
        self.UserArmTrials = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_id, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector) 
        self.b += articlePicked_FeatureVector * click # r_ai * x_ai
        self.AInv = np.linalg.inv(self.A) #
        self.UserTheta = np.dot(self.AInv, self.b) #parameter of interest - used for arm reward estimation
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article
        # print("EpsilonGreedy: greedy")
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = np.dot(self.UserTheta, article.featureVector)
            uncertainty = self.alpha * np.sqrt(
                            np.dot(np.dot(article.featureVector.T, self.AInv), article.featureVector.T))
            ucb_value = article_pta + uncertainty       
            # pick article with highest Prob
            if maxPTA < ucb_value:
                articlePicked = article
                maxPTA = ucb_value

        return articlePicked

class UCBLinearBandit:
    def __init__(self, dimension, lambda_, epsilon, alpha):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.alpha = alpha
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBLinearBanditStruct(self.dimension, self.lambda_, self.epsilon, self.alpha)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


