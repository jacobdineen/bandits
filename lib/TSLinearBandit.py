import numpy as np

class LinThompsonSamplingStruct:
    def __init__(self, featureDimension, lambda_, epsilon, sigma):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d) #d-dim identity matrix
        self.lambda_ = lambda_
        self.epsilon = 0.5
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.random.multivariate_normal(np.zeros(self.d), self.A)
        self.sigma = sigma
        self.UserArmTrials = np.zeros(self.d)
        self.time = 0
        self.R = 0.01
        self.delta = 0.5

    def updateParameters(self, articlePicked_id, articlePicked_FeatureVector, click):
#         print(self.UserTheta)
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector) / (self.sigma ** 2)
        self.AInv = np.linalg.inv(self.A)

        self.b += (articlePicked_FeatureVector * click)  / (self.sigma ** 2)

        self.v = self.R * np.sqrt(24 / self.epsilon
                             * self.d
                             * np.log(1 / self.delta))
                             
        self.UserTheta = np.random.multivariate_normal(
                                                        self.AInv.dot(self.b), 
                                                        self.v**2 * self.AInv
                                                        ) 
        # print(self.UserTheta)
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

            article_pta = article.featureVector.dot(self.UserTheta)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class ThompsonSamplingLinearBandit:
    def __init__(self, dimension, lambda_, epsilon, sigma):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.sigma = sigma
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinThompsonSamplingStruct(self.dimension, self.lambda_, self.epsilon, self.sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


