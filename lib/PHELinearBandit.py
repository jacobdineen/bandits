import numpy as np
from scipy.stats import bernoulli

class LinPHEStruct:
    def __init__(self, featureDimension, lambda_, epsilon, alpha, sigma):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.sigma = sigma
        self.alpha = alpha
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_id, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector) * (self.alpha + 1)
        self.b += (click + np.random.binomial(self.UserArmTrials[articlePicked_id]*self.alpha,0.5)) * articlePicked_FeatureVector
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
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
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
        
class PerturbedHistoryExplorationLinearBandit:
    def __init__(self, dimension, lambda_, epsilon, alpha, sigma):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.alpha = alpha
        self.sigma = sigma
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinPHEStruct(self.dimension, self.lambda_, self.epsilon, self.alpha, self.sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


