import numpy as np
from scipy.stats import bernoulli

class PHEStruct:
    def __init__(self, num_arm, epsilon, alpha,sigma):
        self.d = num_arm
        self.epsilon = epsilon
        self.alpha = alpha
        self.sigma = sigma
        self.UserArmMean = np.zeros(self.d)
        self.PerturbedUserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)

        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.PerturbedUserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id] + \
                                                       np.random.binomial(self.UserArmTrials[articlePicked_id]*self.alpha,0.5)) \
                                                       / (1+ self.alpha)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article

        # print("EpsilonGreedy: greedy")
        maxPTA = float('-inf')
        articlePicked = None
        
        for article in pool_articles:
            article_pta = self.PerturbedUserArmMean[article.id]
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked


class PerturbedHistoryExplorationMultiArmedBandit:
    def __init__(self, num_arm, epsilon, alpha, sigma):
        self.users = {}
        self.num_arm = num_arm
        self.epsilon = epsilon
        self.alpha = alpha
        self.sigma = sigma
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm, self.epsilon, self.alpha, self.sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean

