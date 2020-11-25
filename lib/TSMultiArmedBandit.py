import numpy as np

class ThompsonSamplingStruct:
    def __init__(self, num_arm, epsilon, sigma):
        self.d = num_arm
        self.sigma = sigma
        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)
        self.R = 0.01
        self.delta = 0.5
        self.epsilon = 0.5
        self.time = 0
        

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
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
        v = self.R * np.sqrt(24 / self.epsilon
                        * self.d
                        * np.log(1 / self.delta))
        for article in pool_articles:
            article_pta = np.random.normal(self.UserArmMean[article.id], self.sigma)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked
        

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, num_arm, epsilon, sigma):
        self.users = {}
        self.num_arm = num_arm
        self.epsilon = epsilon
        self.sigma = sigma
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = ThompsonSamplingStruct(self.num_arm, self.epsilon, self.sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean

