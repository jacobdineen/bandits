import numpy as np

class UCBStruct:
    def __init__(self, num_arm, epsilon):
        self.d = num_arm # number of arms, k
        self.UserArmMean = np.zeros(self.d) # average reward from action a -- Q_t(a)
        self.UserArmTrials = np.zeros(self.d) # number of times an arm has been pulled -- N_a(t)
        self.epsilon = epsilon
        self.time = 0

    def updateParameters(self, articlePicked_id, click):
        """Parameter Update Step For MAB - UCB1

        Parameters
        ----------
        articlePicked_id : int
            id associated with a particle article
        click : int
            reward signal, implicit
        """
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] \
                                              + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles, sigma):
        """Decision function for MAB - UCB1

        Parameters
        ----------
        pool_articles : list
            article indices - Correspond to individual arms in a k-armed bandit setting

        Returns
        -------
        An arm to pull, given a user and no additional context.
        Uses the UCB1 algorithm to choose an arm which maximizes upper confidence bound
        """
        # If an arm has not been pulled, we need to encourage exploration 
        # Greedily select the first arm in the articles list that hasn't been explored          
        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article

        # print("EpsilonGreedy: greedy")
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            expected_reward = self.UserArmMean[article.id]
            uncertainty = sigma * np.sqrt(2 * np.log(self.time)
                                  / self.UserArmTrials[article.id])
            article_pta = expected_reward + uncertainty
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked


class UCBMultiArmedBandit:
    def __init__(self, num_arm, epsilon, sigma):
        self.users = {}
        self.num_arm = num_arm
        self.epsilon = epsilon
        self.sigma = sigma
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm, self.epsilon)
            
#         print(self.users[userID].decide(pool_articles).id) #arm id
        return self.users[userID].decide(pool_articles, self.sigma)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean