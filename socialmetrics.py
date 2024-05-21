
class SocialMetrics:
    def __init__(self):
        self.observations = []
        self.rewards = []

        self.utilitarian_eff = None
        self.equality = None
        self.sustainability = None
        self.peace = None
    def add_step(self, obs, rws):
        self.observations.append(obs)
        self.rewards.append(rws)

    def compute_metrics(self):
        self.compute_utilitarian_eff()
        self.compute_equality()
        self.compute_sustainability()
        self.compute_peace()
    def compute_utilitarian_eff(self):
        eff = 0

        self.utilitarian_eff = eff

    def compute_equality(self):
        eq = 0

        self.equality = eq

    def compute_sustainability(self):
        sus = 0

        self.sustainability = sus

    def compute_peace(self):
        p = 0

        self.peace = p
