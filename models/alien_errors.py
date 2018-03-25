class Error(Exception):
    pass


class ProbabilityError(Error):
    def __init__(self, probs):
        self.probs = probs