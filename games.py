import numpy as np


class Game:
    """
    Base class for all games.
    """

    def __init__(self, num_iters=1000):
        """
        Initializes a new game with the specified number of iterations.

        Parameters:
        num_iters (int): The number of iterations for the game. Default is 1000.
        """
        self.moves = []
        self.num_iters = num_iters

    def get_payoff(self):
        """
        Returns the payoff matrix for the game.
        """
        pass

    def get_num_iters(self):
        """
        Returns the number of iterations for the game.
        """
        return self.num_iters


class Prisoners_Dilemma(Game):
    """
    Class for the Prisoner's Dilemma game.
    """

    def __init__(self):
        """
        Initializes a new Prisoner's Dilemma game.
        """
        super().__init__()
        self.COOPERATE = 0
        self.DEFECT = 1
        self.NONE = 2
        self.moves = ["COOPERATE", "DEFECT", "None"]

        self.coop = 3.5  # cooperate-cooperate payoff
        self.defect = 1  # defect-defect payoff
        self.temptation = 5  # cooperate-defect (or vice-versa) tempation payoff
        self.sucker = 0  # cooperate-defect (or vice-versa) sucker payoff

        self.payoff = {
            (self.COOPERATE, self.COOPERATE): (self.coop, self.coop),
            (self.COOPERATE, self.DEFECT): (self.sucker, self.temptation),
            (self.DEFECT, self.COOPERATE): (self.temptation, self.sucker),
            (self.DEFECT, self.DEFECT): (self.defect, self.defect),
        }

    def get_payoff(self):
        """
        Returns the payoff matrix for the Prisoner's Dilemma game.
        """
        return self.payoff


class Samaritans_Dilemma(Game):
    """
    Class for the Samaritan's Dilemma game.
    """

    def __init__(self):
        """
        Initializes a new Samaritan's Dilemma game.
        """
        super().__init__()

        self.SOCIAL = 0
        self.ANTI_SOCIAL = 1
        self.NONE = 2
        self.moves = ["SOCIAL", "ANTI_SOCIAL", "None"]

        self.payoff = {
            (self.ANTI_SOCIAL, self.SOCIAL): (2, 2),  # no help, work
            (self.ANTI_SOCIAL, self.ANTI_SOCIAL): (1, 1),  # no help, no work
            (self.SOCIAL, self.SOCIAL): (4, 3),  # help, work
            (self.SOCIAL, self.ANTI_SOCIAL): (3, 4),  # help, no work
        }

    def get_payoff(self):
        """
        Returns the payoff matrix for the Samaritan's Dilemma game.
        """
        return self.payoff


class Stag_Hunt(Game):
    """
    Class for the Stag Hunt game.
    """

    def __init__(self):
        """
        Initializes a new Stag Hunt game.
        """
        super().__init__()

        self.STAG = 0
        self.HARE = 1
        self.NONE = 2
        self.moves = ["STAG", "HARE", "None"]

        self.payoff = {
            (self.STAG, self.STAG): (4, 4),
            (self.STAG, self.HARE): (1, 3),
            (self.HARE, self.STAG): (3, 1),
            (self.HARE, self.HARE): (2, 2),
        }

    def get_payoff(self):
        """
        Returns the payoff matrix for the Stag Hunt game.
        """
        return self.payoff


class Chicken(Game):
    """
    Class for the Chicken game.
    """

    def __init__(self):
        """
        Initializes a new Chicken game.
        """
        super().__init__()

        self.SWERVE = 0
        self.STRAIGHT = 1
        self.NONE = 2
        self.moves = ["SWERVE", "STRAIGHT", "None"]

        self.payoff = {
            (self.SWERVE, self.SWERVE): (0, 0),
            (self.STRAIGHT, self.SWERVE): (1, -1),
            (self.SWERVE, self.STRAIGHT): (-1, 1),
            (self.STRAIGHT, self.STRAIGHT): (-1000, -1000),
        }

    def get_payoff(self):
        """
        Returns the payoff matrix for the Chicken game.
        """
        return self.payoff


def get_game_class(name="prisoners_dilemma"):
    """
    Returns the game class corresponding to the specified name.
    """
    if name.lower() == "prisoners_dilemma" or name.lower() == "pd":
        return Prisoners_Dilemma
    elif name.lower() == "samaritans_dilemma" or name.lower() == "sd":
        return Samaritans_Dilemma
    elif name.lower() == "stag_hunt" or name.lower() == "sh":
        return Stag_Hunt
    elif name.lower() == "chicken" or name.lower() == "ch":
        return Chicken
    else:
        raise ValueError("Invalid game name: %s" % name)


7
