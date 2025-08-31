# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] ==
                        bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total_score = 0
        food_bonus = 10
        ghost_bonus = -99999
        scared_ghost_bonus = 10
        stop_bonus = -50

        normal_score = successorGameState.getScore()
        total_score += normal_score

        food_list = newFood.asList()
        # food is the (x, y) coordinate tuple indicating food at (x, y)
        if food_list:
            min_dist = min(util.manhattanDistance(food, newPos) for food in 
                           food_list)
            total_score += food_bonus / (min_dist + 1)
        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.getPosition()
            if ghost_pos == newPos:
                if ghost_state.scaredTimer == 0:
                    total_score += ghost_bonus
                else:
                    total_score += scared_ghost_bonus
        return total_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Follow the gragh in note05.pdf
        def value(state, depth, agentIndex):
            if depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            elif agentIndex > 0:
                return min_value(state, depth, agentIndex)
            else:
                return max_value(state, depth, agentIndex)
            
        def min_value(state, depth, agentIndex):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, value(successor, depth - 1, nextAgent))
            return v
        
        def max_value(state, depth, agentIndex):
            v = -float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(successor, depth - 1, nextAgent))
            return v
        
        PacmanIndex = 0
        PacmanActions = gameState.getLegalActions(0)
        desiredAction = None
        desiredValue = -float('inf')
        totalDepth = self.depth * gameState.getNumAgents() - 1
        for action in PacmanActions:
            successor = gameState.generateSuccessor(PacmanIndex, action)
            cur_value = value(successor, totalDepth, PacmanIndex + 1)
            if cur_value > desiredValue:
                desiredValue = cur_value
                desiredAction = action
        return desiredAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Similar to MinimaxAgent
        def value(state, depth, agentIndex, alpha, beta):
            if depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            elif agentIndex > 0:
                return min_value(state, depth, agentIndex, alpha, beta)
            else:
                return max_value(state, depth, agentIndex, alpha, beta)
            
        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, value(successor, depth - 1, nextAgent, alpha, beta))
                if v < alpha: return v
                beta = min(beta, v)
            return v
        
        def max_value(state, depth, agentIndex, alpha, beta):
            v = -float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(successor, depth - 1, nextAgent, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v
        
        PacmanIndex = 0
        PacmanActions = gameState.getLegalActions(0)
        desiredAction = None
        desiredValue = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        totalDepth = self.depth * gameState.getNumAgents() - 1
        for action in PacmanActions:
            successor = gameState.generateSuccessor(PacmanIndex, action)
            cur_value = value(successor, totalDepth, PacmanIndex + 1, alpha, 
                              beta)
            if cur_value > desiredValue:
                desiredValue = cur_value
                desiredAction = action
            alpha = max(alpha, desiredValue)
        return desiredAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # In this case, we no longer need the min_value() but average_value()
        def value(state, depth, agentIndex):
            if depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            elif agentIndex > 0:
                return average_value(state, depth, agentIndex)
            else:
                return max_value(state, depth, agentIndex)
            
        def average_value(state, depth, agentIndex):
            legalActions = state.getLegalActions(agentIndex)
            divisor = len(legalActions)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            sum_value = 0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                sum_value += value(successor, depth - 1, nextAgent)
            return sum_value / divisor
        
        def max_value(state, depth, agentIndex):
            v = -float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(successor, depth - 1, nextAgent))
            return v
        
        PacmanIndex = 0
        PacmanActions = gameState.getLegalActions(0)
        desiredAction = None
        desiredValue = -float('inf')
        totalDepth = self.depth * gameState.getNumAgents() - 1
        for action in PacmanActions:
            successor = gameState.generateSuccessor(PacmanIndex, action)
            cur_value = value(successor, totalDepth, PacmanIndex + 1)
            if cur_value > desiredValue:
                desiredValue = cur_value
                desiredAction = action
        return desiredAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    For food, we want pacman to be as close to food as possible to eat all the
    dots, using manhattanDistance.
    For ghosts, as long as pacman never lies in the same place as any normal 
    ghosts, any distance between them is acceptable. Moreover, if a ghost is
    scared, pacman is encouraged to eat that ghost.
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    total_score = 0
    food_bonus = 10
    ghost_bonus = -99999
    scared_ghost_bonus = 50

    normal_score = currentGameState.getScore()
    total_score += normal_score

    food_list = curFood.asList()
    # food is the (x, y) coordinate tuple indicating food at (x, y)
    if food_list:
        min_dist = min(util.manhattanDistance(food, curPos) for food in 
                       food_list)
        total_score += food_bonus / (min_dist + 1)
    for ghost_state in curGhostStates:
        ghost_pos = ghost_state.getPosition()
        if ghost_pos == curPos:
            if ghost_state.scaredTimer == 0:
                total_score += ghost_bonus
            else:
                total_score += scared_ghost_bonus
    return total_score

# Abbreviation
better = betterEvaluationFunction
