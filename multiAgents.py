from Agents import Agent
from Game import GameState
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState : GameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        return self.minmax(gameState, 0, self.depth)[1] #position decided by minmax 

    def minmax(self, gameState: GameState, agentIndex, depth):

        if gameState.isGameFinished() or depth == 0:
            return (self.evaluationFunction(gameState), None)
        
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum

        if agentIndex == agentsNum-1:
            depth -= 1

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth) #return a tuple (highestScore , chosenPos)
        else:
            return self.minimizer(gameState, agentIndex, depth) #return a tuple (lowestScore , chosenPos)
    
    def maximizer(self, gameState: GameState,agentIndex, depth ):    
        positions_after_actions = [] #array of tupples (score, pos after an action)
        for pos in gameState.getLegalActions(agentIndex):
            positions_after_actions.append( (self.minmax(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth)[0], pos))
        return max(positions_after_actions) #returns the tuple with the highest score

    def minimizer(self, gameState: GameState, agentIndex, depth):
        positions_after_actions = [] #array of tupples (score, pos after an action)
        for pos in gameState.getLegalActions(agentIndex):
            positions_after_actions.append( (self.minmax(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth)[0], pos) )    
        return min(positions_after_actions) #returns the tuple with the lowest score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        return self.minmax_ab(gameState, 0, self.depth)[1]
    
    def minmax_ab(self, gameState: GameState, agentIndex, depth, alpha= -999999, beta = 999999):
        #alpha -> maximizer's best option
        #beta -> minimizer's best option

        if gameState.isGameFinished() or depth == 0:
            return (self.evaluationFunction(gameState), None)
        
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum

        if agentIndex == agentsNum-1:
            depth -= 1

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth, alpha, beta) 
        else:
            return self.minimizer(gameState, agentIndex, depth, alpha, beta) 
    
    def maximizer(self, gameState: GameState,agentIndex, depth, alpha, beta):    
        positions_after_actions = [] 

        for pos in gameState.getLegalActions(agentIndex):
            v = self.minmax_ab(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth, alpha , beta)[0]
            positions_after_actions.append( (v , pos) )

            if v > beta:
                return (v, pos)
            
            alpha = max(alpha, v)

        return max(positions_after_actions) 

    def minimizer(self, gameState: GameState, agentIndex, depth, alpha, beta):
        positions_after_actions = [] 

        for pos in gameState.getLegalActions(agentIndex):
            v = self.minmax_ab(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth, alpha, beta)[0]
            positions_after_actions.append( (v, pos) )

            if v < alpha:
                return (v, pos)
            
            beta = min(beta, v)

        return min(positions_after_actions)
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, 0, self.depth)[1]
    
    def expectimax(self, gameState: GameState, agentIndex, depth):

        if gameState.isGameFinished() or depth == 0:
            return (self.evaluationFunction(gameState), None)
        
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum

        if agentIndex == agentsNum-1:
            depth -= 1

        if agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth)
        else:
            return self.averageValue(gameState, agentIndex, depth) 
    
    def maximizer(self, gameState: GameState,agentIndex, depth ):    
        positions_after_actions = [] 
        for pos in gameState.getLegalActions(agentIndex):
            positions_after_actions.append( (self.expectimax(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth)[0], pos))
        return max(positions_after_actions) 

    def averageValue(self, gameState: GameState, agentIndex, depth):
        positions_after_actions = [] 
        sum = 0
        for pos in gameState.getLegalActions(agentIndex):
            v = self.expectimax(gameState.generateSuccessor(agentIndex, pos), agentIndex + 1, depth)[0]
            positions_after_actions.append( (v, pos) )
            sum += v
        
        return ( sum/len(positions_after_actions) , None)    

def betterEvaluationFunction(currentGameState : GameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list    positions of each player's pieces
    gameState.getCorners() -> 4-tuple     Returns a 4-tuple each with the index of the player who occupies that corner. Returns -1 if it is free
    gameState.getScore() -> list          
    gameState.getScore(index) -> int

    """
    agentsNum = currentGameState.getNumAgents()
    score = currentGameState.getScore()
    #Each heuristic scales its return value from -100 to 100

    # parity
    parity_heuristic = parity(currentGameState , agentsNum)
    # corners

    # mobility
    actual_mob_heuristic , potential_mob_heuristic = mobility(currentGameState , agentsNum)
    # stability

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

def parity(currentGameState : GameState , agentsNum):
    if agentsNum == 2:
        parity_heuristic = 100 * (currentGameState.getScore(0) - currentGameState.getScore(1)) / (currentGameState.getScore(0) + currentGameState.getScore(1))

    elif agentsNum == 4:
        parity_heuristic = 100 * (currentGameState.getScore(0) - currentGameState.getScore(1) - currentGameState.getScore(2) - currentGameState.getScore(3)) / (currentGameState.getScore(0) + currentGameState.getScore(1) + currentGameState.getScore(2) + currentGameState.getScore(3))

    return parity_heuristic

def mobility(currentGameState : GameState , agentsNum) :
    #Actual mobility is the number of next moves a player has, given the current state of the game.
    #Actual mobility is calculated by examining the board and counting the number of legal moves for the player
    actualMaxMob = len(currentGameState.getLegalActions(0))
    actualMinMob = len(currentGameState.getLegalActions(1))

    if agentsNum == 2:
        if (actualMaxMob + actualMinMob) != 0 :
            actual_mob_heuristic = 100 * (actualMaxMob - actualMinMob) / (actualMaxMob + actualMinMob)
        else :
            actual_mob_heuristic = 0

    elif agentsNum == 4: 
        actualMin2Mob = len(currentGameState.getLegalActions(2))
        actualMin3Mob = len(currentGameState.getLegalActions(3))
        if (actualMaxMob + actualMinMob + actualMin2Mob + actualMin3Mob) != 0 :
            actual_mob_heuristic = 100 * (actualMaxMob - actualMinMob  - actualMin2Mob - actualMin3Mob) / (actualMaxMob + actualMinMob  + actualMin2Mob + actualMin3Mob)
        else :
            actual_mob_heuristic = 0
    #Potential mobility is the number of possible moves the player might have over the next few moves.
    #Potential mobility is calculated by counting the number of empty spaces next to atleast one of the opponent’s coin
    max_positions = currentGameState.getPieces(0)
    potentialMaxMob = potential_mobility(max_positions) 
    min_positions = currentGameState.getPieces(1)
    potentialMinMob = potential_mobility(min_positions)

    if agentsNum == 2 :
        if (potentialMaxMob + potentialMinMob) != 0 :
            potential_mob_heuristic = 100 * (potentialMaxMob - potentialMinMob) / (potentialMaxMob + potentialMinMob)
        else :
            potential_mob_heuristic = 0
    elif agentsNum == 4:
       min2_positions = currentGameState.getPieces(2)
       potentialMin2Mob = potential_mobility(min2_positions)        

       min3_positions = currentGameState.getPieces(3)
       potentialMin3Mob = potential_mobility(min3_positions)   

       if (potentialMaxMob + potentialMinMob) != 0 :
            potential_mob_heuristic = 100 * (potentialMaxMob - potentialMinMob - potentialMin2Mob - potentialMin3Mob) / (potentialMaxMob + potentialMinMob + potentialMin2Mob + potentialMin3Mob)
       else :
            potential_mob_heuristic = 0
    
    return actual_mob_heuristic , potential_mob_heuristic

def potential_mobility(currentGameState : GameState , positions) :
    potentialMob = 0 
    for pos in positions:
        for dir in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            if currentGameState.nextUnoccupiedPos(0, pos, dir) is not None:
                potentialMob += 1 

    return potentialMob            
