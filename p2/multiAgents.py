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

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState):
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
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
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

		if successorGameState.isLose():
		  return -float("inf")

		if successorGameState.isWin():
		  return float("inf")


		# if no food taken, then move towards food
		if successorGameState.getNumFood() == currentGameState.getNumFood():
			return -min([manhattanDistance(newPos, food) for food in newFood.asList()])
		

		return successorGameState.getScore()


		
		# capsules= successorGameState.getCapsules()
		# capsulesLeft = len(capsules)

		# minScaredDistance = minNonScaredDistance = minCapsuleDist = minFoodDist = 0

		# if foodLeft > 0:
		# 	minFoodDist = min([manhattanDistance(newPos, food) for food in newFood.asList()])

		# if capsulesLeft > 0:
		#   minCapsuleDist = min([manhattanDistance(newPos, capsule) for capsule in capsules])

		
		# scaredGhosts = [ghost for ghost in newGhostStates if ghost.scaredTimer]
		# nonScaredGhosts = [ghost for ghost in newGhostStates if not ghost.scaredTimer]

		# if len(scaredGhosts) > 0:
		#   minScaredDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in scaredGhosts])

		# if len(nonScaredGhosts) > 0:
		#   minNonScaredDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in nonScaredGhosts])

		

		# distances = 0
		# for ghost in nonScaredGhosts:
		# 	distances += max(manhattanDistance(newPos, ghost.getPosition()),5)**2

		# if minNonScaredDistance < 5:
		#   minNonScaredDistance = 5

		# minNonScaredDistance = 1/minNonScaredDistance**2
		# if minNonScaredDistance == 0:
		#     minNonScaredDistance = float("inf")

		# numGhosts = len(newGhostStates)

		# return score -4*foodLeft +  - 10 * capsulesLeft - 1.5*minFoodDist -5*minScaredDistance + minNonScaredDistance



def scoreEvaluationFunction(currentGameState):
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

	def getAction(self, gameState):
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
		n = gameState.getNumAgents()
		treeDepth = self.depth * n

		def minimaxHelper(depth, agent, state):

			if depth == treeDepth or state.isWin() or state.isLose():
				return self.evaluationFunction(state)

			if agent == 0:
				evalFunction = max
			else:
				evalFunction = min

			newStates = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
			return evalFunction([minimaxHelper(depth+1, (agent+1) % n, nextState) for nextState in newStates])

		actions = gameState.getLegalActions(0)
		return max(actions, key = lambda x: minimaxHelper(1,1,gameState.generateSuccessor(0, x)))
		

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		n = gameState.getNumAgents()
		treeDepth = self.depth * n
		def minimaxHelper(depth, agent, state, alpha, beta):

			if depth == treeDepth or state.isWin() or state.isLose():
				return self.evaluationFunction(state)

			if agent == 0:
				value = float("-inf")
			else:
				value = float("inf")

			for action in state.getLegalActions(agent):
				nextState = state.generateSuccessor(agent, action)
				
				if agent == 0:
					value = max(value, minimaxHelper(depth+1, (agent+1) % n, nextState, alpha, beta))
					if beta < value:
						return value
					alpha = max(alpha, value)

				else:
					value = min(value , minimaxHelper(depth+1, (agent+1) % n, nextState, alpha, beta))
					if value < alpha:
						return value
					beta = min(value, beta)
			
			return value

		alpha = float("-inf")
		beta = float("inf")
		for action in gameState.getLegalActions(0):
			value = minimaxHelper(1,1,gameState.generateSuccessor(0, action),alpha,beta)
			if value > alpha:
				best = action
				alpha = value 
		return best

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"
		n = gameState.getNumAgents()
		treeDepth = self.depth * n

		avg = lambda x: sum(x)/len(x)

		def minimaxHelper(depth, agent, state):

			if depth == treeDepth or state.isWin() or state.isLose():
				return self.evaluationFunction(state)

			if agent == 0:
				evalFunction = max
			else:
				evalFunction = avg

			newStates = [state.generateSuccessor(agent, action) for action in state.getLegalActions(agent)]
			return evalFunction([minimaxHelper(depth+1, (agent+1) % n, nextState) for nextState in newStates])

		actions = gameState.getLegalActions(0)
		return max(actions, key = lambda x: minimaxHelper(1,1,gameState.generateSuccessor(0, x)))

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: The evaluation function is a linear combination of features (as described in 
	the notes). The baseline for the function is triviallly the score. 

	Feature #1 = Food
	The food feature is defined as the (food left + distance to the closest food). This feature
	is subtracted so that the agent is encouraged to lower the food left and lower the distance 
	to the closest food. 

	Feature #2 = Capsules + scared ghosts
	This feature represents the agent's ability to eat the other ghosts. A weight of 10 times 
	this feature is subtracted to emphasize how important it is to eat the ghosts to improve 
	the score.

	Feature #3 = ghosts that are not scared
	This feature represents the quantified version of the danger the agent faces. It is capped
	at 6, since past 6 spaces the threat is pretty consistent. This feature is added to push 
	towards keeping a distance of at least 6 at all times.
	"""
	"*** YOUR CODE HERE ***"

	pos = currentGameState.getPacmanPosition()
	food = currentGameState.getFood().asList()
	ghostStates = currentGameState.getGhostStates()
	score = currentGameState.getScore()
	foodLeft = currentGameState.getNumFood()
	# newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

	capsules= currentGameState.getCapsules()
	capsulesLeft = len(capsules)

	minScaredDistance = minNonScaredDistance = minCapsuleDist = minFoodDist = 0

	if foodLeft > 0:
		minFoodDist = min([manhattanDistance(pos, food) for food in food])

	if capsulesLeft > 0:
		minCapsuleDist = min([manhattanDistance(pos, capsule) for capsule in capsules])

	
	scaredGhosts = [ghost for ghost in ghostStates if ghost.scaredTimer]
	nonScaredGhosts = [ghost for ghost in ghostStates if not ghost.scaredTimer]

	if len(scaredGhosts) > 0:
		minScaredDistance = min([manhattanDistance(pos, ghost.getPosition()) for ghost in scaredGhosts])

	if len(nonScaredGhosts) > 0:
		minNonScaredDistance = min([manhattanDistance(pos, ghost.getPosition()) for ghost in nonScaredGhosts])


	if minNonScaredDistance < 6:
	  minNonScaredDistance = 6



	return score -(foodLeft + minFoodDist) -10 *(minScaredDistance+capsulesLeft) + minNonScaredDistance


# Abbreviation
better = betterEvaluationFunction
