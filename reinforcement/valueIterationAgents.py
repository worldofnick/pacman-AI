# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.tempValues = self.values.copy()

        # Write value iteration code here
        for i in range(0, iterations):
          self.value_iteration()
          self.values = self.tempValues.copy()

    def value_iteration(self):

        for state in self.mdp.getStates():
          max_value = float('-inf')
          for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            max_value = max(max_value, qValue)
            self.tempValues[state] = max_value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        sum = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action) # array of tuples (state, prob)
        for nextState, prob in transitions:
          sum += prob * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState]))
        
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        max_action = None
        max_reward = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
          transitions = self.mdp.getTransitionStatesAndProbs(state, action)
          qValue = self.computeQValueFromValues(state, action)

          if qValue > max_reward:
            max_reward = qValue
            max_action = action

        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
