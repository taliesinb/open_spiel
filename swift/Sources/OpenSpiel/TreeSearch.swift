//
//  TreeSearch.swift
//  alphazero_draft
//
//  Created by Taliesin Beynon on 1/15/20.
//  Copyright © 2020 Taliesin Beynon. All rights reserved.
//

import Foundation

extension StateProtocol {
    func utilities() -> [Double] {
        return (0..<game.playerCount).map { utility(for:.player($0)) }
    }
}

/// Different possibilities for player utilities of nodes in the search tree.
enum Outcome {
    
    /// A subtree has been solved, so we know the optimal utilties for the current player
    case solved([Double])
    
    /// A subtree has not been solved.
    case unsolved
    
    /// Return the solved optimal utility for a particular player, or nil if this isn't known.
    func rewardFor(player : Player) -> Double? {
        switch (self, player) {
        case (.solved(let rewards), .player(let playerNumber)):
            return rewards[playerNumber]
        default:
            return nil
        }
    }
}

/// Represents a node in the tree for a given game. We gradually layer in additional functionality via subclasses.
class SearchNode<Game : GameProtocol> {

    typealias SelfType = SearchNode<Game>
    
    /// Whether this node has been visited, meaning that we've examined the corresponding game state.
    var visited : Bool = false
    
    /// Whether the corresponding game state was examined and found to be terminal
    var terminal : Bool = false
    
    /// Whether we are a leaf node or not (this is also equal to `terminal || !visited`)
    var leaf : Bool { get { children.count == 0 } }

    /// The number of simulations that passed through or ended on this node.
    var exploreCount : Double = 0
    
    /// The prior probability of the action that lead to this node, produced by the StateEstimator.
    var priorProbability : Double = 0.0

    /// Saving the game state allows us to repeatedly traverse the tree without re-evaluating the game each time.
    var state : Game.State? = nil
   
    /// *Tree pointers*
    
    /// Dictionary mapping actions to child nodes
    var children : [Game.Action : SelfType] = [:]

    /// The action our parent took to visit us (nil for root)
    var parentAction : Game.Action? = nil

    /// The parent node (nil for the root of the tree)
    var parent : SelfType? = nil
    
    /// *Initializers*
    
    /// Create an empty root node
    required init() { }
    
    /// Create a new unvisited search node
    required init(priorProbability prior : Double, parent par : SelfType?, parentAction action : Game.Action?) {
        priorProbability = prior
        parent = par
        parentAction = action
    }
    
    /// *Methods inherited by subclasses*
    
    /// Inherited by subclasses to process value estimates
    func reachedNonterminalState(value : ValueEstimate) {
    }
    
    /// *Code for solving subtrees*

    /// The (solved) outcome for this node. Used to solve parts of the tree.
    var outcome : Outcome = .unsolved
    
    /// If this part of the tree has been solved, the best action for the current player to take, otherwise nil
    var bestAction : Game.Action? = nil

    func solvedValue() -> Double? {
        return outcome.rewardFor(player: state!.currentPlayer)
    }

    /// Propogate the outcome up the tree as much as possible, opportunistically attempting to solve as large a subtree as possible.
    func trySolve() {
        guard let state = state else { return } /// We are guaranteed to have a state for visited nodes
        let maxUtility = state.game.maxUtility
        let currentPlayer = state.currentPlayer
        var bestReward = -Double.infinity
        for (action, child) in children {
            if let reward = child.outcome.rewardFor(player:currentPlayer) {
                if reward >= bestReward {
                    bestAction = action
                    bestReward = reward
                    /// if one of the actions has the maximum possibile utility, we can short-circuit the search
                    /// since a better solution is not possible
                    if reward == maxUtility { break }
                }
            } else {
                /// if one of the children isn't solved, this node can't be solved either
                bestAction = nil
                return
            }
        }
        outcome = children(self)[bestAction!]!.outcome
        parent?.trySolve()
    }
}

/// Methods relating to visiting novel states.
extension SearchNode {
    
    /// Visit a previously unvisited node, evaluating and expanding if it is not terminal, otherwise calling `reachedTerminalState`
    func evaluateLeaf<Evaluator : EvaluatorProtocol>(evaluator : Evaluator) where Evaluator.Game == Game {
        if visited { return }
        visited = true
        /// Derive the new state from the parent's state and the action that lead from it to this node.
        let newState = parent!.state!.applying(parentAction!)
        terminal = newState.isTerminal
        if terminal {
            reachedTerminalState(state:newState)
        } else {
            expandNonterminalState(state:newState, evaluation:evaluator.evaluate(forState: newState))
        }
    }
    
    /// Attach an outcome to a previously unsolved node
    func reachedTerminalState(state : Game.State) {
        outcome = .solved(state.utilities())
        parent?.trySolve()
    }
    
    /// Evaluate the leaf state and create new child nodes for each of the actions that have non-zero prior.
    func expandNonterminalState(state newState : Game.State, evaluation : Evaluation<Game>) {
        state = newState /// save the state, we will need it to derive the state for our children, too.
        for (action, prob) in evaluation.prior where prob > 0 {
            children[action] = type(of:self).init(priorProbability:prob, parent:self, parentAction:action)
        }
        /// This method will be override by subclasses of `SearchNode` and implement techniques like value backpropogation
        reachedNonterminalState(value:evaluation.value)
    }
    
}

/// Methods relating to actions and exploration
extension SearchNode {
    
    /// Choose an action using an arbitrary score function
    func chooseAction(scoreFn : (SelfType) -> Double) -> Game.Action {
        precondition(!leaf)
        var bestAction = children.first!.0, bestReward = -Double.infinity
        for (action, child) in children {
            let reward = scoreFn(child)
            if reward > bestReward {
                bestAction = action
                bestReward = reward
            }
        }
        return bestAction
    }

    /// The fraction of simulations that choose the various available actions.
    func visitDistribution() -> [Game.Action : Double] {
        precondition(!leaf)
        let totalExploreCount = exploreCount
        return children.mapValues { Double($0.exploreCount) / totalExploreCount }
    }
    
}

extension SearchNode {
    
    /// Subclasses of `SearchNode` need to obtain a similarly-typed parent and children pointers, this makes it easy to do this
    func parent<SubSelf>(_ s : SubSelf) -> SubSelf? {
        if parent == nil { return nil }
        return (parent! as! SubSelf)
    }
    
    func children<SubSelf>(_ s : SubSelf) -> [Game.Action : SubSelf] {
        return (children as! [Game.Action : SubSelf])
    }
    
}

extension SearchNode : CustomStringConvertible {
    var description: String {
        return "SearchNode(\(exploreCount), \(visited), \(terminal))"
    }
}

/// `TreeSearchProtocol` holds basic parameters about how a tree should be searched.
struct TreeSearchParams {
    var initialExplorationNoise : DirichletNoise? = nil
    var numSimulations : Int = 100
    // var maxDepth
    // var maxTime
}

/// `TreeSearchProtocol` represents the global state, node class, and behavior associated with a particular form of tree search.
protocol TreeSearchProtocol {
    
    associatedtype Game : GameProtocol
    associatedtype Node
    /// Note: it would be natural to put the constraint : SearchNode<Game> above, because it would remove the need to put this same constraint
    /// downstream (on the buildTree method and TreeSearchPolicy struct), but Swift 5 cannot detect when the Node types of implementers
    /// of TreeSearchProtocol inherit from the same 'Game' as above, which is required.
    
    /// this is the score of a node for the purposes of exploration
    func explorationScore(node : Node) -> Double
    
    /// this is the value of a node as estimated by MCTS
    func estimateValue(node : Node) -> ValueEstimate
}

/// `TreeSearchProtocol` is about how to explore nodes, the following methods are agnostic to how this is done so we declare them on the protocol itself.
extension TreeSearchProtocol {
    
    /// Choose an action that maximizes the exploration score
    func chooseExplorationAction(node : SearchNode<Game>) -> Game.Action {
        return node.chooseAction { explorationScore(node: $0 as! Node) }
    }
    
    /// Repeatedly call `chooseExplorationAction` to explore the tree until a leaf is reached, which we return.
    func exploreToLeaf(node : SearchNode<Game>) -> SearchNode<Game> {
        var node = node
        while !node.leaf {
            let action = chooseExplorationAction(node:node)
            node = node.children[action]!
            node.exploreCount += 1
        }
        return node
    }
    
    /// Starting from a particular state, run `numSimulations` different episodes to build up the search tree via iterative deepening
    func buildTree<Evaluator : EvaluatorProtocol>(state : Game.State, evaluator : Evaluator, params : TreeSearchParams) -> Node where Game == Evaluator.Game, Node : SearchNode<Game> {
        
        /// Create a new root state
        let root = Node.init()
        
        /// The root has optional noise applied to it before being expanded
        let rootEvaluation = evaluator.evaluate(forState: state)
            params.initialExplorationNoise?.applyToPrior(rootEvaluation.prior)
        root.expandNonterminalState(state:state, evaluation:rootEvaluation)
        
        for _ in 1...params.numSimulations {
            /// iteratively deepen the tree by starting at the root, exploring by picking actions that maximize the exploration score,
            /// and once we reach a leaf, evaluating it (which will expand it for non-terminal game states)
            let leaf = exploreToLeaf(node:root)
            leaf.evaluateLeaf(evaluator:evaluator)
        }
        
        return root
    }
}

/// `TreeSearchPolicy` binds a particular leaf evaluator object with a tree search protocol object, giving us a `StochasticPolicy`etc.
struct TreeSearchPolicy<TreeSearch : TreeSearchProtocol, Evaluator : EvaluatorProtocol> where TreeSearch.Node : SearchNode<TreeSearch.Game>, TreeSearch.Game == Evaluator.Game {
    
    typealias Game = TreeSearch.Game
    
    /// The evaluator that will be used on leaves of the search tree to estimate values and bias the search towards specific actions.
    let evaluator : Evaluator
    
    /// The tree exploration strategy that will be used to expand nodes and estimate their value
    let search : TreeSearch
    
    /// Basic parameters about how long to search
    let params : TreeSearchParams
}


/// `TreeSearchPolicy` satisfies `StochasticPolicy` through the visit distribution of the root node.
extension TreeSearchPolicy : StochasticPolicy {
    
    func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
        let root = search.buildTree(state:state, evaluator:evaluator, params:params)
        return root.visitDistribution()
    }
}

/// `TreeSearchPolicy` satisfies `EvaluatorProtocol` by evaluation of the root node.
extension TreeSearchPolicy : EvaluatorProtocol {

    func evaluate(forState state : Game.State) -> Evaluation<Game> {
        let root = search.buildTree(state:state, evaluator:evaluator, params:params)
        let prior = root.visitDistribution()
        let value = search.estimateValue(node:root)
        return Evaluation(prior:prior, value:value)
    }
}

///// If the leaf evaluator is trainable, the MCTS evaluator is also trainable (by passing the rollouts through)
//extension MCTSEvaluator : TrainableEvaluatorProtocol where Evaluator : TrainableEvaluatorProtocol {
//
//    func train(rollouts : GameRolloutData<Game>) {
//        evaluator.train(rollouts:rollouts)
//    }
//
//}
//




/// `SelfPlayTrainer` trains a `TrainableEvaluator` via self play. Games are played. States, values, and outcomes for each
/// game are stored, and the evaluator is trained to predict the outcomes (although how the evaluator chooses to train itself with trajectories is not
/// part of `SelfPlayTrainer`).

//struct SelfPlayTrainer<TrainableEvaluator : TrainableEvaluatorProtocol> {
//
//    var evaluator : TrainableEvaluator
//    var game : Game
//    var numGames : Int = 100
//
//    func train() {
//        for n in 1...numGames {
//            evaluator.
//        }
//
//    }
//}
