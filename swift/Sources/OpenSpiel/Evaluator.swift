//
//  Evaluator.swift
//  alphazero_draft
//
//  Created by Taliesin Beynon on 2020/01/09.
//  Copyright © 2020 Taliesin Beynon. All rights reserved.
//

import Foundation
import TensorFlow


/// `ValueEstimate` is how an object satisfying `EvaluatorProtocol` estimates values of a game state.
protocol ValueEstimate {
    func mean() -> Double
    func sample() -> Double
}

/// This is a point estimate, which is the most common, and also used by AlphaZero. Distributional ValueEstimates are declared in Thompson.swift.
extension Double : ValueEstimate {
    func mean() -> Double { return self }
    func sample() -> Double { return self }
}

/// `Evaluation` is the structure returned by a `EvaluatorProtocol` that describes the value of a state for the current player.
/// It contains a prior over actions, and a value estimate.
struct Evaluation<Game : GameProtocol> {

    /// mapping from Action to its prior probability
    let prior : [Game.Action: Double]

    /// a distribution over values
    let value : ValueEstimate
}

/// `EvaluatorProtocol` defines a method that evaluates game states, yielding an `Evaluation`
protocol EvaluatorProtocol  {
    associatedtype Game : GameProtocol
    func evaluate(forState : Game.State) -> Evaluation<Game>
}

/// Default implementations: objects that implement`EvaluatorProtocol` can naturally produce values and actionProbabilities individually,
/// since an `Evaluation` is the combination of these.
extension EvaluatorProtocol {
    func value(forState state: Game.State) -> Double {
        return evaluate(forState: state).value.mean()
    }
    func actionProbabilities(forState state : Game.State) -> [Game.Action : Double] {
        return evaluate(forState: state).prior
    }
}

func uniformPrior<Action:Hashable>(_ actions : [Action]) -> [Action : Double] {
    let uniform = repeatElement(1.0 / Double(actions.count), count:actions.count)
    return .init(uniqueKeysWithValues:zip(actions, uniform))
}


/// `RandomEvaluator` produces uniform action distributions, with no constant value estimate.
class RandomEvaluator<Game:GameProtocol> : EvaluatorProtocol {
    func evaluate(forState state : Game.State) -> Evaluation<Game> {
        return Evaluation(prior:uniformPrior(state.legalActions), value:0.0)
    }
}

/// `RandomRolloutEvaluator`s perform random rollouts to estimate the value of a game state, with a uniform prior.
class RandomRolloutEvaluator<Game:GameProtocol> : EvaluatorProtocol {
    let numRollouts = 100
    func evaluate(forState state : Game.State) -> Evaluation<Game> {
        let player = state.currentPlayer
        var total : Double = 0.0
        for _ in 1...numRollouts {
            var scratch = state
            while !scratch.isTerminal {
                scratch.apply(scratch.legalActions.randomElement()!)
            }
            total += scratch.utility(for:player)
        }
        let value = total / Double(numRollouts)
        return Evaluation(prior:uniformPrior(state.legalActions), value:value)
    }
}


//typealias ValueFunction<Game : GameProtocol> = (Game.State) -> Double
//
///// `MaxEvaluator` estimates the value of a state to be the best score that can be reached under one action
///// (for an arbitrary score function). It also produces a prior that is uniform over the maximizing actions.
//struct MaxEvaluator<Game : GameProtocol> : EvaluatorProtocol {
//    let scoreFn : ValueFunction<Game>
//    func evaluate(forState state : Game.State) -> Evaluation<Game> {
//        var bestActions : [Game.Action] = []
//        var bestReward = -Double.infinity
//        for action in state.legalActions {
//            let nextState = state.applying(action)
//            let nextValue = scoreFn(nextState)
//            if nextValue == bestReward {
//                bestActions.append(action)
//            } else if nextValue > bestReward {
//                bestActions = [action]
//                bestReward = nextValue
//            }
//        }
//        return Evaluation(prior:uniformPrior(bestActions), value:.point(bestReward))
//    }
//}

///// `MinimaxEvaluator` estimates the value of a state to be the maximum score possible when opponents maximize
///// their scores in response to a single action (for an arbitrary score function).
//struct MinimaxEvaluator<Game : GameProtocol> : MaxEvaluator<Game> {
//    let innerEvaluator = GreedyScoreEvaluator<Game>
//    init(scoreFn fn: ValueFunction<Game>) {
//        inner = .init(scoreFn: fn)
//        self.init(scoreFn: inner.value)
//    }
//    func evaluate(forState state : Game.State) -> Evaluation<Game> {
//        bestActions
//        return Evaluation(prior:uniformPrior(bestActions), value:bestReward)
//    }
//}


class FirstActionEvaluator<Game:GameProtocol> : EvaluatorProtocol {
    func evaluate(forState state : Game.State) -> Evaluation<Game> {
        let firstAction = state.legalActions.first!
        return Evaluation(prior:uniformPrior([firstAction]), value:0.0)
    }
}


/// `GameRolloutData` collects states, evaluations, and outcomes from a game, which can be used to train a trainable evaluator.
struct GameRolloutData<Game : GameProtocol> {
    var states : [Game.State]
    var evaluations : [Evaluation<Game>]
    var outcomes : [Double]
    var playerID : [Int]
}

/// `TrainableEvaluatorProtocol` extends `EvaluatorProtocol` with the ability to train on `GameRolloutData`.
protocol TrainableEvaluatorProtocol : EvaluatorProtocol {
    func train(rollouts : GameRolloutData<Game>)
}

//extension EvaluatorProtocol {
//    
//    func selfPlay(game : Game) -> GameRolloutData<Game> {
//        var state = game.initialState
//        var rollout : GameRolloutData<Game>
//        while !state.isTerminal {
//            let player = state.currentPlayer
//            if player == .chance {
//                // TODO: fill this in
//            } else if case let player() {
//                let evaluation = self.evaluate(forState: state)
//                evaluation.prior
//                rollout.states.append(state)
//                rollout.evaluations.append(evaluation)
//                rollout.playerID.append(player)
//            }
//        }
//        let outcome = state.utilities()
//        for id in outcome.playerID
//    }
//
//}
