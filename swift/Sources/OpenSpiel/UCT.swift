//
//  UCT.swift
//  alphazero_draft
//
//  Created by Taliesin Beynon on 2020/01/12.
//  Copyright © 2020 Taliesin Beynon. All rights reserved.
//

import Foundation

class MeanRewardSearch<Game : GameProtocol> : TreeSearchProtocol {
    
    class Node : SearchNode<Game> {

        /// The total reward of simulations that passed through this node.
        var totalReward = 0.0
       
        var meanReward : Double { get { totalReward / exploreCount } }
        
        override func reachedNonterminalState(value : ValueEstimate) {
            backpropogate(value.mean())
        }

        func backpropogate(_ value : Double) {
            totalReward += value
            parent(self)?.backpropogate(value)
        }
    }
    
    func explorationScore(node : Node) -> Double {
        return node.meanReward
    }
    
    func estimateValue(node : Node) -> ValueEstimate {
        return node.meanReward
    }
}

class UCTSearch<Game : GameProtocol> : MeanRewardSearch<Game> {
        
    class Node : MeanRewardSearch<Game>.Node {

        func uctBonus() -> Double {
            return sqrt(log(parent!.exploreCount) / exploreCount)
        }
        
        func puctBonus() -> Double {
            return priorProbability * sqrt(parent!.exploreCount) / (exploreCount + 1)
        }
    }
    
    let uctC : Double = 0.1
    
    let score : Score = .PUCT
    enum Score { case PUCT, UCT }
    
    func explorationScore(node : Node) -> Double {
        let bonus = (score == .PUCT) ? node.puctBonus() : node.uctBonus()
        return node.meanReward + uctC * bonus
    }
}

