//
//  Random.swift
//  alphazero_draft
//
//  Created by Taliesin Beynon on 2020/01/13.
//  Copyright © 2020 Taliesin Beynon. All rights reserved.
//

import Foundation

import Python

var np = Python.import("numpy")

extension Array where Element == Double {
    
    func sum() -> Double {
        return self.reduce(0.0, +)
    }
    
    func normalized() -> [Double] {
        let total = sum()
        return self.map { $0 / total }
    }

}

//func sampleNormal(mean : Double, stddev : Double) -> Double {
//    return TensorFlow.NormalDistribution(mean: mean, standardDeviation: stddev).next(using:ARC4RandomNumberGenerator.global)
//}

func sampleNormal(_ mean : Double, _ stddev : Double) -> Double {
    return Double(np.random.normal(mean, stddev))!
}

func sampleBeta(_ s1 : Double, _ s2 : Double) -> Double {
    return Double(np.random.beta(s1, s2))!
}

func sampleDirichlet(_ shape : [Double]) -> [Double] {
    return Array(np.random.dirichlet(shape))!
}

func sampleBool(_ p : Double) -> Bool {
    return Double.random(in: 0...1) < p
}

func sampleUniform() -> Double {
    return Double.random(in: 0...1)
}

func sampleWeightedIndex(_ weights : [Double]) -> Int {
    var z = sampleUniform()
    for (i, p) in weights.enumerated() {
        z -= p
        if z <= 0 { return i }
    }
    return -1
}

struct DirichletNoise {
    
    let alpha : Double = 0.0
    let epsilon : Double = 0.0
        
    func applyToPrior<Action : Hashable>(_ prior : [Action : Double]) {
        if epsilon == 0 { return }
        let constantAlpha = [Double](repeating:alpha, count:prior.count)
        let noise = sampleDirichlet(constantAlpha)
        for (i, (_, var prob)) in prior.enumerated() {
            prob = (1 - epsilon) * prob + epsilon * noise[i]
        }
    }
}
