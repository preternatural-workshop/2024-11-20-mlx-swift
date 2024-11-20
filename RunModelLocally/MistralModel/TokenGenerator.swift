//
//  TokenGenerator.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

//import AsyncAlgorithms
import Foundation
import MLX
import MLXRandom

/// Synchronous generator of tokens.
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenGenerator: Sequence, IteratorProtocol {
    let model: MistralModel
    let parameters: GenerateParameters

    var prompt: MLXArray
    var cache: [KVCache]
    var repetitionContext: TokenRepetitionContext
    let sampleContext: TokenSampleContext

    public init(
        prompt: MLXArray,
        model: MistralModel,
        parameters: GenerateParameters
    ) {
        self.model = model
        self.parameters = parameters
        self.prompt = prompt
        self.cache = model.newCache(parameters: parameters)

        self.repetitionContext = TokenRepetitionContext(prompt: prompt, parameters: parameters)
        self.sampleContext = TokenSampleContext(parameters: parameters)

        // prepare the prompt in chunks if larger than the prefill size
        while prompt.size > parameters.prefillStepSize {
            _ = model(
                prompt[.newAxis,..<parameters.prefillStepSize],
                cache: cache.isEmpty ? nil : cache
            )
            
            eval(cache)
            
            self.prompt = prompt[parameters.prefillStepSize...]
        }

        // evaluate the remainder of the prompt -- this primes the pump
        self.prompt = step(previous: prompt)
        
        asyncEval(prompt)
    }

    /// Evaluate the next token and return the new token (y) and cache state.
    ///
    /// This may mutate the repititionContext.
    mutating func step(
        previous: MLXArray
    ) -> MLXArray {
        var logits: MLXArray
        logits = model(previous[.newAxis], cache: cache.isEmpty ? nil : cache)

        logits = logits[0..., -1, 0...]
        logits = repetitionContext.applyRepetitionPenalty(logits: logits)

        let prompt = sampleContext.sample(logits: logits)

        repetitionContext.append(token: prompt)

        return prompt
    }

    mutating public func next() -> Int? {
        // save current value -- this will be returned
        let previousPrompt = prompt

        // compute the next state and async eval the next token
        prompt = step(previous: previousPrompt)
        asyncEval(prompt)

        return previousPrompt.item(Int.self)
    }
}

extension TokenGenerator {
    /// Encapsulaton of the repetitionPenalty
    struct TokenRepetitionContext: Sendable {
        /// tokens in the repetition context sliding window
        var tokens: [Int]
        
        /// current write into into the tokens circular array
        var index = 0
        
        /// penalty factor for repeating tokens
        let repetitionPenalty: Float?
        
        /// number of tokens to consider for repetition penalty
        let repetitionContextSize: Int
        
        init(
            prompt: MLXArray,
            parameters: GenerateParameters
        ) {
            self.repetitionPenalty = parameters.repetitionPenalty
            self.repetitionContextSize = parameters.repetitionContextSize
            
            if repetitionPenalty != nil && repetitionContextSize > 1 {
                if prompt.shape[0] <= repetitionContextSize {
                    self.tokens = prompt.asArray(Int.self)
                } else {
                    self.tokens = prompt[(-repetitionContextSize)...].asArray(Int.self)
                }
            } else {
                self.tokens = []
            }
        }
        
        func applyRepetitionPenalty(
            logits: MLXArray
        ) -> MLXArray {
            if let penalty = repetitionPenalty, tokens.count > 0 {
                let indices = MLXArray(tokens.map { UInt32($0) })
                var selectedLogits = logits[0..., indices]
                
                selectedLogits = MLX.where(
                    selectedLogits .< 0, selectedLogits * penalty, selectedLogits / penalty)
                
                logits[0..., indices] = selectedLogits
                return logits
            }
            
            return logits
        }
        
        mutating func append(
            token: MLXArray
        ) {
            if repetitionPenalty != nil {
                if tokens.count >= repetitionContextSize {
                    tokens[index] = token.item(Int.self)
                    index = (index + 1) % repetitionContextSize
                } else {
                    tokens.append(token.item(Int.self))
                }
            }
        }
    }
    
    struct TokenSampleContext {
        let temp: MLXArray
        let topP: MLXArray
        let useTopP: Bool
        let useArgMax: Bool
        
        init(parameters: GenerateParameters) {
            self.temp = MLXArray(parameters.temperature)
            self.topP = MLXArray(parameters.topP)
            self.useTopP = parameters.topP > 0 && parameters.topP < 1
            self.useArgMax = parameters.temperature == 0
        }
        
        private let compiledTopPSampling: (MLXArray, MLXArray, MLXArray) -> MLXArray = {
            compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) {
                logits, topP, temp in
                let probs = softmax(logits / temp, axis: -1)
                let sortedIndices = argSort(probs, axis: -1)
                
                // probs shape is [B,V] and after take it will be [1, B, V], so we squeeze it back to [B, V]
                let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)
                
                let cumulativeProbs = cumsum(sortedProbs, axis: -1)
                
                let topProbs = MLX.where(
                    cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))
                
                let sortedToken = categorical(log(topProbs))
                return sortedIndices.squeezed(axis: 0)[sortedToken]
            }
        }()
        
        private let compiledCategorical: (MLXArray, MLXArray) -> MLXArray = {
            compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) { logits, temp in
                categorical(logits * (1 / temp))
            }
        }()
        
        private func topPSampling(logits: MLXArray) -> MLXArray {
            var logits = logits
            if logits.dtype == .bfloat16 {
                logits = logits.asType(.float32)
            }
            
            return compiledTopPSampling(logits, topP, temp)
        }
        
        func sample(logits: MLXArray) -> MLXArray {
            if useArgMax {
                return argMax(logits, axis: -1)
            } else {
                if useTopP {
                    return topPSampling(logits: logits)
                } else {
                    return compiledCategorical(logits, temp)
                }
            }
        }
    }
}
