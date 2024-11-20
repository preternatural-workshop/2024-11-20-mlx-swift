//
//  MistralModelManager.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import Generation
import MLX
import MLXRandom
import Tokenizers

class MistralModelManager {
    public enum ModelState {
        case ready(Double?)
        case generating(Double)
        case failed(String)
    }
    
    public var modelState: ModelState = .ready(0)
    
    public private(set) var generationParameters: GenerateParameters
    
    /// The model here is already expected to have the weights set on it.
    private var model: MistralModel
    
    private var tokenizer: Tokenizer
    private let maxTokens: Int
    private let seed: UInt64
    
    @preconcurrency @MainActor
    init(
        model: MistralModel,
        tokenizer: Tokenizer,
        maxTokens: Int,
        temperature: Float,
        seed: UInt64
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.maxTokens = maxTokens
        self.generationParameters = .init()

        self.seed = seed
        
        MLXRandom.seed(seed)
    }
    
    @_optimize(speed)
    public func generate(
        parameters: GenerateParameters,
        prompt: String
    ) throws -> String {
        self.modelState = .generating(0.0)
        
        let initialTokens: [Int] = tokenizer.encode(text: prompt)
        let initialPrompt = MLXArray(initialTokens)
        
        var tokens = [Int]()
        var progress: Double = 0
        
        for token in TokenGenerator(
            prompt: initialPrompt,
            model: model,
            parameters: parameters
        ) {
            if token == tokenizer.unknownTokenId || tokens.count == maxTokens {
                break
            }
            
            tokens.append(token)
            
            progress = Double(tokens.count) / Double(maxTokens)
            
            self.modelState = .generating(progress)
        }
        
        self.modelState = .ready(progress)
        
        return tokenizer.decode(tokens: tokens)
    }
}
