//
//  GenerateParameters.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import Foundation

/// Parameters for text generation, see ``TokenGenerator``
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize = 512

    /// sampling temperature
    public var temperature: Float = 0.6

    /// top p sampling
    public var topP: Float = 1.0

    /// penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int = 20

    public init(
        temperature: Float = 0.6, topP: Float = 1.0, repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}
