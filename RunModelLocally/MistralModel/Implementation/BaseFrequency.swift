//
//  BaseFrequency.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import Foundation

func computeBaseFrequency(
    base: Float, dims: Int, ropeType: String, ropeScaling: [String: StringOrNumber]?
)
    -> Float
{
    if ropeType != "llama3" {
        return base
    }

    guard let ropeScaling = ropeScaling else {
        return base
    }

    guard case .float(let factor) = ropeScaling["factor"],
        case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
        case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
        case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
            ?? .float(8192)
    else {
        return base
    }

    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    let freqs = (0 ..< dims).compactMap { index -> Float? in
        if index % 2 == 0 {
            return pow(base, Float(index) / Float(dims))
        }
        return nil
    }

    let newBaseFreqs = freqs.map { freq -> Float in
        let wavelen = 2 * .pi / freq
        let smooth = max(
            0, min(1, (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen)))
        return freq * ((1 - smooth) * factor + smooth)
    }

    return newBaseFreqs.reduce(0, +) / Float(newBaseFreqs.count)
}

