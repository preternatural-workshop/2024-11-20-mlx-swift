//
//  DynamicNTKScalingRoPE.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import Foundation
import MLX
import MLXNN
import MLXFast

public class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int?
    let traditional: Bool
    let base: Float
    var scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?

    init(
        dims: Int, maxPositionEmbeddings: Int?, traditional: Bool = false,
        base: Float = 10000, scale: Float = 1.0, ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional
        self.base = computeBaseFrequency(
            base: base, dims: dims, ropeType: ropeType, ropeScaling: ropeScaling)
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(1) + offset
        var base = self.base
        if let maxPositionEmbeddings, seqLen > maxPositionEmbeddings {
            let factorAdjustment = Float(seqLen) / Float(maxPositionEmbeddings) - 1
            let dimensionRatio = Float(dims) / Float(Float(dims) - 2)
            let adjustedScale = scale * pow(1 + factorAdjustment, dimensionRatio)
            base *= adjustedScale
        }
        return MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional, base: base, scale: scale, offset: offset)
    }
}

