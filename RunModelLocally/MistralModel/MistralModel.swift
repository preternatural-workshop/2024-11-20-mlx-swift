//
//  MistralModel.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import MLX
import MLXNN

public class MistralModel: Module, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let headDim: IntOrPair

    fileprivate let model: MistralModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MistralConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.headDim = .init(args.resolvedHeadDimensions)
        self.model = MistralModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
    
    public func newCache(parameters: GenerateParameters) -> [KVCache] {
        kvHeads.map { n in
            KVCacheSimple(headDim: headDim, kvHeads: n)
        }
    }
}
