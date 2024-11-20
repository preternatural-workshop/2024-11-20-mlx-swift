//
//  ModelLoader.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import SwiftUI
import MLX
import MLXNN
import Tokenizers
import Hub

public final class ModelLoader: ObservableObject {
    
    @MainActor
    public static let shared = ModelLoader()
    
    public static let modelPath = "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX"
    
    @Published public var model = DownloadableModel(
        name: ModelLoader.modelPath,
        url: URL(string: "\(HuggingFaceDownloader.downloadsURL)/\(modelPath)"),
        state: .notDownloaded
    )
    
    @MainActor
    private init() {
        if HuggingFaceDownloader.shared.isModelDownloaded() {
            model.state = .downloaded
        }
    }
}

// from: https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/LLM/Load.swift
extension ModelLoader {
    /// Loads `MistralModel` and sets the input parameters (the weights from the `.safetensors` files).
    public func loadModel() async throws -> MistralModel {
        let configurationURL = model.url!.appendingPathComponent("config.json")
        
        let configuration = try JSONDecoder().decode(
            MistralConfiguration.self,
            from: Data(contentsOf: configurationURL)
        )
        
        let model = MistralModel(configuration)
        let weights: [String: MLXArray] = try loadWeights()
        
        quantizeIfNeeded(
            model: model,
            weights: weights,
            quantization: configuration.quantization
        )
        
        let parameters: NestedDictionary<String, MLXArray> = ModuleParameters.unflattened(consume weights)
        
        try model.update(parameters: parameters, verify: [.none])
            
        eval(model)
        
        return model
    }
    
    public func loadTokenizer() async throws -> Tokenizers.Tokenizer {
        do {
            let config = LanguageModelConfigurationFromHub(modelName: model.name)

            guard let tokenizerConfig = try await config.tokenizerConfig else {
                throw Error(message: "missing config")
            }

            var tokenizerData = try await config.tokenizerData

            if let tokenizerClass = tokenizerConfig.tokenizerClass?.stringValue {
                switch tokenizerClass {
                case "T5Tokenizer":
                    break
                default:
                    tokenizerData = discardUnhandledMerges(tokenizerData: tokenizerData)
                }
            }

            return try PreTrainedTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } catch {
            return try await AutoTokenizer.from(pretrained: model.id)
        }
    }
}

extension ModelLoader {
    
    
    private func loadWeights() throws -> [String: MLXArray] {
        let directory = model.url!
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
        
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let weightArrays = try loadArrays(url: url)
                for (key, value) in weightArrays {
                    weights[key] = value
                }
            }
        }
        
        return weights
    }
    
    private func discardUnhandledMerges(
        tokenizerData: Config
    ) -> Config {
        if let model = tokenizerData.model {
            if let merges = model.dictionary["merges"] as? [String] {
                let newMerges =
                merges
                    .filter {
                        $0.split(separator: " ").count == 2
                    }
                if newMerges.count != merges.count {
                    var newModel = model.dictionary
                    newModel["merges"] = newMerges
                    var newTokenizerData = tokenizerData.dictionary
                    newTokenizerData["model"] = newModel
                    return Config(newTokenizerData)
                }
            }
        }
        return tokenizerData
    }
    
    private func quantizeIfNeeded(
        model: MistralModel,
        weights: [String: MLXArray],
        quantization: MistralConfiguration.Quantization
    ) {
        
        func linearFilter(_ path: String, layer: Module) -> Bool {
            if let layer = layer as? Linear {
                return layer.weight.dim(0) != 8
            }
            return false
        }
        
        var filter = linearFilter
        
        if weights["lm_head.scales"] == nil {
            let vocabularySize = model.vocabularySize
            func vocabularySizeFilter(_ path: String, layer: Module) -> Bool {
                if let layer = layer as? Linear {
                    return layer.weight.dim(0) != 8 && layer.weight.dim(0) != vocabularySize
                }
                return false
            }
            filter = vocabularySizeFilter
        }
        
        func apply(_ layer: Module, groupSize: Int, bits: Int) -> Module? {
            if let linear = layer as? Linear {
                return QuantizedLinear(linear, groupSize: groupSize, bits: bits)
            }
            return nil
        }
        
        quantize(
            model: model,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            filter: filter,
            apply: apply
        )
    }
}

extension ModelLoader {
    struct Error: Swift.Error {
        let message: String
    }
}

