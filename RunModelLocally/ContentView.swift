//
//  ContentView.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var modelLoader = ModelDownloadManager.shared
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var userPrompt = ""
    @State private var generatedOutput = ""
    @State private var isGenerating = false
    @State private var mistralManager: MistralManager?
    
    var body: some View {
        VStack(spacing: 16) {
            Group {
                switch modelLoader.model.state {
                case .notDownloaded:
                    Button("Download Model") {
                        Task {
                            await HuggingFaceDownloader.shared.downloadModel()
                        }
                    }
                    
                case .downloading(let progress):
                    VStack {
                        ProgressView(value: progress) {
                            Text("Downloading Model: \(Int(progress * 100))%")
                        }
                        
                        Button("Cancel Download") {
                            handleModelDeletion()
                        }
                    }
                    
                case .downloaded:
                    VStack(spacing: 20) {
                        Text("Model Loaded")
                            .foregroundColor(.green)
                            .font(.headline)
                        
                        // Input Area
                        VStack(alignment: .leading) {
                            Text("Enter your prompt:")
                                .font(.subheadline)
                            TextEditor(text: $userPrompt)
                                .frame(height: 50)
                                .padding(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.gray.opacity(0.2))
                                )
                        }
                        
                        // Generate Button
                        Button(action: {
                            generateResponse()
                        }) {
                            HStack {
                                Text(isGenerating ? "Generating..." : "Generate Response")
                                if isGenerating {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle())
                                }
                            }
                        }
                        .disabled(userPrompt.isEmpty || isGenerating)
                        
                        VStack(alignment: .leading) {
                            Text("Response:")
                                .font(.subheadline)
                            TextEditor(text: .constant(generatedOutput))
                                .frame(height: 200)
                                .padding(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.gray.opacity(0.2))
                                )
                                .scrollContentBackground(.hidden)
                                .scrollDisabled(false)
                        }
                        
                        
                        Button("Remove Model") {
                            handleModelDeletion()
                        }
                        .buttonStyle(BorderedProminentButtonStyle())
                        .foregroundColor(.red)
                        .padding(.top)
                        .tint(.red)
                        
                    }
                    .padding()
                    .task {
                        do {
                            let model = try await ModelDownloadManager.shared.loadModel()
                            let tokenizer = try await ModelDownloadManager.shared.loadTokenizer()
                            
                            mistralManager = MistralManager(
                                model: model,
                                tokenizer: tokenizer,
                                maxTokens: 100,
                                temperature: 0.6,
                                seed: 0
                            )
                        } catch {
                            print(error.localizedDescription)
                        }
                    }
                    
                case .failed(let error):
                    VStack {
                        Text("Download Failed")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                        
                        Button("Try Again") {
                            Task {
                                await HuggingFaceDownloader.shared.downloadModel()
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .padding()
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func generateResponse() {
        guard let mistralManager = mistralManager else {
            errorMessage = "Model not properly initialized"
            showError = true
            return
        }
        
        isGenerating = true
        Task {
            do {
                let result = try mistralManager.generate(
                    parameters: GenerateParameters(),
                    prompt: userPrompt
                )
                
                await MainActor.run {
                    generatedOutput = result
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    showError = true
                    isGenerating = false
                }
            }
        }
    }
    
    private func handleModelDeletion() {
        Task {
            do {
                try await HuggingFaceDownloader.shared.deleteDownloadedModel()
                await MainActor.run {
                    mistralManager = nil
                    userPrompt = ""
                    generatedOutput = ""
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
}
