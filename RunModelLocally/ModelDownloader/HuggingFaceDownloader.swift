//
//  HuggingFaceDownloader.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import SwiftUI
import Hub

public final class HuggingFaceDownloader: ObservableObject {
    
    @MainActor
    public static let shared = HuggingFaceDownloader()
    
    private static let huggingFaceToken = "YOUR_TOKEN"
    
    private static let hub = HubApi(hfToken: HuggingFaceDownloader.huggingFaceToken)
    
    public static var downloadsURL: URL {
        let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        return documentsDirectory
            .appending(component: "huggingface")
            .appending(component: "models")
    }
    
    @MainActor
    private init() {
        if !FileManager.default.fileExists(atPath: HuggingFaceDownloader.downloadsURL.path) {
            _ = try? FileManager.default.createDirectory(
                at: HuggingFaceDownloader.downloadsURL,
                withIntermediateDirectories: true
            )
        }
    }
}

extension HuggingFaceDownloader {
    
    public func downloadModel() async {
        await MainActor.run {
            ModelDownloadManager.shared.model.state = .downloading(progress: 0)
        }
        
        do {
            try await HuggingFaceDownloader.hub.snapshot(
                from: ModelDownloadManager.modelPath,
                progressHandler: { progress in
                    Task { @MainActor in
                        ModelDownloadManager.shared.model.state = .downloading(progress: Double(progress.fractionCompleted))
                    }
                }
            )
            
            await MainActor.run {
                ModelDownloadManager.shared.model.state = .downloaded
            }
        } catch {
            await MainActor.run {
                ModelDownloadManager.shared.model.state = .failed(error.localizedDescription)
            }
        }
    }
    
    public func isModelDownloaded() -> Bool {
        let modelFolder = HuggingFaceDownloader.downloadsURL
            .appending(component: ModelDownloadManager.modelPath)
                
        return FileManager.default.fileExists(atPath: modelFolder.path)
    }
    
    public func deleteDownloadedModel() async throws {
        let modelFolder = HuggingFaceDownloader.downloadsURL
                
        if FileManager.default.fileExists(atPath: modelFolder.path) {
            do {
                try FileManager.default.removeItem(at: modelFolder)
            } catch {
                throw error
            }
            
            await MainActor.run {
                ModelDownloadManager.shared.model.state = .notDownloaded
            }
        } else {
            print("Folder not found at: \(modelFolder.path)")
        }
    }
}


