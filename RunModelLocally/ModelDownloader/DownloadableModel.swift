//
//  DownloadableModel.swift
//  RunModelLocally
//
//  Created by Natasha Murashev on 11/20/24.
//

import Foundation

public struct DownloadableModel: Hashable, Identifiable, Sendable {
    
    public typealias ID = String
    
    public var name: String
    public var url: URL?
    public var state: DownloadState
    
    
    public var id: ID {
        name
    }
    
    public var displayName: String {
        url?.lastPathComponent ?? name
    }
    
    public enum DownloadState: Hashable, Sendable {
        case notDownloaded
        case downloading(progress: Double)
        case downloaded
        case failed(String)
    }
}
