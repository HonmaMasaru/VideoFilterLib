import Foundation
import Metal

/// Metalのコンピュートパイプライン状態を管理するマネージャー
internal final class MetalKernelManager: @unchecked Sendable {
    /// シングルトン
    static let shared: MetalKernelManager = .init()

    /// エラー
    enum Error: Swift.Error {
        case libraryNotFound
        case kernelNotFound(String)
    }

    /// デバイス
    let device: MTLDevice
    /// ライブラリ
    private var library: MTLLibrary?
    /// パイプラインのキャッシュ
    private var pipelineStates: [String: MTLComputePipelineState] = [:]
    /// ロック
    private let lock: NSLock = .init()

    /// 各カーネルで共通して使用するMetalのヘッダー部分
    static let sharedHeader = """
        #include <metal_stdlib>
        using namespace metal;
        
        constant float3 kLuminanceWeights = float3(0.30, 0.59, 0.11);
        
        float to_gray(float3 rgb, float3 weights) {
            float sum = weights.r + weights.g + weights.b;
            return dot(rgb, weights / sum);
        }
        
        """

    /// 初期化
    private init() {
        self.device = MTLCreateSystemDefaultDevice()!
    }

    /// ソース文字列からパイプライン状態を取得（キャッシュ機能付き）
    /// - Parameters:
    ///   - source: Metalソースコード（共通ヘッダーは自動的に付与されます）
    ///   - kernelName: カーネル名
    /// - Returns: コンピュートパイプライン状態
    func pipelineState(source: String, kernelName: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        // キャッシュチェック
        if let state = pipelineStates[kernelName] {
            return state
        }
        // ソースの結合
        let fullSource = Self.sharedHeader + "\n" + source
        // コンパイル
        let options = MTLCompileOptions()
        let library = try device.makeLibrary(source: fullSource, options: options)
        guard let function = library.makeFunction(name: kernelName) else {
            throw Error.kernelNotFound(kernelName)
        }
        let state = try device.makeComputePipelineState(function: function)
        pipelineStates[kernelName] = state
        return state
    }
}

// MARK: -

extension MetalKernelManager.Error: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .libraryNotFound:
            "Default Metal library not found. Make sure .metal files are included in the target."
        case .kernelNotFound(let kernelName):
            "Kernel not found: \(kernelName)"
        }
    }
}

