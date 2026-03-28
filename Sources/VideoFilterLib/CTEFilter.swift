import CoreImage
import Metal
import MetalKit
import MetalPerformanceShaders
import simd

/// CTE (Color Temperature Enhancement) フィルター
public final class CTEFilter: CIFilter {
    @objc public var inputImage: CIImage?

    /// 適用強度
    @objc public var inputIntensity: CGFloat = 0.5

    public override var outputImage: CIImage? {
        guard let inputImage else { return nil }
        do {
            return try CTEKernel.apply(
                withExtent: inputImage.extent, inputs: [inputImage],
                arguments: [
                    "intensity": inputIntensity
                ])
        } catch {
            print("CTEFilter error: \(error)")
            return nil
        }
    }
}

/// CTE (Color Temperature Enhancement) 用の CIImageProcessorKernel
internal final class CTEKernel: CIImageProcessorKernel {
    /// カーネルソース
    static let metalSource = """
        kernel void cte_kernel(
            texture2d<float, access::read> inTexture [[texture(0)]],
            texture2d<float, access::write> outTexture [[texture(1)]],
            texture2d<float, access::read> avgTexture [[texture(2)]],
            constant float &intensity [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
                return;
            }

            // 1x1テクスチャから平均色を取得
            float4 avgColor = avgTexture.read(uint2(0, 0));
            float3 avgRGB = avgColor.rgb;
            
            // バランスの計算
            float luma = (avgRGB.x + avgRGB.y + avgRGB.z) / 3.0;
            float3 targetBalance = float3(
                luma > 0 ? (avgRGB.x / luma) : 1.0,
                luma > 0 ? (avgRGB.y / luma) : 1.0,
                luma > 0 ? (avgRGB.z / luma) : 1.0
            );
            float3 balance = mix(float3(1.0, 1.0, 1.0), targetBalance, intensity);

            // フィルタの適用
            float4 inColor = inTexture.read(gid);
            float3 rgb = inColor.rgb * balance;

            outTexture.write(float4(rgb, inColor.a), gid);
        }
        """

    override class func process(
        with inputs: [CIImageProcessorInput]?, arguments: [String: Any]?,
        output: CIImageProcessorOutput
    ) throws {
        guard let input = inputs?.first else {
            print("kernel failed: inputs is nil")
            return
        }
        guard let sourceTexture = input.metalTexture else {
            print("kernel failed: sourceTexture is nil")
            return
        }
        guard let destinationTexture = output.metalTexture else {
            print("kernel failed: destinationTexture is nil")
            return
        }
        guard let commandBuffer = output.metalCommandBuffer else {
            print("kernel failed: commandBuffer is nil")
            return
        }

        let device = MetalKernelManager.shared.device
        let intensity = Float((arguments?["intensity"] as? CGFloat) ?? 0.5)

        // 1. 平均色の計算 (MPSを利用)
        let meanFilter = MPSImageStatisticsMean(device: device)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: 1,
            height: 1,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        guard let meanTexture = device.makeTexture(descriptor: textureDescriptor) else {
            return
        }

        meanFilter.encode(
            commandBuffer: commandBuffer, sourceTexture: sourceTexture,
            destinationTexture: meanTexture)

        // 2. カーネルの実行
        let pipelineState = try MetalKernelManager.shared.pipelineState(
            source: metalSource,
            kernelName: "cte_kernel"
        )
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(sourceTexture, index: 0)
        computeEncoder.setTexture(destinationTexture, index: 1)
        computeEncoder.setTexture(meanTexture, index: 2)

        var intensityValue = intensity
        computeEncoder.setBytes(&intensityValue, length: MemoryLayout<Float>.size, index: 0)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadgroupsPerGrid = MTLSizeMake(
            (destinationTexture.width + w - 1) / w,
            (destinationTexture.height + h - 1) / h,
            1
        )

        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
}
