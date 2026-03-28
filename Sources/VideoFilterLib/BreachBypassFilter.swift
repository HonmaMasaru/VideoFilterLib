import CoreImage
import Metal
import MetalKit

/// 銀残し (Breach Bypass) フィルター
public final class BreachBypassFilter: CIFilter {
    @objc public var inputImage: CIImage?

    /// 適用強度
    @objc public var inputIntensity: CGFloat = 0.8

    public override var outputImage: CIImage? {
        guard let inputImage else { return nil }
        do {
            return try BreachBypassKernel.apply(
                withExtent: inputImage.extent, inputs: [inputImage],
                arguments: [
                    "intensity": inputIntensity
                ])
        } catch {
            print("BreachBypassFilter error: \(error)")
            return nil
        }
    }
}

/// 銀残し (Breach Bypass) 用の CIImageProcessorKernel
internal final class BreachBypassKernel: CIImageProcessorKernel {
    /// カーネルソース
    static let metalSource = """
        kernel void breach_bypass_kernel(
            texture2d<float, access::read> inTexture [[texture(0)]],
            texture2d<float, access::write> outTexture [[texture(1)]],
            constant float &intensity [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
                return;
            }

            float4 inColor = inTexture.read(gid);
            float3 rgb = inColor.rgb;

            // 輝度の計算 (Luminance)
            float luma = dot(rgb, float3(0.2126, 0.7152, 0.0722));
            float3 grayscale = float3(luma);

            // オーバーレイ合成 (Overlay blend mode)
            float3 result;
            for (int i = 0; i < 3; i++) {
                if (rgb[i] < 0.5) {
                    result[i] = 2.0 * rgb[i] * grayscale[i];
                } else {
                    result[i] = 1.0 - 2.0 * (1.0 - rgb[i]) * (1.0 - grayscale[i]);
                }
            }

            // 強度の適用
            float3 finalRGB = mix(rgb, result, intensity);

            outTexture.write(float4(finalRGB, inColor.a), gid);
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

        let intensity = (arguments?["intensity"] as? CGFloat).map { Float($0) } ?? 0.8

        let pipelineState = try MetalKernelManager.shared.pipelineState(
            source: metalSource,
            kernelName: "breach_bypass_kernel"
        )

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(sourceTexture, index: 0)
        computeEncoder.setTexture(destinationTexture, index: 1)

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

    override class var outputFormat: CIFormat {
        return .BGRA8
    }
}
