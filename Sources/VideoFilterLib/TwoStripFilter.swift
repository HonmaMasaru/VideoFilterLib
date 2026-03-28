import CoreImage
import Metal
import MetalKit

/// Two-Strip (2ストリップ) フィルター
public final class TwoStripFilter: CIFilter {
    @objc public var inputImage: CIImage?

    public override var outputImage: CIImage? {
        guard let inputImage else { return nil }
        do {
            return try TwoStripKernel.apply(
                withExtent: inputImage.extent, inputs: [inputImage],
                arguments: nil)
        } catch {
            print("TwoStripFilter error: \(error)")
            return nil
        }
    }
}

/// Two-Strip (2ストリップ) 用の CIImageProcessorKernel
internal final class TwoStripKernel: CIImageProcessorKernel {
    /// カーネルソース
    static let metalSource = """
        kernel void two_strip_kernel(
            texture2d<float, access::read> inTexture [[texture(0)]],
            texture2d<float, access::write> outTexture [[texture(1)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
                return;
            }

            float4 inColor = inTexture.read(gid);
            float3 rgb = inColor.rgb;

            float c = to_gray(rgb, float3(0.0, kLuminanceWeights.g, kLuminanceWeights.b));
            float m = to_gray(rgb, float3(kLuminanceWeights.r, 0.0, kLuminanceWeights.b));
            float y = to_gray(rgb, float3(kLuminanceWeights.r, kLuminanceWeights.g, 0.0));

            float3 mask = 1.0 - clamp(rgb - float3(c, m, y), 0.0, 1.0);

            float3 processedRgb;
            processedRgb.r = clamp(rgb.r - (1.0 - (mask.g * mask.b)), 0.0, 1.0);
            processedRgb.g = clamp(rgb.g - (1.0 - (mask.r * mask.b)), 0.0, 1.0);
            processedRgb.b = clamp(rgb.b - (1.0 - (mask.r * mask.g)), 0.0, 1.0);

            float l = to_gray(processedRgb, kLuminanceWeights);
            float finalR = processedRgb.r;
            float finalG = clamp(l, 0.0, 1.0);
            float finalB = processedRgb.g;

            outTexture.write(float4(finalR, finalG, finalB, inColor.a), gid);
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

        let pipelineState = try MetalKernelManager.shared.pipelineState(
            source: metalSource,
            kernelName: "two_strip_kernel"
        )

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(sourceTexture, index: 0)
        computeEncoder.setTexture(destinationTexture, index: 1)

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
