import CoreImage
import Metal
import MetalKit

/// VHS風フィルター
public final class VHSFilter: CIFilter {
    @objc public var inputImage: CIImage?
    
    /// 色ズレ
    @objc public var inputShift: CGFloat = 5
    /// ぼかし
    @objc public var inputBlurSamples: CGFloat = 10
    /// 減色度
    @objc public var inputLevels: CGFloat = 16

    public override var outputImage: CIImage? {
        guard let inputImage else { return nil }
        do {
            return try VHSKernel.apply(
                withExtent: inputImage.extent, inputs: [inputImage],
                arguments: [
                    "shift": inputShift,
                    "blurSamples": inputBlurSamples,
                    "levels": inputLevels,
                ])
        } catch {
            print("VHSFilter error: \(error)")
            return nil
        }
    }
}

/// VHS用の CIImageProcessorKernel
internal final class VHSKernel: CIImageProcessorKernel {
    /// カーネルソース
    static let metalSource = """
        constant float3x3 rgb2yiq = float3x3(
            float3(0.299, 0.5959, 0.2115),  // Col 0 (R)
            float3(0.587, -0.2744, -0.5229), // Col 1 (G)
            float3(0.114, -0.3213, 0.3111)  // Col 2 (B)
        );
        
        constant float3x3 yiq2rgb = float3x3(
            float3(1.0, 1.0, 1.0),          // Col 0 (Y)
            float3(0.956, -0.272, -1.106),  // Col 1 (I)
            float3(0.621, -0.647, 1.703)   // Col 2 (Q)
        );

        kernel void vhs_color_under_kernel(
            texture2d<float, access::read> inTexture [[texture(0)]],
            texture2d<float, access::write> outTexture [[texture(1)]],
            constant int &shift [[buffer(0)]],
            constant int &blur_samples [[buffer(1)]],
            constant int &quantize_levels [[buffer(2)]],
            constant int2 &offset [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
                return;
            }

            // 出力座標に対応する入力テクスチャ上の座標 (ROI内の座標)
            uint2 centerCoords = gid + uint2(offset);
            // 輝度の抽出
            float4 centerColor = inTexture.read(centerCoords);
            float3 centerYIQ = rgb2yiq * centerColor.rgb;
            float y = centerYIQ.r;

            // 色差の抽出と水平方向のぼかし（色解像度低下と色ズレ）
            float3 resultYIQ;
            if (blur_samples > 0) {
                float i_sum = 0.0;
                float q_sum = 0.0;
                for (int dx = 0; dx < blur_samples; dx++) {
                    uint2 sampleCoords = uint2(max(0, int(centerCoords.x) - shift - dx), centerCoords.y);
                    float4 sampleColor = inTexture.read(sampleCoords);
                    float3 sampleYIQ = rgb2yiq * sampleColor.rgb;
                    i_sum += sampleYIQ.g;
                    q_sum += sampleYIQ.b;
                }

                float i = i_sum / float(blur_samples);
                float q = q_sum / float(blur_samples);

                if (quantize_levels > 0) {
                    i = round(i * float(quantize_levels)) / float(quantize_levels);
                    q = round(q * float(quantize_levels)) / float(quantize_levels);
                }
                resultYIQ = float3(y, i, q);
            } else {
                resultYIQ = centerYIQ;
            }

            // RGBへ再変換
            float3 finalRGB = yiq2rgb * resultYIQ;
            outTexture.write(float4(finalRGB, centerColor.a), gid);
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

        let shift = (arguments?["shift"] as? CGFloat).map { Int32($0) } ?? 5
        let blurSamples = (arguments?["blurSamples"] as? CGFloat).map { Int32($0) } ?? 10
        let levels = (arguments?["levels"] as? CGFloat).map { Int32($0) } ?? 16

        let pipelineState = try MetalKernelManager.shared.pipelineState(
            source: metalSource,
            kernelName: "vhs_color_under_kernel"
        )

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(sourceTexture, index: 0)
        computeEncoder.setTexture(destinationTexture, index: 1)

        var shiftValue = shift
        var blurValue = blurSamples
        var levelsValue = levels
        computeEncoder.setBytes(&shiftValue, length: MemoryLayout<Int32>.size, index: 0)
        computeEncoder.setBytes(&blurValue, length: MemoryLayout<Int32>.size, index: 1)
        computeEncoder.setBytes(&levelsValue, length: MemoryLayout<Int32>.size, index: 2)

        // ROIの拡大分（オフセット）を計算して渡す
        let totalOffset = Int32(abs(Double(shift)) + abs(Double(blurSamples)))
        var offsetValue = SIMD2<Int32>(totalOffset, 0)
        computeEncoder.setBytes(&offsetValue, length: MemoryLayout<SIMD2<Int32>>.size, index: 3)

        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadgroupsPerGrid = MTLSizeMake(
            (destinationTexture.width + w - 1) / w,
            (destinationTexture.height + h - 1) / h,
            1
        )

        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }

    override class var outputFormat: CIFormat {
        return .BGRA8
    }

    override class func roi(
        forInput inputIndex: Int32, arguments: [String: Any]?, outputRect: CGRect
    ) -> CGRect {
        let shift = (arguments?["shift"] as? CGFloat) ?? 5
        let blurSamples = (arguments?["blurSamples"] as? CGFloat) ?? 10
        let totalOffset = abs(shift) + abs(blurSamples)
        return outputRect.insetBy(dx: -totalOffset, dy: 0)
    }
}
