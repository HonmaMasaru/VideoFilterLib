# VideoFilterLib

VideoFilterLibは、Metalを使用した高性能な画像・ビデオフィルタライブラリです。VHS風、テクニカラー風（2-Strip/3-Strip）、銀残し、CTEなどのエフェクトを簡単に適用できます。

## 特徴

- **Metal Compute Shader**: 高速な画像処理を実現。
- **多様なフィルタ**:
    - **VHSFilter**: VHSテープのようなレトロな質感、色ズレ、ノイズ。
    - **TwoStripFilter**: 初期カラー映画のような2色式テクニカラー風。
    - **ThreeStripFilter**: 鮮明な3色式テクニカラー風。
    - **CTEFilter**: 色温度強調（Color Temperature Enhancement）。
    - **BreachBypassFilter**: コントラストが高く彩度が低い「銀残し」エフェクト。
- **Core Image準拠**: すべてのフィルタが `CIFilter` を継承しており、標準的な Core Image パイプラインで利用可能。

## 動作環境

- macOS 11.0以上
- iOS 14.0以上
- Metalをサポートするデバイス

## インストール方法

Swift Package Managerを使用して、プロジェクトの依存関係に追加します。

```swift
dependencies: [
    .package(url: "https://your-repository-url/VideoFilterLib.git", from: "0.1.0")
]
```

## 使い方

すべてのフィルタは `CIFilter` を継承しており、`inputImage` をセットして `outputImage` を取得する、Core Image の標準的な記法で利用できます。

```swift
import VideoFilterLib
import CoreImage

// VHSフィルタ
let vhsFilter = VHSFilter()
vhsFilter.inputImage = inputCIImage
vhsFilter.inputShift = 5         // 色ズレの量
vhsFilter.inputBlurSamples = 10  // ぼかしの強さ
vhsFilter.inputLevels = 16       // 減色レベル
let vhsImage = vhsFilter.outputImage

// 銀残し (Breach Bypass)
let bbFilter = BreachBypassFilter()
bbFilter.inputImage = inputCIImage
bbFilter.inputIntensity = 0.8
let result = bbFilter.outputImage

// 2-Strip テクニカラー風
let twoStrip = TwoStripFilter()
twoStrip.inputImage = inputCIImage
let result2 = twoStrip.outputImage

// CTE (Color Temperature Enhancement)
let cteFilter = CTEFilter()
cteFilter.inputImage = inputCIImage
cteFilter.inputIntensity = 0.5
let result3 = cteFilter.outputImage
```

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
