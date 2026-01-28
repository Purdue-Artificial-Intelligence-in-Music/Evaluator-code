//
//  UIImage.swift
//  Evaluator-Video-Test
//
//  Created by Sivamurugan Velmurugan on 1/28/26.
//

import UIKit
import CoreVideo
import CoreImage

func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer) -> UIImage {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
    return UIImage(cgImage: cgImage)
}
