//
//  UIImage.swift
//  Evaluator-Video-Test
//
//  Created by Sivamurugan Velmurugan on 1/28/26.
//

import UIKit
import CoreVideo
import CoreImage

func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer,
                          orientation: CGImagePropertyOrientation = .up) -> UIImage {
    // Apply the video track's orientation so portrait clips aren't sideways/upside down.
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
    let context = CIContext()
    let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
    return UIImage(cgImage: cgImage)
}

// Maps a video track's preferredTransform to the equivalent image orientation.
func imageOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
    switch (transform.a, transform.b, transform.c, transform.d) {
    case (0, 1, -1, 0):   return .right  // recorded in portrait
    case (0, -1, 1, 0):   return .left   // portrait upside down
    case (-1, 0, 0, -1):  return .down   // landscape, home button left
    default:              return .up     // landscape, home button right
    }
}

extension UIImage {
    func rotated(by degrees: CGFloat) -> UIImage? {
        // Convert degrees to radians
        let radians = degrees * .pi / 180.0
        
        // Calculate the size of the rotated view's containing box for the new drawing space
        let rotatedViewBox = UIView(frame: CGRect(origin: .zero, size: size))
        let t = CGAffineTransform(rotationAngle: radians)
        rotatedViewBox.transform = t
        let rotatedSize = rotatedViewBox.frame.size
        
        // Use UIGraphicsImageRenderer for modern context handling
        let renderer = UIGraphicsImageRenderer(size: rotatedSize)
        let newImage = renderer.image { context in
            // Move the origin to the middle of the new image so we rotate around the center
            context.cgContext.translateBy(x: rotatedSize.width / 2.0, y: rotatedSize.height / 2.0)
            
            // Rotate the context
            context.cgContext.rotate(by: radians)
            
            // Draw the original image into the context
            // It needs to be offset by half its size to center it correctly before rotation
            let rect = CGRect(x: -size.width / 2.0, y: -size.height / 2.0, width: size.width, height: size.height)
            context.cgContext.draw(self.cgImage!, in: rect)
        }
        
        return newImage
    }
}

