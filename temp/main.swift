//
//  main.swift
//  evaluator_test
//
//  Created by Jackson Shields on 11/3/25.
//
import UIKit

var detector = try Detector()
if let myImage = UIImage(named: "Sample Input") {
    // Successfully loaded the image
    for i in 1...100 {
        let start = CFAbsoluteTimeGetCurrent()
        let result = detector.detect(frame: myImage)
        let diff = CFAbsoluteTimeGetCurrent() - start
        print("Took \(diff) seconds")
    }
} else {
    // Image not found in the Assets Catalog
    print("Error: Image 'MyImageName' not found.")
}

    
