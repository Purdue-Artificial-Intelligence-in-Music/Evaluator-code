import ExpoModulesCore

public class ExpoVideoAnalyzerModule: Module {
    public func definition() -> ModuleDefinition {
        Name("ExpoVideoAnalyzer")
        
        Function("getStatus") { () -> String in
            "Video Analyzer Module Connected"
        }
        
        AsyncFunction("analyzeVideo") { (videoUri: String) -> [String: Any] in
            return [
                "status": "analysis_complete",
                "videoUri": videoUri,
                "result": "Mock analysis result",
                "timestamp": Date().timeIntervalSince1970 * 1000
            ]
        }
    }
}