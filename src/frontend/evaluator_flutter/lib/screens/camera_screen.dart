import 'package:flutter/material.dart';
import '../services/camera_service.dart';
import '../widgets/drawing_canvas.dart';
import '../models/point.dart';
import '../models/line.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraService _cameraService;
  List<Point> _points = [
    Point(x: 0, y: 0),
    Point(x: 100, y: 100),
    Point(x: 2, y: 2),
    Point(x: 3, y: 3),
    Point(x: 4, y: 4),
    Point(x: 5, y: 5),
    Point(x: 6, y: 6),
    Point(x: 7, y: 7),
  ];
  List<Line> _lines = [];

  @override
  void initState() {
    super.initState();
    _cameraService = CameraService();
    _cameraService.initialize().then((_) => setState(() {}));
  }

  @override
  void dispose() {
    _cameraService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _cameraService.isInitialized
        ? Stack(
            children: [
              _cameraService.buildPreview(),
              DrawingCanvas(points: _points, lines: _lines),
              Positioned(
                bottom: 20,
                left: 20,
                child: FloatingActionButton(
                  onPressed: () async {
                    List<Point> newPoints = await _cameraService.takePicture();
                    print('points$newPoints');
                    setState(() {
                      _points = [
                        Point(x: 0, y: 0),
                        Point(x: 100, y: 100),
                        Point(x: 2, y: 2),
                        Point(x: 3, y: 3),
                        Point(x: 400, y: 400),
                        Point(x: 5, y: 5),
                        Point(x: 6, y: 6),
                        Point(x: 7, y: 7),
                      ];
                    });
                  },
                  child: const Icon(Icons.camera_alt),
                ),
              ),
            ],
          )
        : const Center(child: CircularProgressIndicator());
  }
}
