import 'package:flutter/material.dart';
import '../services/camera_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraService _cameraService;

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
              Positioned(
                bottom: 20,
                left: 20,
                child: FloatingActionButton(
                  onPressed: _cameraService.takePicture,
                  child: const Icon(Icons.camera_alt),
                ),
              ),
            ],
          )
        : const Center(child: CircularProgressIndicator());
  }
}
