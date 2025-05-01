import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CameraService {
  late CameraController _controller;
  bool isInitialized = false;

  Future<void> initialize() async {
    final cameras = await availableCameras();
    _controller = CameraController(cameras.first, ResolutionPreset.medium);
    await _controller.initialize();
    isInitialized = true;
  }

  Widget buildPreview() {
    return CameraPreview(_controller);
  }

  Future<void> takePicture() async {
    final picture = await _controller.takePicture();
    // TODO: Send picture to backend
  }

  void dispose() {
    _controller.dispose();
  }
}
