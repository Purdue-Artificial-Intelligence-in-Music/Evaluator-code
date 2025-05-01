import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:namer_app/widgets/drawing_canvas.dart';
import 'dart:convert';
import 'api_service.dart';
import '../models/response_data.dart';
import '../models/point.dart';
import 'dart:typed_data';

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

  Future<List<Point>> takePicture() async {
    await _controller.setFocusMode(FocusMode.auto);

    XFile picture = await _controller.takePicture();
    Uint8List bytes = await picture.readAsBytes();
    print('bytes $bytes');
    String base64Image = base64Encode(bytes);
    ResponseData responseData = await ApiService.uploadImage(base64Image);

    // assumes that responseDara.pointsis a list
    return responseData.points as List<Point>;
  }

  void dispose() {
    _controller.dispose();
  }
}
