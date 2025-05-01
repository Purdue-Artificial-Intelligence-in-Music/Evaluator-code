import 'package:flutter/material.dart';
import '../widgets/video_player_widget.dart';
import '../screens/camera_screen.dart';
import '../services/image_picker_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? _videoPath;
  bool _isCameraOpen = false;

  void _pickVideo() async {
    final path = await ImagePickerService.pickVideo();
    if (path != null) {
      setState(() {
        _videoPath = path;
        _isCameraOpen = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Home')),
      body: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                  onPressed: _pickVideo, child: const Text("Pick Video")),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _isCameraOpen = !_isCameraOpen;
                    _videoPath = null;
                  });
                },
                child: Text(_isCameraOpen ? 'Close Camera' : 'Open Camera'),
              ),
            ],
          ),
          Expanded(
            child: _isCameraOpen
                ? const CameraScreen()
                : _videoPath != null
                    ? VideoPlayerWidget(videoPath: _videoPath!)
                    : const Center(child: Text("No video selected")),
          ),
        ],
      ),
    );
  }
}
