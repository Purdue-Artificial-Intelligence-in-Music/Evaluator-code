import 'package:image_picker/image_picker.dart';

class ImagePickerService {
  static final ImagePicker _picker = ImagePicker();

  static Future<String?> pickVideo() async {
    final result = await _picker.pickVideo(source: ImageSource.gallery);
    return result?.path;
  }
}
