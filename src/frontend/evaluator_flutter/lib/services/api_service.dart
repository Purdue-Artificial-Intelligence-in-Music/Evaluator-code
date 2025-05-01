import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/response_data.dart';

class ApiService {
  static const String baseUrl = 'http://127.0.0.1:8000';

  static Future<ResponseData> uploadImage(String base64Image) async {
    final response = await http.post(
      Uri.parse('$baseUrl/upload/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'title': 'Test Image',
        'content': 'This is a test image',
        'image': base64Image,
      }),
    );

    if (response.statusCode == 200) {
      return ResponseData.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to upload image');
    }
  }

  static Future<void> uploadVideoDemo() async {
    await http.post(
      Uri.parse('$baseUrl/api/upload/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'title': 'demo video',
        'videouri': 'this is a test',
      }),
    );
  }
}
