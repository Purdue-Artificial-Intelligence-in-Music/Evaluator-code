import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/response_data.dart';

class ApiService {
  static const String baseUrl = 'http://10.0.2.2:8000';

  static Future<ResponseData> uploadImage(String base64Image) async {
    print(base64Image);
    final response = await http.post(
      Uri.parse('$baseUrl/upload/'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
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
