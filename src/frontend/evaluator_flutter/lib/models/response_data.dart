import 'point.dart';

class ResponseData {
  final Map<String, Point> points;

  ResponseData({required this.points});

  factory ResponseData.fromJson(Map<String, dynamic> json) {
    final mappedPoints = json.map((key, value) => MapEntry(
        key, Point.fromJson(value is Map<String, dynamic> ? value : {})));
    return ResponseData(points: mappedPoints);
  }
}
