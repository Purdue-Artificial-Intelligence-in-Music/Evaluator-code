class Point {
  final double x;
  final double y;

  Point({required this.x, required this.y});

  factory Point.fromJson(Map<String, dynamic> json) {
    return Point(
      x: (json['x'] ?? 0).toDouble(),
      y: (json['y'] ?? 0).toDouble(),
    );
  }
}
