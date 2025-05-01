import 'package:flutter/material.dart';
import '../models/point.dart';
import '../models/line.dart';

class DrawingCanvas extends StatelessWidget {
  final List<Point> points;
  final List<Line> lines;

  const DrawingCanvas({required this.points, required this.lines, super.key});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _CanvasPainter(points: points, lines: lines),
      child: Container(),
    );
  }
}

class _CanvasPainter extends CustomPainter {
  final List<Point> points;
  final List<Line> lines;

  _CanvasPainter({required this.points, required this.lines});

  @override
  void paint(Canvas canvas, Size size) {
    final paintCircle = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.fill;

    final paintLine = Paint()
      ..color = Colors.red
      ..strokeWidth = 1;

    for (int i = 0; i < points.length; i++) {
      final p = points[i];
      paintCircle.color =
          Color.fromARGB(255, 255 - i * 30, i * 30, 255 - i * 30);
      canvas.drawCircle(Offset(p.x, p.y), 10, paintCircle);
    }

    for (var line in lines) {
      canvas.drawLine(
        Offset(line.start.x, line.start.y),
        Offset(line.end.x, line.end.y),
        paintLine,
      );
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
