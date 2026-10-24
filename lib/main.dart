import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const HandSignApp());
}

class HandSignApp extends StatelessWidget {
  const HandSignApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hand Sign Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const RecognitionPage(),
    );
  }
}

class RecognitionPage extends StatefulWidget {
  const RecognitionPage({super.key});

  @override
  State<RecognitionPage> createState() => _RecognitionPageState();
}

class _RecognitionPageState extends State<RecognitionPage> {
  static const int inputSize = 224;
  static const Duration scanInterval = Duration(milliseconds: 800);
  

  CameraController? _camera;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;

  HandLandmarkerPlugin? _landmarker;
  StreamSubscription<List<Hand>>? _landmarkSubscription;
  Timer? _scanTimer;

  Interpreter? _interpreter;
  List<String> _labels = [];
  List<Hand> _hands = [];
  CameraImage? _latestCameraImage;
  bool _signRecognized = false;

  bool _modelReady = false;
  bool _cameraReady = false;
  bool _recognitionBusy = false;
  bool _cameraSwitchBusy = false;

  String _prediction = 'Preparing scanner...';
  String _detail = 'Allow camera access and wait for the hand tracker.';

  String _diagnosticLabel = 'call_me';
bool _savingDiagnosticFrame = false;

static const List<String> _diagnosticLabels = [
  'call_me',
  'fist',
  'heart',
  'i_love_you',
  'open_palm',
  'peace',
  'pointing',
  'small',
  'thumbs_up',
];


Future<void> _saveDiagnosticFrame() async {
  if (_savingDiagnosticFrame) return;

  final camera = _camera;
  final frame = _latestCameraImage;

  if (camera == null ||
      !camera.value.isInitialized ||
      frame == null ||
      frame.planes.length < 3) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Camera frame is not ready yet.')),
    );
    return;
  }

  setState(() => _savingDiagnosticFrame = true);

  try {
    // Match the current preprocessing used by recognition.
    final rgb = _cameraImageToRgb(frame);
    final orientation = camera.description.sensorOrientation;
    final oriented = orientation == 0
        ? rgb
        : img.copyRotate(rgb, angle: orientation);

    final resized = img.copyResize(
      oriented,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    final directory = await getExternalStorageDirectory();
    if (directory == null) {
      throw Exception('Could not access app storage.');
    }

    final testDirectory = Directory(
      '${directory.path}${Platform.pathSeparator}phone_test',
    );
    await testDirectory.create(recursive: true);

    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final filename = '${_diagnosticLabel}_$timestamp.jpg';
    final file = File(
      '${testDirectory.path}${Platform.pathSeparator}$filename',
    );

    await file.writeAsBytes(img.encodeJpg(resized, quality: 95));

    debugPrint('DIAGNOSTIC IMAGE SAVED: ${file.path}');

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Saved $filename')),
    );
  } catch (e, stackTrace) {
    debugPrint('Diagnostic capture failed: $e');
    debugPrintStack(stackTrace: stackTrace);

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save frame: $e')),
      );
    }
  } finally {
    if (mounted) {
      setState(() => _savingDiagnosticFrame = false);
    }
  }
}


  String? _modelError;
  String? _cameraError;

  double? _confidence;

  // Only predictions at or above this threshold are treated as recognized signs.
  static const double _minimumConfidence = 60.0;
  static const String _historyStorageKey = 'handsign_detection_history_v1';
  final List<Map<String, dynamic>> _history = [];
  String? _lastSavedLabel;
  DateTime? _lastSavedAt;

  Future<void> _loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_historyStorageKey);
      if (raw == null) return;
      final decoded = jsonDecode(raw);
      if (decoded is! List) return;
      if (!mounted) return;
      setState(() {
        _history
          ..clear()
          ..addAll(decoded.whereType<Map>().map(
                (item) => Map<String, dynamic>.from(item),
              ));
      });
    } catch (e) {
      debugPrint('Could not load detection history: $e');
    }
  }

  Future<void> _saveHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_historyStorageKey, jsonEncode(_history));
    } catch (e) {
      debugPrint('Could not save detection history: $e');
    }
  }

  void _recordDetection(String label, double confidence) {
    final now = DateTime.now();

    // Avoid saving the same continuously detected sign every 10 seconds.
    if (_lastSavedLabel == label &&
        _lastSavedAt != null &&
        now.difference(_lastSavedAt!) < const Duration(seconds: 10)) {
      return;
    }

    _lastSavedLabel = label;
    _lastSavedAt = now;
    _history.insert(0, {
      'label': label,
      'confidence': confidence,
      'timestamp': now.toIso8601String(),
    });

    // Keep the latest 100 detections.
    if (_history.length > 100) {
      _history.removeRange(100, _history.length);
    }

    if (mounted) setState(() {});
    _saveHistory();
  }

  Future<void> _clearHistory() async {
    _history.clear();
    _lastSavedLabel = null;
    _lastSavedAt = null;
    if (mounted) setState(() {});
    await _saveHistory();
  }

  void _openHistory() {
    Navigator.of(context)
        .push(
      MaterialPageRoute<void>(
        builder: (_) => DetectionHistoryPage(
          records: List<Map<String, dynamic>>.from(_history),
          onClear: _clearHistory,
        ),
      ),
    )
        .then((_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadHistory();
    _initializeTrackerAndCamera();
  }

  // ------------------------------------------------------------
  // LOAD TENSORFLOW LITE MODEL
  // ------------------------------------------------------------

  Future<void> _loadModel() async {
    Interpreter? loadedInterpreter;

    try {
      final labelText = await rootBundle.loadString('assets/labels.txt');

      final labels = labelText
          .split(RegExp(r'\r?\n'))
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      if (labels.length != 9) {
        throw Exception(
          'Expected 9 labels in labels.txt, found ${labels.length}.',
        );
      }

      loadedInterpreter = await Interpreter.fromAsset('assets/model.tflite');

      final inputTensor = loadedInterpreter.getInputTensor(0);
      final outputTensor = loadedInterpreter.getOutputTensor(0);

      final inputShape = inputTensor.shape;
      final outputShape = outputTensor.shape;

      debugPrint('Model input shape: $inputShape');
      debugPrint('Model input type: ${inputTensor.type}');
      debugPrint('Model output shape: $outputShape');
      debugPrint('Model output type: ${outputTensor.type}');
      debugPrint('Labels: $labels');

      if (inputShape.length != 4 ||
          inputShape[0] != 1 ||
          inputShape[1] != inputSize ||
          inputShape[2] != inputSize ||
          inputShape[3] != 3) {
        throw Exception(
          'Expected model input [1, 224, 224, 3], '
          'but found $inputShape.',
        );
      }

      if (!inputTensor.type.toString().toLowerCase().contains('float32')) {
        throw Exception(
          'This file expects a float32 model input, '
          'but the model input type is ${inputTensor.type}.',
        );
      }

      if (outputShape.isEmpty || outputShape.last != labels.length) {
        throw Exception(
          'Model output $outputShape does not match '
          '${labels.length} labels. Use model.tflite and labels.txt '
          'from the same training run.',
        );
      }

      if (!mounted) {
        loadedInterpreter.close();
        return;
      }

      setState(() {
        _interpreter = loadedInterpreter;
        _labels = labels;
        _modelReady = true;
        _modelError = null;
        _prediction = 'Show your hand';
        _detail = 'Place one trained sign inside the guide box.';
      });

      loadedInterpreter = null;
    } catch (e, stackTrace) {
      loadedInterpreter?.close();

      debugPrint('Model loading error: $e');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) return;

      setState(() {
        _modelReady = false;
        _modelError = e.toString();
        _prediction = 'Model not ready';
        _detail = 'Check model.tflite and labels.txt.';
      });
    }
  }

  // ------------------------------------------------------------
  // INITIALIZE MEDIAPIPE AND CAMERA
  // ------------------------------------------------------------

  Future<void> _initializeTrackerAndCamera() async {
    try {
      final permission = await Permission.camera.request();

      if (!permission.isGranted) {
        throw Exception(
          'Camera permission denied. Allow camera access in Android settings.',
        );
      }

      _cameras = await availableCameras();

      if (_cameras.isEmpty) {
        throw Exception('No camera was found.');
      }

      final backCameraIndex = _cameras.indexWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
      );

      _cameraIndex = backCameraIndex >= 0 ? backCameraIndex : 0;

      _landmarker = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        delegate: HandLandmarkerDelegate.gpu,
      );

      _landmarkSubscription = _landmarker!.landmarkStream.listen(
        (hands) {
          if (!mounted) return;
          setState(() {
            _hands = hands;
          });
        },
        onError: (Object error) {
          debugPrint('Landmark stream error: $error');
        },
      );

      await _openCameraAt(_cameraIndex);
    } catch (e, stackTrace) {
      debugPrint('Camera initialization error: $e');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) return;

      setState(() {
        _cameraError = e.toString();
        _prediction = 'Scanner could not start';
        _detail = 'Check camera permission and MediaPipe setup.';
      });
    }
  }

  Future<void> _openCameraAt(int index) async {
    final controller = CameraController(
      _cameras[index],
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      _camera = controller;
      _cameraIndex = index;

      await controller.startImageStream(_processCameraImage);

      if (!mounted) return;

      setState(() {
        _cameraReady = true;
        _cameraError = null;
        _hands = [];
        _latestCameraImage = null;
        _signRecognized = false;
        _prediction = 'Looking for a hand...';
        _detail = 'Keep your hand inside the guide box.';
        _confidence = null;
      });

      _scanTimer?.cancel();

      _scanTimer = Timer.periodic(
        scanInterval,
        (_) => _recognizeAutomatically(),
      );
    } catch (e) {
      if (identical(_camera, controller)) {
        _camera = null;
      }

      await controller.dispose();
      rethrow;
    }
  }

  void _processCameraImage(CameraImage image) {
    final camera = _camera;
    final tracker = _landmarker;

    if (camera == null ||
        tracker == null ||
        !camera.value.isInitialized ||
        _cameraSwitchBusy) {
      return;
    }

    _latestCameraImage = image;

    try {
      tracker.processFrame(
        image,
        camera.description.sensorOrientation,
      );
    } catch (e) {
      debugPrint('MediaPipe frame error: $e');
    }
  }

  // ------------------------------------------------------------
  // RECOGNIZE HAND SIGN
  // ------------------------------------------------------------

  Future<void> _recognizeAutomatically() async {
    if (!mounted ||
        _recognitionBusy ||
        _cameraSwitchBusy ||
        !_modelReady ||
        _interpreter == null ||
        !_cameraReady) {
      return;
    }

    if (_hands.isEmpty) {
      setState(() {
        _signRecognized = false;
        _prediction = 'No hand detected';
        _detail = 'Move your hand into the guide box.';
        _confidence = null;
      });
      return;
    }

    final camera = _camera;
    final frame = _latestCameraImage;

    if (camera == null ||
        !camera.value.isInitialized ||
        frame == null ||
        frame.planes.length < 3) {
      return;
    }

    _recognitionBusy = true;

    try {
      final rgb = _cameraImageToRgb(frame);

final orientation = camera.description.sensorOrientation;
final oriented = orientation == 0
    ? rgb
    : img.copyRotate(rgb, angle: orientation);

final resized = img.copyResize(
  oriented,
  width: inputSize,
  height: inputSize,
  interpolation: img.Interpolation.linear,
);


      // Training model includes Rescaling(1/127.5, offset: -1), so pass
      // unnormalized RGB values from 0 to 255.
      final input = [
        List.generate(
          inputSize,
          (y) => List.generate(
            inputSize,
            (x) {
              final pixel = resized.getPixel(x, y);
              return <double>[
                pixel.r.toDouble(),
                pixel.g.toDouble(),
                pixel.b.toDouble(),
              ];
            },
          ),
        ),
      ];

      final output = [List<double>.filled(_labels.length, 0.0)];
      _interpreter!.run(input, output);
      final scores = output.first;

      if (scores.length != _labels.length) {
        throw Exception(
          'Model returned ${scores.length} predictions, '
          'but there are ${_labels.length} labels.',
        );
      }

      var bestIndex = 0;
      for (var i = 1; i < scores.length; i++) {
        if (scores[i] > scores[bestIndex]) bestIndex = i;
      }

      final rawScore = scores[bestIndex];
      if (!rawScore.isFinite) {
        throw Exception('The model returned an invalid prediction.');
      }

      final confidence = rawScore.clamp(0.0, 1.0) * 100.0;
      final label = _labels[bestIndex];
      final accepted = confidence >= _minimumConfidence;
      final displayLabel = label.replaceAll('_', ' ');

      debugPrint(
        'Prediction: $label | Confidence: '
        '${confidence.toStringAsFixed(2)}% | Accepted: $accepted',
      );

      if (!mounted) return;
      setState(() {
        _signRecognized = accepted;
        _prediction = accepted ? displayLabel : 'Uncertain — try again';
        _confidence = accepted ? confidence : null;
        _detail = accepted
            ? 'Sign recognized successfully! Hold the sign steady to confirm.'
            : 'Sign not confirmed. Improve lighting, center your hand, '
                'and hold the pose steady.';
      });

      if (accepted) _recordDetection(displayLabel, confidence);
    } catch (e, stackTrace) {
      debugPrint('Recognition error: $e');
      debugPrintStack(stackTrace: stackTrace);
      if (mounted) {
        setState(() {
          _signRecognized = false;
          _prediction = 'Could not recognize';
          _detail = e.toString();
          _confidence = null;
        });
      }
    } finally {
      _recognitionBusy = false;
    }
  }

  /// Crop a square region around all currently detected hand landmarks.
  ///
  /// A little padding keeps fingertips from being clipped. If no usable
  /// landmarks are available, the original image is returned as a fallback.


  /// Convert Android YUV_420_888 camera planes to an RGB image.
  /// Plane row/pixel strides are respected for different Android devices.
  img.Image _cameraImageToRgb(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final yPlane = cameraImage.planes[0];
    final uPlane = cameraImage.planes[1];
    final vPlane = cameraImage.planes[2];
    final rgb = img.Image(width: width, height: height);

    for (var y = 0; y < height; y++) {
      final yRow = y * yPlane.bytesPerRow;
      final uvRow = (y ~/ 2) * uPlane.bytesPerRow;
      final vRow = (y ~/ 2) * vPlane.bytesPerRow;
      for (var x = 0; x < width; x++) {
        final yy = yPlane.bytes[yRow + x];
        final uIndex = uvRow + (x ~/ 2) * (uPlane.bytesPerPixel ?? 1);
        final vIndex = vRow + (x ~/ 2) * (vPlane.bytesPerPixel ?? 1);
        final u = uPlane.bytes[uIndex] - 128;
        final v = vPlane.bytes[vIndex] - 128;
        final c = math.max(0, yy - 16);
        final r = ((298 * c + 409 * v + 128) >> 8).clamp(0, 255);
        final g = ((298 * c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255);
        final b = ((298 * c + 516 * u + 128) >> 8).clamp(0, 255);
        rgb.setPixelRgb(x, y, r, g, b);
      }
    }
    return rgb;
  }

  // ------------------------------------------------------------
  // SWITCH FRONT/BACK CAMERA
  // ------------------------------------------------------------

  Future<void> _switchCamera() async {
    if (_cameraSwitchBusy || _recognitionBusy || _cameras.length < 2) {
      return;
    }

    _cameraSwitchBusy = true;
    _scanTimer?.cancel();

    if (mounted) {
      setState(() {
        _cameraReady = false;
        _hands = [];
        _latestCameraImage = null;
        _signRecognized = false;
        _prediction = 'Switching camera...';
        _detail = 'Please wait.';
        _confidence = null;
      });
    }

    try {
      final oldCamera = _camera;
      _camera = null;

      if (oldCamera != null) {
        if (oldCamera.value.isStreamingImages) {
          await oldCamera.stopImageStream();
        }

        await oldCamera.dispose();
      }

      final nextIndex = (_cameraIndex + 1) % _cameras.length;

      await _openCameraAt(nextIndex);
    } catch (e) {
      debugPrint('Camera switch error: $e');

      if (mounted) {
        setState(() {
          _cameraError = e.toString();
          _prediction = 'Camera switch failed';
          _detail = 'Try switching again or restarting the app.';
        });
      }
    } finally {
      _cameraSwitchBusy = false;
    }
  }

  // ------------------------------------------------------------
  // DISPOSE RESOURCES
  // ------------------------------------------------------------

  @override
  void dispose() {
    _scanTimer?.cancel();
    _landmarkSubscription?.cancel();

    final camera = _camera;

    if (camera != null) {
      if (camera.value.isStreamingImages) {
        camera.stopImageStream();
      }

      camera.dispose();
    }

    _landmarker?.dispose();
    _interpreter?.close();

    super.dispose();
  }

  // ------------------------------------------------------------
  // USER INTERFACE
  // ------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    final previewSize = camera?.value.previewSize;
    final lens = camera?.description.lensDirection;

    final previewAspectRatio =
        previewSize == null ? 3 / 4 : previewSize.height / previewSize.width;

    final hasError = _modelError != null || _cameraError != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Hand Sign Recognition'),
        actions: [
          IconButton(
            onPressed: _openHistory,
            tooltip: 'Detection history',
            icon: const Icon(Icons.history_rounded),
          ),
          IconButton(
            onPressed:
                _cameraSwitchBusy || _recognitionBusy || _cameras.length < 2
                    ? null
                    : _switchCamera,
            tooltip: lens == CameraLensDirection.front
                ? 'Switch to back camera'
                : 'Switch to front camera',
            icon: const Icon(Icons.cameraswitch_rounded),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // CAMERA PREVIEW AND LANDMARKS
            ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: AspectRatio(
                aspectRatio: previewAspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (camera != null && camera.value.isInitialized)
                      CameraPreview(camera)
                    else
                      Container(
                        color: Colors.black87,
                        alignment: Alignment.center,
                        child: const CircularProgressIndicator(),
                      ),

                    if (camera != null &&
                        camera.value.isInitialized &&
                        previewSize != null &&
                        _landmarker != null)
                      CustomPaint(
                        painter: LandmarkPainter(
                          hands: _hands,
                          previewSize: previewSize,
                          lensDirection: camera.description.lensDirection,
                          sensorOrientation:
                              camera.description.sensorOrientation,
                        ),
                      ),

                    // HAND PLACEMENT GUIDE
                    IgnorePointer(
                      child: Center(
                        child: FractionallySizedBox(
                          widthFactor: 0.72,
                          heightFactor: 0.60,
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              border: Border.all(
                                color: _hands.isNotEmpty
                                    ? Colors.greenAccent
                                    : Colors.white,
                                width: 2.5,
                              ),
                              borderRadius: BorderRadius.circular(22),
                            ),
                            child: Align(
                              alignment: Alignment.topCenter,
                              child: Container(
                                margin: const EdgeInsets.only(top: 8),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 10,
                                  vertical: 5,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.55),
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  _hands.isNotEmpty
                                      ? 'HAND DETECTED'
                                      : 'PLACE HAND HERE',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.8,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),

                    if (_signRecognized)
                      Positioned(
                        top: 12,
                        left: 12,
                        right: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 11,
                          ),
                          decoration: BoxDecoration(
                            color:
                                Colors.green.shade700.withValues(alpha: 0.94),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.check_circle,
                                  color: Colors.white),
                              const SizedBox(width: 9),
                              Expanded(
                                child: Text(
                                  'SIGN RECOGNIZED: ${_prediction.toUpperCase()}',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 13,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),

                      
Card(
  child: Padding(
    padding: const EdgeInsets.all(12),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'Temporary model diagnostic',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          initialValue: _diagnosticLabel,
          decoration: const InputDecoration(
            labelText: 'Sign you are showing',
            border: OutlineInputBorder(),
          ),
          items: _diagnosticLabels.map((label) {
            return DropdownMenuItem(
              value: label,
              child: Text(label),
            );
          }).toList(),
          onChanged: _savingDiagnosticFrame
              ? null
              : (value) {
                  if (value != null) {
                    setState(() => _diagnosticLabel = value);
                  }
                },
        ),
        const SizedBox(height: 8),
        FilledButton.icon(
          onPressed: _savingDiagnosticFrame
              ? null
              : _saveDiagnosticFrame,
          icon: const Icon(Icons.save_alt),
          label: Text(
            _savingDiagnosticFrame
                ? 'Saving...'
                : 'Save test frame',
          ),
        ),
      ],
    ),
  ),
),


                    // TRACKING STATUS
                    Positioned(
                      left: 12,
                      bottom: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 7,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.6),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _hands.isNotEmpty
                                  ? Icons.track_changes
                                  : Icons.center_focus_strong,
                              color: _hands.isNotEmpty
                                  ? Colors.greenAccent
                                  : Colors.white,
                              size: 17,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              _hands.isNotEmpty
                                  ? 'MediaPipe tracking'
                                  : 'Searching for hand...',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // RECOGNITION RESULT
            Card(
              child: Padding(
                padding: const EdgeInsets.all(18),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'RECOGNIZED SIGN',
                      style: TextStyle(
                        color: Colors.black54,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _prediction,
                      style: const TextStyle(
                        fontSize: 27,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(_detail),
                    if (_signRecognized) ...[
                      const SizedBox(height: 12),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.green.withValues(alpha: 0.10),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.green.shade400),
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.check_circle,
                                color: Colors.green.shade700),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Recognized successfully: $_prediction',
                                style: TextStyle(
                                  color: Colors.green.shade800,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                    if (_confidence != null) ...[
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('Model confidence'),
                          Text(
                            '${_confidence!.toStringAsFixed(1)}%',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: LinearProgressIndicator(
                          value: (_confidence! / 100).clamp(0.0, 1.0),
                          minHeight: 8,
                        ),
                      ),
                    ],
                    if (hasError) ...[
                      const SizedBox(height: 12),
                      Text(
                        _modelError ?? _cameraError ?? 'Unknown error',
                        style: const TextStyle(
                          color: Colors.red,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            const Text(
              'Keep your whole hand inside the guide box. '
              'Use a clear, well-lit background and hold each sign '
              'steady while the app makes its prediction.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.black54),
            ),

          ],
        ),
      ),
    );
  }
}

// ------------------------------------------------------------
// MEDIAPIPE HAND LANDMARK PAINTER
// ------------------------------------------------------------

class LandmarkPainter extends CustomPainter {
  LandmarkPainter({
    required this.hands,
    required this.previewSize,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  final List<Hand> hands;
  final Size previewSize;
  final CameraLensDirection lensDirection;
  final int sensorOrientation;

  static const List<List<int>> connections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    final scale = size.width / previewSize.height;

    final pointPaint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = Colors.lightBlueAccent
      ..strokeWidth = 3 / scale
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    canvas.save();

    final center = Offset(size.width / 2, size.height / 2);

    canvas.translate(center.dx, center.dy);
    canvas.rotate(sensorOrientation * math.pi / 180);

    if (lensDirection == CameraLensDirection.front) {
      canvas.scale(-1, 1);
      canvas.rotate(math.pi);
    }

    canvas.scale(scale);

    final logicalWidth = previewSize.width;
    final logicalHeight = previewSize.height;

    Offset pointFor(Landmark landmark) {
      return Offset(
        (landmark.x - 0.5) * logicalWidth,
        (landmark.y - 0.5) * logicalHeight,
      );
    }

    for (final hand in hands) {
      for (final connection in connections) {
        if (connection[0] >= hand.landmarks.length ||
            connection[1] >= hand.landmarks.length) {
          continue;
        }

        canvas.drawLine(
          pointFor(hand.landmarks[connection[0]]),
          pointFor(hand.landmarks[connection[1]]),
          linePaint,
        );
      }

      for (final landmark in hand.landmarks) {
        canvas.drawCircle(
          pointFor(landmark),
          5 / scale,
          pointPaint,
        );
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant LandmarkPainter oldDelegate) {
    return oldDelegate.hands != hands ||
        oldDelegate.previewSize != previewSize ||
        oldDelegate.lensDirection != lensDirection ||
        oldDelegate.sensorOrientation != sensorOrientation;
  }
}

// ------------------------------------------------------------
// DETECTION HISTORY
// ------------------------------------------------------------

class DetectionHistoryPage extends StatelessWidget {
  const DetectionHistoryPage({
    super.key,
    required this.records,
    required this.onClear,
  });

  final List<Map<String, dynamic>> records;
  final Future<void> Function() onClear;

  String _formatDate(String? value) {
    if (value == null) return 'Unknown date';
    final date = DateTime.tryParse(value)?.toLocal();
    if (date == null) return 'Unknown date';
    final hour = date.hour % 12 == 0 ? 12 : date.hour % 12;
    final minute = date.minute.toString().padLeft(2, '0');
    final period = date.hour >= 12 ? 'PM' : 'AM';
    return '${date.month}/${date.day}/${date.year} • $hour:$minute $period';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection History'),
        actions: [
          IconButton(
            tooltip: 'Clear history',
            icon: const Icon(Icons.delete_outline_rounded),
            onPressed: records.isEmpty
                ? null
                : () async {
                    final confirmed = await showDialog<bool>(
                      context: context,
                      builder: (dialogContext) => AlertDialog(
                        title: const Text('Clear detection history?'),
                        content: const Text(
                          'This will permanently remove all saved detections.',
                        ),
                        actions: [
                          TextButton(
                            onPressed: () =>
                                Navigator.pop(dialogContext, false),
                            child: const Text('Cancel'),
                          ),
                          FilledButton(
                            onPressed: () => Navigator.pop(dialogContext, true),
                            child: const Text('Clear'),
                          ),
                        ],
                      ),
                    );
                    if (confirmed == true) {
                      await onClear();
                      if (context.mounted) {
                        Navigator.pop(context);
                      }
                    }
                  },
          ),
        ],
      ),
      body: records.isEmpty
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(28),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.history_rounded,
                      size: 64,
                      color: Theme.of(context).colorScheme.outline,
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'No detections yet',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Recognized signs with sufficient confidence will '
                      'appear here automatically.',
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: records.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, index) {
                final record = records[index];
                final label = record['label']?.toString() ?? 'Unknown sign';
                final confidenceValue = record['confidence'];
                final confidence =
                    confidenceValue is num ? confidenceValue.toDouble() : 0.0;
                return Card(
                  child: ListTile(
                    leading: const CircleAvatar(
                      child: Icon(Icons.front_hand_rounded),
                    ),
                    title: Text(
                      label,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    subtitle:
                        Text(_formatDate(record['timestamp']?.toString())),
                    trailing: Text(
                      '${confidence.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: confidence >= 80
                            ? Colors.green
                            : Theme.of(context).colorScheme.primary,
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }
}
