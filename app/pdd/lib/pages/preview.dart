import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:pdd/pages/result.dart';
import 'package:pdd/protocol.dart';
// import 'package:susya/widgets/login_button.dart';

class ImagePreview extends StatefulWidget {
  final String imagePath;

  const ImagePreview({Key? key, required this.imagePath}) : super(key: key);

  @override
  _ImagePreviewState createState() => _ImagePreviewState();
}

class _ImagePreviewState extends State<ImagePreview> {
  bool isLoading = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Preview Generated"),
        backgroundColor: Colors.green,
      ),
      body: Row(
        children: [
          Expanded(
            flex: 1,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Spacer(
                  flex: 2,
                ),
                Image.file(File(widget.imagePath),
                    height: 300, fit: BoxFit.cover),
                Spacer(
                  flex: 1,
                ),
                ElevatedButton(
                  child: Text('Send'),
                  onPressed: () async {
                    setState(() {
                      isLoading = true;
                    });
                    var result = await sendToPredictor(widget.imagePath);
                    final String plant = result['plant'];
                    final String disease = result['disease'];
                    final String remedy = result['remedy'];
                    print(result);
                    setState(() {
                      isLoading = false;
                    });
                    Get.to(() => ResultPage(
                          disease: disease,
                          plant: plant,
                          remedy: remedy,
                        ));
                  },
                  style: ButtonStyle(
                    backgroundColor:
                        MaterialStateProperty.all<Color>(Colors.green),
                  ),
                ),
                isLoading
                    ? Expanded(
                        flex: 2,
                        child: Center(child: CircularProgressIndicator()))
                    : Spacer(flex: 2),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
