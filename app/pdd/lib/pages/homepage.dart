import 'package:flutter/material.dart';

import 'package:get/get.dart';
import 'package:pdd/pages/camera_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  HomePageState createState() => HomePageState();
}

class HomePageState extends State<HomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.green[700],
        title: Text("Plant Disease Detection (PDD)"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'assets/images/PDD.png',
              height: 160,
            ),
            // Scan Crop button to open Camera page
            ElevatedButton(
              onPressed: () => Get.to(() => CameraPage()), //Link to  CameraPage
              child: Text('Submit'),
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.all<Color>(Colors.green),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
