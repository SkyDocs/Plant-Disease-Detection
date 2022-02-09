# api-backend

Flask API server written for the prediction of the image, passed to the server. 


## working 

The api server accepts the image in base64 encoded string in json format of

```json
{"image": "base_64_string_image"}
```

And then pass the json file via the post request to the server at the root location. 

Here is an example of curl post request of file name `image.jpg` and server serving at ip `127.0.0.1:8000`

```bash
(echo -n '{"image": "'; base64 image.jpg; echo '"}') | curl -H "Content-Type: application/json" -d @- http://127.0.0.1:8000
```

Change the ip and image location accordingly.

And the reponse will be in the json format of the Plant name and disease.

*__NOTE__: The trained h5 model was too big(870MiB) to push, so its kept sperate.*

Download the trained model from [https://drive.google.com/drive/folders/1wkAUa0dKp0GbN4YMvxPuFa5ZiOoMcHyg?usp=sharing](https://drive.google.com/drive/folders/1wkAUa0dKp0GbN4YMvxPuFa5ZiOoMcHyg?usp=sharing)
