# Plant Disease Detection web-app

## Local Development 

Run `npm start` to start the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

## Production

Run `npm run build` to build the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

Install pm2 and express.

After that Run `pm2 start server.js` to start the server.

Now the react app is serving at port 3000 (port specified in server.js). 

Now you may install a server like nginx and pass the http:localhost:3000 as proxy_pass to host.
example:

```
server {
	server_name 20.219.105.135; # Whatever is your IP or domain
	location / {
		proxy_pass http://localhost:3000; # Whatever port your app runs on
		proxy_http_version 1.1;
		proxy_set_header Upgrade $http_upgrade;
		proxy_set_header Connection 'upgrade';
		proxy_set_header Host $host;
		proxy_cache_bypass $http_upgrade;
	}
}	
```