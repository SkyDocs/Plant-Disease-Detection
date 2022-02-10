# Plant-Disease-Detection

Plant Disease Detection project to detect the diseases in the plants by scanning the images of the leaves and then passing to through the neural network to detect wether the plant is infected or not. 

For this we are planning to develop an flutter app and a web-app, from where the users can upload the image of the leaves and get them detected. 

The images will be sent to a remote cloud sever where the neural network will detect the disease and then will return the result.


### Pulling the repo from origin

Since we all are working from the same repo and do pushes code while other too have to and at times it leads to confilct and then we have to merge, and mostly git does it by itself but then there is commit message like "Merge branch 'master' of https://github.com/SkyDocs/Plant-Disease-Det…" and also a hell lot of files are re pushed. This just pollutes the commit history and makes it more difficult to understand what exactly the changes are made.

So, I request you guys is to rebase the changes at the pulling so that, the remote changes are overidden with the local and thus keeping the commit history clean. 
It can be simply be done by `git pull --rebase` 

Just add the `--rebase` tag every time you pull.
