import React, { useEffect, useState } from "react";
import './header.css';


let base64String = "";

function SendImages(imageBase64Stringsep) {
    const url = "http://127.0.0.1:5000/";
    let data = {"image":imageBase64Stringsep };

    return fetch(url, {
        method:"POST",
        mode:"cors",
        headers:{
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    }).catch(error => {
        console.warn(error);
    })
}

function imageUploaded(){
    var file = document.querySelector(
        'input[type=file]')['files'][0];
  
    var reader = new FileReader();
    console.log("next");

    reader.onload = function () {
        base64String = reader.result.replace("data:", "")
            .replace(/^.+,/, "");
  
        // let imageBase64Stringsep = base64String;
        console.log(base64String);
    }
    reader.readAsDataURL(file);

    SendImages(base64String);
}


/*fetch(" http://13.235.128.146", {
    method: 'POST',
    headers:{
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        Image:base64String,
    })
})*/




function Header() {
    const [loading, setLoading] = useState(true);
    return(
<div className="title">Plant Disease Detection
            <button>
            Click
            </button>
            <input type="file" id="fileid" onChange={imageUploaded}>
            </input>
        </div>
    )
}

export default Header;


















