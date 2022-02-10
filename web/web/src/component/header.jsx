import React, { useEffect, useState } from "react";
import './header.css';


let base64String = "";

function imageUploaded(){
    var file = document.querySelector(
        'input[type=file]')['files'][0];
  
    var reader = new FileReader();
    console.log("next");

    reader.onload = function () {
        base64String = reader.result.replace("data:", "")
            .replace(/^.+,/, "");
  
        let imageBase64Stringsep = base64String;
        console.log(imageBase64Stringsep);
    }
    reader.readAsDataURL(file);
}

function SendImages(imageBase64Stringsep) {
    const url = "http://35.154.103.98";

    return fetch(url, {
        method:"POST",
        headers:{
            Accept: "application/json",
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            img:imageBase64Stringsep,
        })
    }).catch(error => {
        console.warn(error);
    })
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


















