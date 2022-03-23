import React, { useEffect, useState } from "react";
import './header.css';


let base64String = "";

/*function SendImages(imageBase64Stringsep) {
    const url = "http://52.140.7.124/";
    let data = {"image":imageBase64Stringsep };

    return fetch(url, {
        method:"POST",
        mode:"cors",
        headers:{
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then((response) => response.json())
    .then((json) => console.log(json))
    .catch(error => {
        console.warn(error);
    })
}*/

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
    const [loading, setLoading] = useState({});

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

    function SendImages(imageBase64Stringsep) {
        const url = "http://52.140.7.124/";
        let data = {"image":imageBase64Stringsep };
    
        fetch(url, {
            method:"POST",
            mode:"cors",
            headers:{
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then((response) => response.json())
        .then((json) => {
            console.log(json)
            setLoading(json)
            console.log(loading)
        })
        .catch(error => {
            console.warn(error);
        })
    }
    /*useEffect(() => {
    fetch('http://52.140.7.124/')
        .then(json=>{console.log(json)});
    }, []);*/

    return(
    <div className="main" id='#'>
        <h1 className="title">Plant Disease Detection</h1><br/>
        <div className="input-group mb-3 input-bar">
            <input type="file" class="form-control" id="inputGroupFile02" onChange={imageUploaded}/>
            <label className="input-group-text" for="inputGroupFile02">Upload</label>
        </div>
        <div class="card" id="card">
            <div class="card-body">
                <h1>Plant Details</h1>
        
                <h5 class="card-title list-group-item">Plant: {loading.Plant}</h5>
                <h5 class="card-title list-group-item">Disease: {loading.Disease}</h5>
                <h5 class="card-title list-group-item">Remedy: {loading.remedy}</h5>
            </div>
        </div>
    </div>
    )
}

export default Header;


















