document.getElementById("predictionForm"),addEventListener("submit",async function(e){
    e.preventDefault();
    let formData=new FormData(this);
    let resultDiv=document.getElementById("result");
    resultDiv.textContent="Predicting....";
    const response=await fetch("/predict",{
        method:"POST",
        body:formData,
    });
    const data=await response.json();
    resultDiv.textContent='Predicted Price:${data.prediction}';
});