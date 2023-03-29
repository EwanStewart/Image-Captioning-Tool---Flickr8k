document.addEventListener("DOMContentLoaded", function(event) {
    var el = document.getElementById('imgUpload');
    if(el){
        el.addEventListener('change', previewImage, false);
    }
});

function previewImage() {
    var file = document.getElementById('imgUpload').files[0];
    var reader = new FileReader();

    var imgtag = document.getElementById("imgPreview");
    imgtag.title = file.name;

    reader.onload = function(event) {
        imgtag.src = event.target.result;
    };

    reader.readAsDataURL(file);
    document.getElementById('imgPreview').style.display = 'block';
    document.getElementById('submitBtn').style.display = 'block';
}