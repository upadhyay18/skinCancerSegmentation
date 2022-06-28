function showImage(src,target) {
  if(src && target){
    var fr=new FileReader();
    // when image is loaded, set the src of the image where you want to display it
    fr.onload = function(e) { target.src = this.result; };
    src.addEventListener("change",function() {
      // fill fr with image data    
      fr.readAsDataURL(src.files[0]);
    });
  }
  }
  var src = document.getElementById("imgFile");
  var target = document.getElementById("inputIMG");
  showImage(src,target);




  function saveDiv(divId, name) {
    var content = document.getElementById(divId);
    html2pdf().from(content).save(name);
  }
  