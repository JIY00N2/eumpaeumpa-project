// Declare Variable 'selected' : represent value of selected file
let selected = undefined; //변수선언 값 할당x
// Declare Variable 'inputValue' : represent value of change value
let inputValue = undefined;
//Declare Variable 'audioValue' : represent value of audio value
let audioValue = undefined
function valueChanger(filelist){
    //document.getElementById(): 해당 id의 요소 접근 함수
    selected = document.getElementById("upload");
    inputValue = document.getElementById("file_route");
    inputValue.value = selected.value
}