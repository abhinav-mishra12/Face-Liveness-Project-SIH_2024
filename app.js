let session;
let lastDetectionTime = 0;
const detectionInterval = 1000; // in milliseconds

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('Liveness.onnx');
        console.log("Model loaded successfully.");
    } catch (error) {
        console.error("Error loading the ONNX model: ", error);
    }
}

async function detectLiveness(image) {
    try {
        const inputTensor = preprocessImage(image);
        console.log("Preprocessed tensor: ", inputTensor);

        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        const outputTensor = results.output;
        console.log("Model output: ", outputTensor);

        // Assuming the liveness score is in the second element of the output
        const livenessScore = outputTensor.data[1];
        console.log("Liveness score: ", livenessScore);

        const resultElement = document.getElementById('result');
        if (livenessScore > 0.5) {
            resultElement.textContent = "Live face detected!";
            resultElement.style.color = "green";
        } else {
            resultElement.textContent = "Spoof detected!";
            resultElement.style.color = "red";
        }
    } catch (error) {
        console.error("Error during inference: ", error);
    }
}

function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = 224;
    canvas.height = 224;
    context.drawImage(image, 0, 0, 224, 224);

    const resizedImageData = context.getImageData(0, 0, 224, 224);

    const floatData = new Float32Array(3 * 224 * 224);
    let pixelIndex = 0;
    for (let i = 0; i < resizedImageData.data.length; i += 4) {
        floatData[pixelIndex++] = resizedImageData.data[i] / 255.0;
        floatData[pixelIndex++] = resizedImageData.data[i + 1] / 255.0;
        floatData[pixelIndex++] = resizedImageData.data[i + 2] / 255.0;
    }

    return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

async function handleLiveDetection() {
    const now = Date.now();
    if (now - lastDetectionTime < detectionInterval) {
        return; // Skip detection if it's too soon
    }
    lastDetectionTime = now;

    const video = document.getElementById('video');
    if (video.srcObject) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        await detectLiveness(canvas);
    } else {
        console.error("No video stream available.");
    }
}

function handleFileUpload(event, isVideo = false) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        if (isVideo) {
            const video = document.getElementById('uploadedVideo');
            video.src = e.target.result;
            video.style.display = 'block';
        } else {
            const img = new Image();
            img.onload = function() {
                document.getElementById('uploadedImage').src = img.src;
                document.getElementById('uploadedImage').style.display = 'block';
            };
            img.src = e.target.result;
        }
    };

    if (isVideo) {
        reader.readAsDataURL(file);
    } else {
        reader.readAsDataURL(file);
    }
}

window.onload = async function() {
    await loadModel();

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            const video = document.getElementById('video');
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            console.error("Error accessing the webcam: ", err);
        });

    document.getElementById('imageFile').addEventListener('change', (event) => handleFileUpload(event, false));
    document.getElementById('videoFile').addEventListener('change', (event) => handleFileUpload(event, true));

    document.getElementById('detectLivenessLive').onclick = handleLiveDetection;

    document.getElementById('detectLivenessImage').onclick = function() {
        const img = document.getElementById('uploadedImage');
        if (img.src) {
            detectLiveness(img);
        }
    };

    document.getElementById('detectLivenessVideo').onclick = function() {
        const video = document.getElementById('uploadedVideo');
        if (video.src) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
            detectLiveness(canvas);
        }
    };
};
