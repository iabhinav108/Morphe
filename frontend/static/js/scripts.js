const videoUpload = document.querySelector('input[name="video"]');
const videoPlayer = document.getElementById('videoPlayer');
const startButton = document.getElementById('startButton');
const outputField = document.getElementById('outputField');

// When a video is uploaded
videoUpload.addEventListener('change', function(event) {
    const file = event.target.files[0];
        if (file) {
            const videoURL = URL.createObjectURL(file);
            videoPlayer.src = videoURL;
            videoPlayer.style.display = 'block';
        }
});

// When the Start button is clicked
startButton.addEventListener('click', function() {
    if (videoPlayer.src) {
    // Placeholder for deepfake detection logic
        outputField.textContent = "Detecting deepfake...";
        // Simulate detection process (this should be replaced with actual model call)
        setTimeout(() => {
                   
            outputField.textContent = "Detection complete. No deepfake detected."; // Update output with the result
        }, 2000); // Simulate 2 seconds of processing time
        {/* //    } else { */}
        outputField.textContent = "Please upload a video first.";
    }
});