const video = document.getElementById('video');

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
]).then(startVideo);

async function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.onplay = async function () {
  const canvas = faceapi.createCanvasFromMedia(video);
  const labeledFaceDescriptors = await loadLabelImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  const displaySize = { width: video.width, height: video.height };

  document.body.append(canvas);

  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizeDetection = faceapi.resizeResults(detections, displaySize);

    const result = resizeDetection.map((d) =>
      faceMatcher?.findBestMatch(d.descriptor)
    );

    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    result.map((strike, i) => {
      const box = resizeDetection[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: strike.toString(),
      });

      drawBox.draw(canvas);
    });
  }, 1000);
};

function loadLabelImages() {
  const labels = ['Vic Flores', 'Fernando Aguilar', 'Stefany Lue'];

  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];

      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/VicFlores/Face-Reconog-Img/main/labeled_images/${label}/${i}.jpg `
        );

        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
