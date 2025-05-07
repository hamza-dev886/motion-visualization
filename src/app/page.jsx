import React, { useRef, useEffect, useState } from 'react';
import * as tf from 'tensorflow';
const PoseDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [angles, setAngles] = useState({});
  // Load the MoveNet model
  useEffect(() => {
    async function loadModel() {
      try {
        setLoading(true);
        // Load the MoveNet model
        const movenetModel = await tf.loadGraphModel(
          'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4',
          { fromTFHub: true }
        );
        setModel(movenetModel);
        setLoading(false);
      } catch (error) {
        console.error('Failed to load the model:', error);
        setLoading(false);
      }
    }
    
    loadModel();
  }, []);
  // Set up the camera
  useEffect(() => {
    if (!model) return;
    const setupCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support the camera access API.');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          
          // Start detection once video is playing
          videoRef.current.onloadeddata = () => {
            detectPose();
          };
        }
      } catch (error) {
        console.error('Error accessing the camera:', error);
      }
    };
    setupCamera();
  }, [model]);
  // Calculate angle between three points
  const calculateAngle = (p1, p2, p3) => {
    if (!p1 || !p2 || !p3) return null;
    
    // Convert to vectors
    const v1 = {
      x: p1.x - p2.x,
      y: p1.y - p2.y
    };
    
    const v2 = {
      x: p3.x - p2.x,
      y: p3.y - p2.y
    };
    // Calculate dot product
    const dotProduct = v1.x * v2.x + v1.y * v2.y;
    
    // Calculate magnitudes
    const v1Magnitude = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const v2Magnitude = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    
    // Calculate angle in radians and convert to degrees
    const angle = Math.acos(dotProduct / (v1Magnitude * v2Magnitude)) * (180 / Math.PI);
    
    return Math.round(angle);
  };
  // Detect pose from video frame
  const detectPose = async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.width;
    canvas.height = video.height;
    const detectFrame = async () => {
      // Get video frame as tensor
      const videoFrame = tf.browser.fromPixels(video);
      const resizedFrame = tf.image.resizeBilinear(videoFrame, [192, 192]);
      const input = tf.expandDims(resizedFrame, 0);
      
      // Get pose prediction
      const result = await model.predict(input);
      const poses = await result.array();
      
      // Clean up tensors
      videoFrame.dispose();
      resizedFrame.dispose();
      input.dispose();
      result.dispose();
      // Draw video frame on canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      if (poses && poses.length > 0) {
        // First pose detection
        const pose = poses[0];
        const keypoints = pose[0];
        
        // MoveNet returns 17 keypoints
        const keypointMap = {};
        const connections = [
          ['nose', 'left_eye'], ['nose', 'right_eye'],
          ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
          ['nose', 'left_shoulder'], ['nose', 'right_shoulder'],
          ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
          ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
          ['left_shoulder', 'right_shoulder'],
          ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
          ['left_hip', 'right_hip'],
          ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
          ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']
        ];
        
        // Keypoint names in order returned by MoveNet
        const keypointNames = [
          'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ];
        // Extract keypoints and create a map
        for (let i = 0; i < keypoints.length; i++) {
          const confidence = keypoints[i][2];
          if (confidence > 0.3) { // Only use keypoints with sufficient confidence
            const y = keypoints[i][0] * canvas.height;
            const x = keypoints[i][1] * canvas.width;
            
            keypointMap[keypointNames[i]] = { x, y };
            
            // Draw keypoint
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
            
            // Label keypoint
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(keypointNames[i], x + 10, y + 5);
          }
        }
        
        // Draw connections
        for (const [p1Name, p2Name] of connections) {
          const p1 = keypointMap[p1Name];
          const p2 = keypointMap[p2Name];
          
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = 'lime';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
        
        // Calculate important angles
        const newAngles = {
          leftElbow: calculateAngle(
            keypointMap['left_shoulder'], 
            keypointMap['left_elbow'], 
            keypointMap['left_wrist']
          ),
          rightElbow: calculateAngle(
            keypointMap['right_shoulder'], 
            keypointMap['right_elbow'], 
            keypointMap['right_wrist']
          ),
          leftShoulder: calculateAngle(
            keypointMap['left_elbow'], 
            keypointMap['left_shoulder'], 
            keypointMap['left_hip']
          ),
          rightShoulder: calculateAngle(
            keypointMap['right_elbow'], 
            keypointMap['right_shoulder'], 
            keypointMap['right_hip']
          ),
          leftHip: calculateAngle(
            keypointMap['left_shoulder'], 
            keypointMap['left_hip'], 
            keypointMap['left_knee']
          ),
          rightHip: calculateAngle(
            keypointMap['right_shoulder'], 
            keypointMap['right_hip'], 
            keypointMap['right_knee']
          ),
          leftKnee: calculateAngle(
            keypointMap['left_hip'], 
            keypointMap['left_knee'], 
            keypointMap['left_ankle']
          ),
          rightKnee: calculateAngle(
            keypointMap['right_hip'], 
            keypointMap['right_knee'], 
            keypointMap['right_ankle']
          )
        };
        
        setAngles(newAngles);
      }
      
      // Continue detection loop
      requestAnimationFrame(detectFrame);
    };
    
    detectFrame();
  };
  return (
    <div className="flex flex-col items-center p-4 bg-gray-900 text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">MoveNet Pose Detection</h1>
      
      {loading ? (
        <div className="text-xl">Loading MoveNet model...</div>
      ) : (
        <>
          <div className="relative">
            <video
              ref={videoRef}
              className="hidden"
              width="640"
              height="480"
              autoPlay
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="border-2 border-blue-500 rounded"
              width="640"
              height="480"
            />
          </div>
          
          <div className="mt-4 p-4 bg-gray-800 rounded w-full max-w-lg">
            <h2 className="text-xl font-semibold mb-2">Detected Angles:</h2>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(angles).map(([joint, angle]) => (
                <div key={joint} className="flex justify-between bg-gray-700 p-2 rounded">
                  <span className="capitalize">{joint.replace(/([A-Z])/g, ' $1')}:</span>
                  <span className="font-mono">{angle !== null ? `${angle}°` : 'N/A'}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
export default PoseDetector;