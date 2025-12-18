import React, { useState, useRef } from 'react';
import { Upload, Brain, Loader, Download, X, FileImage } from 'lucide-react';
import './styles.css';

export default function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setOriginalImage(e.target.result);
        setMaskImage(null);
        setSegmentedImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleSegment = async () => {
  if (!originalImage) return;
  
  setIsLoading(true);
  
  try {
    // Convert base64 to blob
      const blob = await fetch(originalImage).then(r => r.blob());
      
      // Create FormData
      const formData = new FormData();
      formData.append('image', blob, 'brain_mri.jpg');
      
      // Send to API
      const response = await fetch('http://localhost:5000/segment', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setMaskImage(data.data.mask);
        setSegmentedImage(data.data.segmented);
        
        // Optional: Show statistics
        console.log('Tumor Statistics:', data.data.statistics);
      } else {
        alert('Segmentation failed: ' + data.message);
      }
      
    } catch (error) {
      console.error('Error during segmentation:', error);
      alert('Terjadi kesalahan saat melakukan segmentasi');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setMaskImage(null);
    setSegmentedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDownload = (imageData, filename) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    link.click();
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div>
            <div className="logo">
              <Brain className="logo-icon" />
              <h1>BrainSight</h1>
            </div>
            <p className="subtitle">AI-Powered Brain Tumor Segmentation</p>
          </div>
        </header>

        {/* Upload Section */}
        {!originalImage && (
          <div className="upload-section">
            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="upload-icon" />
              <h3>Upload Brain MRI Image</h3>
              <p>Drag and drop or click to browse</p>
              <p className="upload-hint">Supports: JPG, PNG, DICOM</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
            </div>
          </div>
        )}

        {/* Image Display Section */}
        {originalImage && (
          <div className="content-section">
            <div className="image-grid">
              {/* Original Image */}
              <div className="image-card">
                <div className="card-header">
                  <FileImage size={20} />
                  <h3>Original Image</h3>
                </div>
                <div className="image-container">
                  <img src={originalImage} alt="Original MRI" />
                </div>
                <button 
                  className="btn btn-download"
                  onClick={() => handleDownload(originalImage, 'original.png')}
                >
                  <Download size={18} />
                  Download
                </button>
              </div>

              {/* Mask Image */}
              <div className="image-card">
                <div className="card-header">
                  <Brain size={20} />
                  <h3>Tumor Mask</h3>
                </div>
                <div className="image-container">
                  {maskImage ? (
                    <img src={maskImage} alt="Tumor Mask" />
                  ) : (
                    <div className="placeholder">
                      <Brain size={48} />
                      <p>Mask will appear here</p>
                    </div>
                  )}
                </div>
                {maskImage && (
                  <button 
                    className="btn btn-download"
                    onClick={() => handleDownload(maskImage, 'mask.png')}
                  >
                    <Download size={18} />
                    Download
                  </button>
                )}
              </div>

              {/* Segmented Image */}
              <div className="image-card">
                <div className="card-header">
                  <FileImage size={20} />
                  <h3>Segmented Result</h3>
                </div>
                <div className="image-container">
                  {segmentedImage ? (
                    <img src={segmentedImage} alt="Segmented Result" />
                  ) : (
                    <div className="placeholder">
                      <FileImage size={48} />
                      <p>Result will appear here</p>
                    </div>
                  )}
                </div>
                {segmentedImage && (
                  <button 
                    className="btn btn-download"
                    onClick={() => handleDownload(segmentedImage, 'segmented.png')}
                  >
                    <Download size={18} />
                    Download
                  </button>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
              <button 
                className="btn btn-secondary"
                onClick={handleReset}
              >
                <X size={20} />
                Reset
              </button>
              <button 
                className="btn btn-primary"
                onClick={handleSegment}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader className="spin" size={20} />
                    Processing...
                  </>
                ) : (
                  <>
                    <Brain size={20} />
                    Segment Tumor
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="footer">
          <p>© 2025 Brain Tumor Segmentation System</p>
          <p className="footer-note">For research and educational purposes only</p>
        </footer>
      </div>
    </div>
  );
}