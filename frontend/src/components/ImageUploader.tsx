'use client';

import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import {
    Box,
    Typography,
    Button,
    CircularProgress,
    Paper,
    Snackbar,
    Alert,
    AlertColor
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';

export default function ImageUploader() {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [prediction, setPrediction] = useState<string | null>(null);
    const [openSnackbar, setOpenSnackbar] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState('');
    const [snackbarSeverity, setSnackbarSeverity] = useState<AlertColor>('info');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file: File) => {
        if (!file.type.match('image.*')) {
            showSnackbar('Please select an image file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            setSelectedImage(e.target?.result as string);
            setPrediction(null);
        };
        reader.readAsDataURL(file);
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleAnalyze = async () => {
        if (!selectedImage) {
            showSnackbar('Please select an image first', 'warning');
            return;
        }

        // Convert base64 string to file object
        const base64Response = await fetch(selectedImage);
        const blob = await base64Response.blob();
        const imageFile = new File([blob], 'image.jpg', { type: 'image/jpeg' });

        // Create form data
        const formData = new FormData();
        formData.append('image', imageFile);

        setIsLoading(true);
        setPrediction(null);

        try {
            const response = await axios.post(
                'http://localhost:5000/predict',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );

            if (response.data && response.data.prediction) {
                setPrediction(response.data.prediction);
                showSnackbar('Analysis complete!', 'success');
            } else {
                showSnackbar('Received invalid response from server', 'error');
            }
        } catch (error) {
            console.error('Error analyzing image:', error);
            showSnackbar('Error analyzing image. Please try again.', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    const showSnackbar = (message: string, severity: AlertColor) => {
        setSnackbarMessage(message);
        setSnackbarSeverity(severity);
        setOpenSnackbar(true);
    };

    const handleCloseSnackbar = () => {
        setOpenSnackbar(false);
    };

    const resetImage = () => {
        setSelectedImage(null);
        setPrediction(null);
    };

    return (
        <Box sx={{ maxWidth: 600, mx: 'auto', p: 2 }}>
            <Typography variant="h4" align="center" gutterBottom>
                Synthetic Image Detector
            </Typography>
            <Typography variant="body1" align="center" gutterBottom>
                Upload an image to check if it is GAN-generated or real
            </Typography>

            <Paper
                sx={{
                    mt: 3,
                    p: 3,
                    border: '2px dashed #ccc',
                    borderRadius: 2,
                    cursor: 'pointer',
                    textAlign: 'center',
                    bgcolor: selectedImage ? 'transparent' : '#f9f9f9',
                }}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={handleUploadClick}
            >
                <input
                    type="file"
                    accept="image/*"
                    hidden
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                />

                {selectedImage ? (
                    <Box>
                        <Box
                            component="img"
                            src={selectedImage}
                            alt="Selected"
                            sx={{
                                maxWidth: '100%',
                                maxHeight: 300,
                                objectFit: 'contain',
                                mb: 2,
                            }}
                        />
                        <Typography variant="body2" color="textSecondary">
                            Click to select a different image
                        </Typography>
                    </Box>
                ) : (
                    <Box sx={{ p: 3 }}>
                        <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                        <Typography variant="h6">Drag and drop an image</Typography>
                        <Typography variant="body2" color="textSecondary">
                            or click to select from your files
                        </Typography>
                    </Box>
                )}
            </Paper>

            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center', gap: 2 }}>
                {selectedImage && (
                    <>
                        <Button
                            variant="contained"
                            onClick={handleAnalyze}
                            disabled={isLoading}
                            startIcon={isLoading ? <CircularProgress size={20} /> : null}
                        >
                            {isLoading ? 'Analyzing...' : 'Analyze Image'}
                        </Button>
                        <Button variant="outlined" onClick={resetImage} disabled={isLoading}>
                            Reset
                        </Button>
                    </>
                )}
            </Box>

            {isLoading && (
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
                    <CircularProgress />
                </Box>
            )}

            {prediction && (
                <Box
                    sx={{
                        mt: 3,
                        p: 2,
                        bgcolor: prediction === 'GAN-generated' ? '#ffebee' : '#e8f5e9',
                        borderRadius: 2,
                        textAlign: 'center',
                    }}
                >
                    <Typography variant="h5">
                        Result: {prediction}
                    </Typography>
                    <Typography variant="body1">
                        {prediction === 'GAN-generated'
                            ? 'This image appears to be AI-generated.'
                            : 'This image appears to be a real photograph.'}
                    </Typography>
                </Box>
            )}

            <Snackbar
                open={openSnackbar}
                autoHideDuration={6000}
                onClose={handleCloseSnackbar}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert onClose={handleCloseSnackbar} severity={snackbarSeverity}>
                    {snackbarMessage}
                </Alert>
            </Snackbar>
        </Box>
    );
} 