# Synthetic Image Detector

A Next.js application that allows users to upload images and check if they are GAN-generated or real photos using a PyTorch model served via a Flask backend.

## Features

- Drag-and-drop or file upload for images
- Image preview
- Integration with a Flask backend for image classification
- Real-time loading indicators
- Error handling with user feedback
- Responsive Material UI design

## Prerequisites

- Node.js 18.0.0 or later
- Python Flask backend running on http://localhost:5000 (see Backend Setup below)

## Getting Started

### Frontend Setup

1. Clone the repository:

   ```
   git clone <repository-url>
   cd synthetic-sniffer
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Run the development server:

   ```
   npm run dev
   ```

4. Open your browser and visit `http://localhost:3000`

### Backend Setup

This frontend requires a Flask backend with a PyTorch model that exposes a `/predict` endpoint. The backend should:

1. Accept image uploads via a POST request to `/predict`
2. Return a JSON response with a `prediction` field containing either "GAN-generated" or "Real"

Example backend API response:

```json
{
  "prediction": "GAN-generated"
}
```

## How to Use

1. Access the application at `http://localhost:3000`
2. Drag and drop an image file or click to select from your files
3. Click "Analyze Image" to send the image to the backend for prediction
4. View the prediction result

## Tech Stack

- Next.js
- TypeScript
- Material UI (MUI)
- Axios for API requests

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
