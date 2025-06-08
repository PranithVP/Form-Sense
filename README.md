# FormSense

AI-powered workout form analysis platform that provides real-time feedback on exercise technique using computer vision and machine learning.

## Features

- Real-time pose estimation and form analysis
- Detailed joint angle measurements
- Exercise-specific feedback and tips
- Progress tracking
- Support for multiple exercises

## Tech Stack

### Frontend
- React with TypeScript
- Vite
- Tailwind CSS
- Shadcn UI Components

### Backend
- FastAPI
- MediaPipe for pose estimation
- PyTorch for exercise classification
- OpenCV for video processing

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/formsense.git
cd formsense
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend/formsense
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend/formsense
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
