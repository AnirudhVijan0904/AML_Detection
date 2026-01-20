# AML (Anti-Money Laundering) System

A comprehensive Anti-Money Laundering (AML) monitoring system with both frontend and backend components. The system helps in detecting and analyzing suspicious financial transactions in real-time and through manual analysis.

## Project Structure

```
.
├── backend/                 # Backend server
│   ├── config/             # Configuration files
│   ├── data/               # Data storage (e.g., CSV files)
│   ├── ml/                 # Machine learning models and processing
│   ├── routes/             # API routes
│   ├── services/           # Business logic and services
│   ├── .env                # Environment variables
│   ├── package.json        # Backend dependencies
│   └── server.js           # Main server file
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React source code
│   ├── package.json        # Frontend dependencies
│   └── README.md           # Frontend specific documentation
└── start-backend.ps1       # Script to start the backend server
```

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)
- MySQL (if using database functionality)

## Environment Variables

### Backend (.env)

Create a `.env` file in the `backend` directory with the following variables:

```env
PORT=5000
CSV_PATH=data/transactions.csv

# Database (MySQL) settings
DB_USE=true
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database
```

### Frontend

The frontend is configured to proxy API requests to `http://localhost:5000` by default. You can modify this in `frontend/package.json` if needed.

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the backend server:
   ```bash
   npm start
   ```
   Or use the provided PowerShell script:
   ```powershell
   .\start-backend.ps1
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   This will open the application in your default browser at `http://localhost:3000`.

## Features

- Real-time transaction monitoring
- Manual transaction analysis
- Statistical insights and visualizations
- Suspicious activity detection
- Database integration for transaction storage

## API Endpoints

- `GET /api/transactions` - Get all transactions
- `POST /api/transactions/analyze` - Analyze transactions for suspicious activity
- `GET /api/stats` - Get transaction statistics
- `WS /ws` - WebSocket endpoint for real-time updates

## Dependencies

### Backend
- Express.js - Web framework
- CORS - Cross-Origin Resource Sharing
- dotenv - Environment variable management
- MySQL2 - MySQL database client
- WebSocket - Real-time communication
- csv-parser - CSV file processing

### Frontend
- React - UI library
- React Router - Navigation
- Axios - HTTP client
- Recharts - Data visualization
- Testing libraries (Jest, React Testing Library)

## License

This project is licensed under the MIT License.
