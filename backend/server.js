const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const manualRoutes = require('./routes/manualRoutes');
const realtimeRoutes = require('./routes/realtimeRoutes');
const statsRoutes = require('./routes/statsRoutes');
const debugRoutes = require('./routes/debugRoutes');

// Load environment variables first
dotenv.config();

// Global error handlers
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // Don't exit immediately, allow the process to continue
  // process.exit(1); // Uncomment if you want to exit on uncaught exceptions
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Application specific logging, throwing an error, or other logic here
});

const amlApp = express();
const PORT = process.env.PORT || 5000;
const { DB_USE, pool } = require('./config/dbConfig');
const LOG_HEALTH = String(process.env.LOG_HEALTH || '').toLowerCase() === 'true';

// Heartbeat to confirm process stays alive
const startTs = Date.now();
setInterval(() => {
  const uptimeSec = Math.round((Date.now() - startTs) / 1000);
  console.log(`[heartbeat] uptime=${uptimeSec}s @ ${new Date().toISOString()}`);
}, 30000);

// Middleware
amlApp.use(cors());
amlApp.use(express.json({ limit: '1mb' }));

// Lightweight request logger (enable via LOG_REQUESTS=true)
if (process.env.LOG_REQUESTS === 'true') {
  amlApp.use((req, res, next) => {
    console.log(`[req] ${req.method} ${req.originalUrl}`);
    next();
  });
}

// Database connection status
if (DB_USE) {
  console.log('DB_USE is enabled; connection pool created');
  // Test the database connection
  if (pool) {
    pool.getConnection()
      .then(conn => {
        console.log('Successfully connected to the database');
        conn.release();
      })
      .catch(err => {
        console.error('Database connection error:', err);
        // Don't exit, let the server start but log the error
      });
  }
} else {
  console.log('DB_USE is disabled; using CSV fallback');}

// Routes
amlApp.use('/api/manual', manualRoutes);
amlApp.use('/api/realtime', realtimeRoutes);
amlApp.use('/api/stats', statsRoutes);
amlApp.use('/api/debug', debugRoutes);

// Health check endpoint
amlApp.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Alias health under /api for CRA proxy-based clients
amlApp.get('/api/health', (req, res) => {
  if (LOG_HEALTH) {
    console.log(`[health] /api/health hit from ${req.ip} at ${new Date().toISOString()}`);
  }
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handling middleware
amlApp.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({ error: 'Internal Server Error', message: err.message });
});

// Start the server
const server = amlApp.listen(PORT, () => {
  console.log('==========================================================');
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`DB_USE: ${DB_USE ? 'enabled' : 'disabled'}`);
  console.log(`Access at: http://localhost:${PORT}`);
  console.log('==========================================================');
});

// Handle server errors
server.on('error', (error) => {
  if (error.syscall !== 'listen') {
    throw error;
  }

  // Handle specific listen errors with friendly messages
  switch (error.code) {
    case 'EACCES':
      console.error(`Port ${PORT} requires elevated privileges`);
      process.exit(1);
      break;
    case 'EADDRINUSE':
      console.error(`Port ${PORT} is already in use`);
      process.exit(1);
      break;
    default:
      throw error;
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received. Shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    if (pool) {
      pool.end(() => {
        console.log('Database connection pool closed');
        process.exit(0);
      });
    } else {
      process.exit(0);
    }
  });
});

// Additional diagnostics for unexpected exits
process.on('SIGINT', () => {
  console.log('SIGINT received (Ctrl+C).');
});
process.on('SIGBREAK', () => {
  console.log('SIGBREAK received.');
});
process.on('beforeExit', (code) => {
  console.log(`beforeExit triggered with code=${code}`);
});
process.on('exit', (code) => {
  console.log(`Process exit with code=${code}`);
});