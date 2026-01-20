const mysql = require('mysql2/promise');
require('dotenv').config();

const DB_USE = process.env.DB_USE === 'true';

let pool = null;

if (DB_USE) {
  const host = process.env.DB_HOST || 'localhost';
  const port = process.env.DB_PORT ? Number(process.env.DB_PORT) : 3306;
  const user = process.env.DB_USER || 'root';
  const password = process.env.DB_PASSWORD || '0904';
  const database = process.env.DB_NAME || 'aml_dat';

  pool = mysql.createPool({
    host,
    port,
    user,
    password,
    database,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
  });
}

module.exports = { DB_USE, pool };
