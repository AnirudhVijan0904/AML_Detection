import React, { useState } from "react";
import "./ManualAnalysis.css";
import { api } from "../api/api";

function ManualAnalysis() {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    // Print a debug line whenever input is received/changed
    console.log(`[ManualAnalysis] Input received: ${name}=${value}`);
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('=== USER CLICKED ANALYZE BUTTON ===');
    console.log('Form Data:', JSON.stringify(formData, null, 2));
    console.log('Sending to:', 'http://localhost:5000/api/manual/predict');
    
    setLoading(true);
    setResult(null);

    try {
      console.log('Making API request...');
      // Use relative path so axios baseURL '/api' applies => '/api/manual/predict'
      const response = await api.post("manual/predict", formData);
      console.log('Response received:', response.data);
      setResult(response.data);
    } catch (err) {
      console.error('ERROR:', err);
      console.error('Error details:', err.response?.data || err.message);
      setResult({
        prediction: "Error",
        message: err.response?.data?.error || "Unable to connect to backend",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="manual-analysis">
      <h2 className="page-title">Manual Transaction Analysis</h2>
      <p className="page-subtitle">
        Enter essential transaction and customer details to analyze laundering risk.
      </p>

      <form className="analysis-form" onSubmit={handleSubmit}>
        <h3>Transaction Details</h3>
        <div className="form-section">
          <input name="fromBank" placeholder="From Bank" onChange={handleChange} />
          <input name="fromAccount" placeholder="From Account" onChange={handleChange} />
          <input name="toBank" placeholder="To Bank" onChange={handleChange} />
          <input name="toAccount" placeholder="To Account" onChange={handleChange} />
          <input type="number" step="any" name="amountReceived" placeholder="Amount Received" onChange={handleChange} />
          <input name="receivingCurrency" placeholder="Receiving Currency" onChange={handleChange} />
          <input type="number" step="any" name="amount" placeholder="Amount" onChange={handleChange} />
          <input name="paymentCurrency" placeholder="Payment Currency" onChange={handleChange} />
          <input name="paymentFormat" placeholder="Payment Format" onChange={handleChange} />
        </div>

        <h3>Customer Information</h3>
        <div className="form-section">
          <input name="fullName" placeholder="Full Name" onChange={handleChange} />
          <input name="nationality" placeholder="Nationality" onChange={handleChange} />
          <input name="occupation" placeholder="Occupation" onChange={handleChange} />
          <input name="kycStatus" placeholder="KYC Status" onChange={handleChange} />
          <input type="number" step="any" name="kycScore" placeholder="KYC Score" onChange={handleChange} />
          <input name="isPep" placeholder="Is PEP (Yes/No)" onChange={handleChange} />
          <input type="number" step="any" name="monthlyIncome" placeholder="Monthly Income" onChange={handleChange} />

          <div className="date-input-wrapper">
            <label htmlFor="dob">Date of Birth</label>
            <input 
              id="dob"
              name="dob" 
              type="date" 
              onChange={handleChange} 
            />
          </div>

          <div className="date-input-wrapper">
            <label htmlFor="customerSince">Customer Since</label>
            <input 
              id="customerSince"
              name="customerSince" 
              type="date" 
              onChange={handleChange} 
            />
          </div>
        </div>

        <h3>Recent Activity (Sender)</h3>
        <div className="form-section">
          <input type="number" step="any" name="txnCountLast7Days" placeholder="Txn Count (Last 7 Days)" onChange={handleChange} />
          <input type="number" step="any" name="totalAmountLast30Days" placeholder="Total Amount (Last 30 Days)" onChange={handleChange} />
          <input type="number" step="any" name="daysSinceLastTxn" placeholder="Days Since Last Txn" onChange={handleChange} />
        </div>

        <h3>Beneficiary Aggregates</h3>
        <div className="form-section">
          <input type="number" step="any" name="beneficiaryReceiveCount" placeholder="Receive Count" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryTotalReceived" placeholder="Total Received" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryAvgReceivedAmount" placeholder="Avg Received Amount" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryUniqueSenders" placeholder="Unique Senders" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryUniqueSenderNationalitiesSoFar" placeholder="Unique Sender Nationalities (So Far)" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryPepSenderCountAtTimeOfTxn" placeholder="PEP Sender Count (At Txn Time)" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryUniqueSendersAtTimeOfTxn" placeholder="Unique Senders (At Txn Time)" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryReceiveCountSoFar" placeholder="Receive Count (So Far)" onChange={handleChange} />
          <input type="number" step="any" name="beneficiaryTotalReceivedSoFar" placeholder="Total Received (So Far)" onChange={handleChange} />
        </div>

        <button type="submit" className="analyze-btn" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Transaction"}
        </button>
      </form>

      {result && (
        <div className="result-card">
          <h3>Analysis Result</h3>
          {result.prediction === "Error" ? (
            <p className="error">{result.message}</p>
          ) : (
            <>
              <p><strong>Risk Level:</strong> {result.prediction}</p>
              <p>
                <strong>Confidence:</strong>{" "}
                {result.confidence ? `${(result.confidence * 100).toFixed(2)}%` : "N/A"}
              </p>
              {result.key_factors && (
                <ul>
                  {result.key_factors.map((factor, idx) => (
                    <li key={idx}>{factor}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default ManualAnalysis;