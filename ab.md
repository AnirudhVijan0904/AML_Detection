$csv = Import-Csv "C:\Users\AnirudhVijan\Desktop\intern\eda\top20_stacked_positive_DECODED.csv"
$api = "http://localhost:5000/api/manual/predict?debug=true"

$i = 0
foreach ($row in ($csv | Select-Object -First 5)) {
  $i++
  Write-Host "`n=== Test #$i ==="
  $body = [ordered]@{
    fromBank = $row.'From Bank'
    fromAccount = $row.'Account'
    toBank = $row.'To Bank'
    toAccount = $row.'Account.1'
    amountReceived = [double]$row.'Amount Received'
    receivingCurrency = $row.'Receiving Currency'
    paymentFormat = $row.'Payment Format'
    paymentCurrency = $row.'Payment Currency'
    amount = [double]$row.'Amount'
    fullName = $row.'name'
    nationality = $row.'nationality'
    occupation = $row.'occupation'
    isPep = $row.'is_pep'
    kycScore = $row.'kyc_score'
    kycStatus = $row.'kyc_status'
    monthlyIncome = $row.'monthly_income'
    dob = $row.'date_of_birth'
    customerSince = $row.'customer_since'
    txnCountLast7Days = $row.'txn_count_last_7_days'
    totalAmountLast30Days = $row.'total_amount_last_30_days'
    daysSinceLastTxn = $row.'days_since_last_txn'
    beneficiaryAvgReceivedAmount = $row.'beneficiary_avg_received_amount'
    beneficiaryTotalReceived = $row.'beneficiary_total_received'
    beneficiaryReceiveCount = $row.'beneficiary_receive_count'
    beneficiaryUniqueSenders = $row.'beneficiary_unique_senders'
    beneficiaryUniqueSenderNationalitiesSoFar = $row.'beneficiary_unique_sender_nationalities_so_far'
    beneficiaryPepSenderCountAtTimeOfTxn = $row.'beneficiary_pep_sender_count_at_time_of_txn'
    beneficiaryTotalReceivedSoFar = $row.'beneficiary_total_received_so_far'
    beneficiaryReceiveCountSoFar = $row.'beneficiary_receive_count_so_far'
    beneficiaryUniqueSendersAtTimeOfTxn = $row.'beneficiary_unique_senders_at_time_of_txn'
  }
  $json = $body | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri $api -Method Post -Body $json -ContentType "application/json"
}