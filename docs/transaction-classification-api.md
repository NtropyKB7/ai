# 소비 내역 분류 API 계약

## Endpoint

POST /api/v1/classify-transactions

## Request

- transactionId
- transactionDate
- txnType: IN | OUT
- amount
- merchantName
- transactionDetails
- jobId
- jobName

## Response

- transactionId
- isConsumption
- category
- expenseType: FIXED | VARIABLE | null
- confidence

## Category

- FOOD
- TRANSPORT
- HOUSING
- HEALTH
- SHOPPING
- LEISURE
- SUBSCRIPTION
- EDUCATION
- FINANCE
- OTHER

## Rule

- 비소비 거래는 category와 expenseType을 null로 반환한다.
- FastAPI는 결과만 반환한다.
- Spring account-service가 TXN_ANALYSIS에 저장한다.