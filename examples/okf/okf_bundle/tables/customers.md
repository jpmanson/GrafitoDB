---
type: BigQuery Table
title: Customers
description: One row per customer across all sales channels.
resource: https://console.cloud.google.com/bigquery?p=acme&d=sales&t=customers
tags: [sales, customers]
timestamp: 2026-05-28T00:00:00Z
---

# Schema

| Column        | Type      | Description                       |
|---------------|-----------|-----------------------------------|
| `customer_id` | STRING    | Globally unique customer id.      |
| `email`       | STRING    | Customer contact email.           |
| `created_at`  | TIMESTAMP | When the customer first signed up.|

Customers place [orders](/tables/orders.md). Part of the
[sales dataset](/datasets/sales.md).
