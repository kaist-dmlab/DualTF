# DualTF

# Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection

## 1. Overview
In light of the remarkable advancements made in time-series anomaly detection (TSAD), recent emphasis has been placed on exploiting the frequency domain as well as the time domain to address the difficulties in precisely detecting pattern-wise anomalies. However, in terms of anomaly scores, the window granularity of the frequency domain is inherently distinct from the data-point granularity of the time domain. Owing to this discrepancy, the anomaly information in the frequency domain has not been utilized to its full potential for TSAD. In this paper, we propose a TSAD framework, Dual-TF, that simultaneously uses both the time and frequency domains while breaking the time-frequency granularity discrepancy. To this end, our framework employs nested-sliding windows, with the outer and inner windows responsible for the time and frequency domains, respectively, and aligns the anomaly scores of the two domains. As a result of the high resolution of the aligned scores, the boundaries of pattern-based anomalies can be identified more precisely. In six benchmark datasets, our framework outperforms state-of-the-art methods by 12.0–147%, as demonstrated by experimental results.

