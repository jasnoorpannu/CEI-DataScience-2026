# Evaluation report

- Pipeline: model `tfidf-lr-v2-9f3b11c2aa41` v2.0.0 (sentence-transformers embeddings), sampled on 60 resumes.

## 1. Role classification
- Accuracy: **0.935**
- Macro F1: **0.955**
- Weighted F1: **0.933**
- *Classification metrics are in-sample (all resumes); held-out test accuracy is recorded in models/metadata.json.*

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AI Engineer | 1.000 | 1.000 | 1.000 | 71 |
| Backend Developer | 1.000 | 1.000 | 1.000 | 76 |
| Blockchain | 1.000 | 0.745 | 0.854 | 47 |
| Blockchain Developer | 1.000 | 1.000 | 1.000 | 30 |
| Business Analyst | 0.892 | 0.987 | 0.937 | 150 |
| Cloud Engineer | 1.000 | 1.000 | 1.000 | 92 |
| Cybersecurity Analyst | 1.000 | 1.000 | 1.000 | 66 |
| Data Science | 0.896 | 0.995 | 0.943 | 200 |
| Database | 0.975 | 0.773 | 0.862 | 150 |
| Database Administrator | 1.000 | 1.000 | 1.000 | 56 |
| DevOps | 0.978 | 0.983 | 0.981 | 180 |
| Digital Media | 0.926 | 0.880 | 0.903 | 100 |
| DotNet Developer | 0.885 | 0.936 | 0.910 | 140 |
| ETL Developer | 0.836 | 0.892 | 0.863 | 120 |
| Engineering Manager | 1.000 | 1.000 | 1.000 | 30 |
| Frontend Developer | 1.000 | 1.000 | 1.000 | 76 |
| Full Stack Developer | 1.000 | 1.000 | 1.000 | 102 |
| Java Developer | 0.894 | 0.975 | 0.933 | 200 |
| Machine Learning Engineer | 1.000 | 1.000 | 1.000 | 81 |
| Mobile Developer | 1.000 | 1.000 | 1.000 | 46 |
| Network Security Engineer | 0.952 | 1.000 | 0.976 | 120 |
| Principal Engineer | 1.000 | 1.000 | 1.000 | 25 |
| Product Manager | 1.000 | 1.000 | 1.000 | 25 |
| Python Developer | 0.878 | 0.825 | 0.851 | 200 |
| QA Engineer | 1.000 | 1.000 | 1.000 | 61 |
| React Developer | 0.841 | 0.633 | 0.722 | 150 |
| SAP Developer | 0.990 | 0.970 | 0.980 | 100 |
| SQL Developer | 0.830 | 0.922 | 0.874 | 180 |
| Site Reliability Engineer | 1.000 | 1.000 | 1.000 | 46 |
| Software Developer | 1.000 | 1.000 | 1.000 | 134 |
| System Administrator | 1.000 | 1.000 | 1.000 | 40 |
| Technical Lead | 1.000 | 1.000 | 1.000 | 35 |
| Technical Writer | 1.000 | 1.000 | 1.000 | 20 |
| Testing | 0.877 | 0.947 | 0.910 | 150 |
| UI/UX Designer | 1.000 | 1.000 | 1.000 | 51 |
| Web Designing | 0.928 | 0.853 | 0.889 | 150 |

## 2. Resume-JD matching (ranking quality)
- NDCG@10: **1.000**
- Hit@1 (correct role in top prediction): **0.900**
- Pairwise accuracy: **0.827**
- Mean overall score: **65.6**

## 3. Retrieval quality (similar-profile search)
- Precision@5: **0.870**
- Recall@5: **0.043**
- MRR@10: **0.981**
- Same-category rate @6: **0.787**
