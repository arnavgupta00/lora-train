---
license: cc-by-sa-4.0
task_categories:
- table-question-answering
- question-answering
language:
- en
size_categories:
- 1K<n<10K
---

# BIRD-SQL Dev

## 🆕 Update 2025-11-06

We would like to express our sincere gratitude to the community for their continuous support and constructive feedback on the **BIRD-SQL Dev** dataset. Over the past year, we have received valuable suggestions through GitHub discussions, emails, and user reports. Based on these insights, we organized a quality review program led by a team of five PhD researchers in Data Science and AI, supported by a globally distributed group of industry engineers with over 10 years of experience and master’s students in DS/AI. The team systematically reviewed all instances in BIRD Dev to minimize ambiguity and correct errors, ensuring improved clarity, consistency, and reliability throughout the dataset. Please note that all questions in BIRD SQL were written by experienced native speakers who received BI training in text-to-SQL annotation. While we have made significant efforts to minimize ambiguity, it remains an inherent feature of natural language and NLP research, reflecting the realistic challenges of interpreting human questions in database contexts. To further address this, we will follow the design of [BIRD-Interact](https://bird-interact.github.io/) and introduce an interactive, clarification-based setting as a new part of our leaderboard in the future, enabling models to handle ambiguity through dynamic interactions and clarification dialogues.

### 🔍 What’s New
This update focuses on improving the overall clarity, correctness, and consistency of the dataset.  
In this release, we have:
- Refined **questions** to remove ambiguity and improve natural-language clarity while preserving their original meaning.  
- Revised **evidence** descriptions to make them concise, accurate, and properly scoped.  
- Corrected **SQL queries** to ensure syntactic validity, logical consistency, and successful execution on all released databases.  

---


### 📥 For New Users
If you are new to **BIRD Dev**, you can download the complete databases using the following link:  
[Download BIRD Dev Complete Package](https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view?usp=sharing)
Then you can pull the dataset from Hugging Face:
```python
from datasets import load_dataset

dataset = load_dataset("birdsql/bird_sql_dev_20251106")
print(dataset["dev_20251106"][0])
```

### 🔄 For Existing Users
If you have already downloaded the BIRD databases, you can pull the latest data updates through Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("birdsql/bird_sql_dev_20251106")
print(dataset["dev_20251106"][0])
```

## 🧱 Dataset Fields

Each entry in **BIRD-SQL Dev** is a JSON object with the following structure:

| Field | Type | Description |
|:------|:-----|:-------------|
| `question_id` | `int` | Unique identifier for each instance. |
| `db_id` | `string` | Database name corresponding to a SQLite file. |
| `question` | `string` | Natural-language question posed by the user. |
| `evidence` | `string` or `null` | Supporting information or definitions needed to interpret the question. |
| `SQL` | `string` | Ground-truth SQL query verified to execute successfully. |
| `difficulty` | `string` | Difficulty level — one of `simple`, `moderate`, or `challenging`. |

### Example
```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "For the school with the highest free meal rate in Alameda County, what are its characteristics including whether it's a charter school, what grades it serves, its SAT performance level, and how much its free meal rate deviates from the county average?",
  "evidence": "Free meal rate = Free Meal Count (K-12) / Enrollment (K-12). SAT performance levels are categorized as: Below Average (total score < 1200), Average (1200-1500), Above Average (> 1500), or No SAT Data if unavailable.",
  "SQL": "WITH CountyStats AS (\n    SELECT \n        f.`County Name`,\n        f.`School Name`,\n        f.`Free Meal Count (K-12)`,\n        f.`Enrollment (K-12)`,\n        CAST(f.`Free Meal Count (K-12)` AS REAL) / f.`Enrollment (K-12)` AS FreeRate,\n        s.sname,\n        s.AvgScrRead,\n        s.AvgScrMath,\n        s.AvgScrWrite,\n        (s.AvgScrRead + s.AvgScrMath + s.AvgScrWrite) AS TotalSATScore,\n        sc.Charter,\n        sc.GSserved,\n        RANK() OVER (PARTITION BY f.`County Name` ORDER BY CAST(f.`Free Meal Count (K-12)` AS REAL) / f.`Enrollment (K-12)` DESC) AS CountyRank\n    FROM frpm f\n    LEFT JOIN schools sc ON f.CDSCode = sc.CDSCode\n    LEFT JOIN satscores s ON f.CDSCode = s.cds\n    WHERE f.`Enrollment (K-12)` > 0 \n    AND f.`County Name` = 'Alameda'\n)\nSELECT \n    cs.`County Name` AS County,\n    cs.`School Name`,\n    cs.FreeRate AS HighestFreeRate,\n    cs.`Free Meal Count (K-12)` AS FreeMealCount,\n    cs.`Enrollment (K-12)` AS TotalEnrollment,\n    CASE \n        WHEN cs.Charter = 1 THEN 'Yes'\n        WHEN cs.Charter = 0 THEN 'No'\n        ELSE 'Unknown'\n    END AS IsCharterSchool,\n    cs.GSserved AS GradesServed,\n    CASE\n        WHEN cs.TotalSATScore IS NULL THEN 'No SAT Data'\n        WHEN cs.TotalSATScore < 1200 THEN 'Below Average'\n        WHEN cs.TotalSATScore BETWEEN 1200 AND 1500 THEN 'Average'\n        ELSE 'Above Average'\n    END AS SATPerformance,\n    (SELECT AVG(CAST(f2.`Free Meal Count (K-12)` AS REAL) / f2.`Enrollment (K-12)`)\n     FROM frpm f2\n     WHERE f2.`County Name` = 'Alameda' AND f2.`Enrollment (K-12)` > 0) AS CountyAverageFreeRate,\n    cs.FreeRate - (SELECT AVG(CAST(f2.`Free Meal Count (K-12)` AS REAL) / f2.`Enrollment (K-12)`)\n                  FROM frpm f2\n                  WHERE f2.`County Name` = 'Alameda' AND f2.`Enrollment (K-12)` > 0) AS DeviationFromCountyAverage\nFROM CountyStats cs\nWHERE cs.CountyRank = 1\nORDER BY cs.FreeRate DESC\nLIMIT 1;",
  "difficulty": "challenging"
}
```

## 📊 Baseline performance on Dev and Test Dataset (EX)


![chart_performance](https://cdn-uploads.huggingface.co/production/uploads/653693cb8ee17cfd44eed8ce/HIXnXOnZJLDMnDlpD8y6M.png)


| Model                         | Dev 1106 | Test  |
| ----------------------------- | -------- | ----- |
| gemini-3-pro-preview          | **68.97**    | **70.43** |
| claude-sonnet-4.5             | 66.56    | 67.02 |
| gemini-2.0-flash-001          | 63.62    | 66.74 |
| qwen3-coder-480b-a35b         | 65.45    | 66.46 |
| GPT-5.1                       | 64.02    | 65.96 |
| gemini-2.5-flash              | 65.91    | 65.34 |
| claude-sonnet-4               | 64.86    | 64.39 |
| gpt-5-2025-08-07              | 63.30    | 64.34 |
| Qwen3-235B-A22B-Thinking-2507 | 61.60    | 64.00 |
| Qwen3-30B-A3B-Instruct-2507   | 63.17    | 63.89 |
| Llama-3.1-70B-Instruct        | 59.39    | 63.00 |
| claude-4-5-haiku              | 60.69    | 62.72 |
| Qwen2.5-Coder-14B-Instruct    | 57.04    | 58.86 |
| Qwen2.5-Coder-32B-Instruct    | 60.95    | 58.36 |
| Qwen2.5-Coder-7B-Instruct     | 49.22    | 54.11 |
| Llama-3.1-8B-Instruct         | 36.70    | 41.08 |

We adapt data processing and the prompt from the [Arctic-Text2SQL-R1 project ](https://www.snowflake.com/en/product/ai/ai-research/). You can find the original [repo](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/arctic_text2sql_r1) and [paper](https://arxiv.org/abs/2505.20315) here. 

## 🙌 Acknowledgement
We sincerely thank the participating members for their time and dedication in improving this release: Benjamin Jun-jie Glover, Pan Enze, Rain Yiran Xu, Ashley (Juyeon) Lee, Eric Yue Wu, Yu Kaijia, Ziye Luo, Tangpirul Tat, Chik Ki Lok, Xu Haosen, Zhao Mingze, Chen Bingshang, Huang Yingrui, Winiera Sutanto, Zhan Mohan, Leia (Heaju) Kim, Veren Florecita, Xu Zixi, Chui Ting Yu George, Annabel Leonardi, Divyansh Tulsyan, Sun Manqi and Liu Zhengyang.


We also appreciate the continuous support and feedback from the open community, including GitHub reviewers (@element154, @NL2SQL-Empirical, @erikskalnes), anonymous users, and those who reached out to us via email, such as Arcwise AI (@hansonw), for their valuable suggestions.

## 📝 Citation
Please cite the repo if you think our work is helpful to you.
```
@article{li2024can,
  title={Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls},
  author={Li, Jinyang and Hui, Binyuan and Qu, Ge and Yang, Jiaxi and Li, Binhua and Li, Bowen and Wang, Bailin and Qin, Bowen and Geng, Ruiying and Huo, Nan and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

---

### ✅ TODOs

- [x] Release updated **Dev data**
- [x] Release **baseline results** on new Dev set
- [x] Release **baseline results** on Test set
- [ ] Integrate **interactive** setting into leaderboard

