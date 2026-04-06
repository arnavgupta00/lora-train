PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS regions_districts (
    district_key INTEGER PRIMARY KEY,
    district_label TEXT NOT NULL,
    county_label TEXT NOT NULL,
    locale_class TEXT NOT NULL,
    enrollment_total INTEGER NOT NULL CHECK (enrollment_total >= 0),
    household_poverty_rate REAL CHECK (household_poverty_rate BETWEEN 0 AND 100),
    charter_flag INTEGER NOT NULL DEFAULT 0 CHECK (charter_flag IN (0, 1))
);

CREATE TABLE IF NOT EXISTS educational_institutions (
    institution_key INTEGER PRIMARY KEY,
    district_key INTEGER NOT NULL,
    institution_label TEXT NOT NULL,
    institution_kind TEXT NOT NULL,
    grade_span TEXT NOT NULL,
    opened_year INTEGER,
    active_status TEXT NOT NULL DEFAULT 'active',
    learner_count INTEGER NOT NULL CHECK (learner_count >= 0),
    attendance_rate REAL CHECK (attendance_rate BETWEEN 0 AND 100),
    FOREIGN KEY (district_key) REFERENCES regions_districts(district_key)
);

CREATE TABLE IF NOT EXISTS academic_programs (
    program_key INTEGER PRIMARY KEY,
    institution_key INTEGER NOT NULL,
    program_title TEXT NOT NULL,
    focus_area TEXT NOT NULL,
    target_grades TEXT,
    participant_count INTEGER NOT NULL CHECK (participant_count >= 0),
    completion_count INTEGER NOT NULL CHECK (completion_count >= 0),
    launch_date TEXT,
    FOREIGN KEY (institution_key) REFERENCES educational_institutions(institution_key)
);

CREATE TABLE IF NOT EXISTS assessment_results (
    result_key INTEGER PRIMARY KEY,
    institution_key INTEGER NOT NULL,
    school_year INTEGER NOT NULL,
    subject_name TEXT NOT NULL,
    tested_learners INTEGER NOT NULL CHECK (tested_learners >= 0),
    proficient_learners INTEGER NOT NULL CHECK (proficient_learners >= 0),
    mean_scaled_score REAL,
    chronic_absence_pct REAL CHECK (chronic_absence_pct BETWEEN 0 AND 100),
    UNIQUE (institution_key, school_year, subject_name),
    FOREIGN KEY (institution_key) REFERENCES educational_institutions(institution_key)
);

CREATE TABLE IF NOT EXISTS budget_allocations (
    allocation_key INTEGER PRIMARY KEY,
    district_key INTEGER,
    institution_key INTEGER,
    fiscal_year INTEGER NOT NULL,
    funding_stream TEXT NOT NULL,
    designated_amount REAL NOT NULL CHECK (designated_amount >= 0),
    spent_amount REAL CHECK (spent_amount >= 0),
    restricted_flag INTEGER NOT NULL DEFAULT 0 CHECK (restricted_flag IN (0, 1)),
    FOREIGN KEY (district_key) REFERENCES regions_districts(district_key),
    FOREIGN KEY (institution_key) REFERENCES educational_institutions(institution_key),
    CHECK (
        (district_key IS NOT NULL AND institution_key IS NULL)
        OR (district_key IS NULL AND institution_key IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_institutions_district ON educational_institutions(district_key);
CREATE INDEX IF NOT EXISTS idx_programs_institution ON academic_programs(institution_key);
CREATE INDEX IF NOT EXISTS idx_results_institution_year ON assessment_results(institution_key, school_year);
CREATE INDEX IF NOT EXISTS idx_budget_district_year ON budget_allocations(district_key, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_budget_institution_year ON budget_allocations(institution_key, fiscal_year);
