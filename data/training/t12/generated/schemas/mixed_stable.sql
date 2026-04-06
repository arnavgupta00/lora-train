CREATE TABLE organizational_units (
    unit_key INTEGER PRIMARY KEY,
    unit_name TEXT NOT NULL,
    parent_unit_key INTEGER,
    location_city TEXT,
    annual_budget REAL,
    established_date DATE,
    FOREIGN KEY (parent_unit_key) REFERENCES organizational_units(unit_key)
);

CREATE TABLE staff_members (
    staff_key INTEGER PRIMARY KEY,
    unit_key INTEGER NOT NULL,
    manager_staff_key INTEGER,
    given_name TEXT NOT NULL,
    family_name TEXT NOT NULL,
    work_email TEXT UNIQUE,
    hire_date DATE,
    employment_status TEXT,
    salary_amount REAL,
    FOREIGN KEY (unit_key) REFERENCES organizational_units(unit_key),
    FOREIGN KEY (manager_staff_key) REFERENCES staff_members(staff_key)
);

CREATE TABLE client_accounts (
    client_key INTEGER PRIMARY KEY,
    account_name TEXT NOT NULL,
    industry_segment TEXT,
    country_code TEXT,
    account_tier TEXT,
    onboarded_on DATE,
    active_flag INTEGER NOT NULL DEFAULT 1 CHECK (active_flag IN (0,1))
);

CREATE TABLE work_projects (
    project_key INTEGER PRIMARY KEY,
    owning_unit_key INTEGER NOT NULL,
    client_key INTEGER,
    project_title TEXT NOT NULL,
    start_date DATE,
    target_end_date DATE,
    project_status TEXT,
    contracted_value REAL,
    FOREIGN KEY (owning_unit_key) REFERENCES organizational_units(unit_key),
    FOREIGN KEY (client_key) REFERENCES client_accounts(client_key)
);

CREATE TABLE task_assignments (
    assignment_key INTEGER PRIMARY KEY,
    project_key INTEGER NOT NULL,
    staff_key INTEGER NOT NULL,
    assigned_role TEXT,
    allocation_percent REAL CHECK (allocation_percent >= 0 AND allocation_percent <= 100),
    assigned_on DATE,
    due_on DATE,
    completion_state TEXT,
    FOREIGN KEY (project_key) REFERENCES work_projects(project_key),
    FOREIGN KEY (staff_key) REFERENCES staff_members(staff_key)
);

CREATE TABLE performance_reviews (
    review_key INTEGER PRIMARY KEY,
    staff_key INTEGER NOT NULL,
    reviewer_staff_key INTEGER,
    review_period TEXT NOT NULL,
    score_overall REAL,
    goals_met_count INTEGER,
    promotion_recommendation TEXT,
    review_submitted_on DATE,
    FOREIGN KEY (staff_key) REFERENCES staff_members(staff_key),
    FOREIGN KEY (reviewer_staff_key) REFERENCES staff_members(staff_key)
);
