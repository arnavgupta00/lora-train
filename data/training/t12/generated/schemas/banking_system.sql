PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_key INTEGER PRIMARY KEY,
    given_name TEXT NOT NULL,
    family_name TEXT NOT NULL,
    birth_date TEXT,
    city_name TEXT,
    risk_band INTEGER CHECK (risk_band BETWEEN 1 AND 5),
    onboard_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branch_locations (
    branch_key INTEGER PRIMARY KEY,
    branch_name TEXT NOT NULL,
    municipality TEXT NOT NULL,
    state_code TEXT NOT NULL,
    opened_on TEXT,
    service_tier INTEGER NOT NULL CHECK (service_tier BETWEEN 1 AND 4)
);

CREATE TABLE IF NOT EXISTS bank_accounts (
    account_key INTEGER PRIMARY KEY,
    branch_key INTEGER NOT NULL,
    account_number TEXT NOT NULL UNIQUE,
    account_type TEXT NOT NULL,
    currency_code TEXT NOT NULL DEFAULT 'USD',
    opened_on TEXT NOT NULL,
    closed_on TEXT,
    available_balance REAL NOT NULL,
    status_code INTEGER NOT NULL CHECK (status_code IN (0, 1, 2)),
    FOREIGN KEY (branch_key) REFERENCES branch_locations(branch_key)
);

CREATE TABLE IF NOT EXISTS account_holders (
    holder_link_key INTEGER PRIMARY KEY,
    account_key INTEGER NOT NULL,
    customer_key INTEGER NOT NULL,
    holder_role TEXT NOT NULL,
    can_withdraw INTEGER NOT NULL DEFAULT 1 CHECK (can_withdraw IN (0, 1)),
    relationship_start TEXT NOT NULL,
    relationship_end TEXT,
    FOREIGN KEY (account_key) REFERENCES bank_accounts(account_key),
    FOREIGN KEY (customer_key) REFERENCES customers(customer_key),
    UNIQUE (account_key, customer_key, holder_role)
);

CREATE TABLE IF NOT EXISTS credit_products (
    credit_key INTEGER PRIMARY KEY,
    account_key INTEGER NOT NULL,
    branch_key INTEGER NOT NULL,
    credit_type TEXT NOT NULL,
    principal_amount REAL NOT NULL CHECK (principal_amount >= 0),
    annual_rate REAL NOT NULL CHECK (annual_rate >= 0),
    issued_on TEXT NOT NULL,
    maturity_on TEXT,
    product_status INTEGER NOT NULL CHECK (product_status IN (0, 1, 2, 3)),
    FOREIGN KEY (account_key) REFERENCES bank_accounts(account_key),
    FOREIGN KEY (branch_key) REFERENCES branch_locations(branch_key)
);

CREATE TABLE IF NOT EXISTS payment_transactions (
    transaction_key INTEGER PRIMARY KEY,
    account_key INTEGER NOT NULL,
    branch_key INTEGER,
    transaction_time TEXT NOT NULL,
    transaction_category INTEGER NOT NULL CHECK (transaction_category BETWEEN 1 AND 9),
    direction_code INTEGER NOT NULL CHECK (direction_code IN (1, -1)),
    transaction_amount REAL NOT NULL CHECK (transaction_amount >= 0),
    balance_after REAL,
    channel_code INTEGER NOT NULL CHECK (channel_code IN (1, 2, 3, 4)),
    FOREIGN KEY (account_key) REFERENCES bank_accounts(account_key),
    FOREIGN KEY (branch_key) REFERENCES branch_locations(branch_key)
);

CREATE INDEX IF NOT EXISTS idx_accounts_branch ON bank_accounts(branch_key);
CREATE INDEX IF NOT EXISTS idx_holders_account ON account_holders(account_key);
CREATE INDEX IF NOT EXISTS idx_holders_customer ON account_holders(customer_key);
CREATE INDEX IF NOT EXISTS idx_credit_account ON credit_products(account_key);
CREATE INDEX IF NOT EXISTS idx_transactions_account_time ON payment_transactions(account_key, transaction_time);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON payment_transactions(transaction_category);
