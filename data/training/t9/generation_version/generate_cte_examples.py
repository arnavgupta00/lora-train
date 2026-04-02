#!/usr/bin/env python3
import json
import random

# Schema definitions
ECOMMERCE_SCHEMA = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT, created_at TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, total_amount REAL, status TEXT);
CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, price REAL);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);"""

UNIVERSITY_SCHEMA = """CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT, major TEXT, gpa REAL, year INTEGER);
CREATE TABLE courses (course_id TEXT PRIMARY KEY, title TEXT, department TEXT, credits INTEGER);
CREATE TABLE enrollments (student_id INTEGER, course_id TEXT, semester TEXT, grade TEXT);
CREATE TABLE professors (prof_id INTEGER PRIMARY KEY, name TEXT, department TEXT);"""

HOSPITAL_SCHEMA = """CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, name TEXT, dob TEXT, gender TEXT);
CREATE TABLE doctors (doctor_id INTEGER PRIMARY KEY, name TEXT, specialty TEXT, department TEXT);
CREATE TABLE appointments (appt_id INTEGER PRIMARY KEY, patient_id INTEGER, doctor_id INTEGER, date TEXT, diagnosis TEXT);
CREATE TABLE prescriptions (rx_id INTEGER PRIMARY KEY, appt_id INTEGER, medication TEXT, dosage TEXT);"""

HR_SCHEMA = """CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, hire_date TEXT, manager_id INTEGER);
CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, name TEXT, budget REAL, location TEXT);
CREATE TABLE projects (proj_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, start_date TEXT, end_date TEXT);
CREATE TABLE assignments (emp_id INTEGER, proj_id INTEGER, role TEXT, hours INTEGER);"""

# Simple CTE examples (40% = 120 examples, 30 per schema)
SIMPLE_CTE_TEMPLATES = {
    "ecommerce": [
        ("Find all orders placed by customers from the USA", 
         "WITH usa_customers AS (\n  SELECT customer_id FROM customers WHERE country = 'USA'\n)\nSELECT o.* FROM orders o WHERE o.customer_id IN (SELECT customer_id FROM usa_customers);"),
        
        ("Show products that have been ordered at least once",
         "WITH ordered_products AS (\n  SELECT DISTINCT product_id FROM order_items\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM ordered_products);"),
        
        ("List customers who have placed orders with status 'completed'",
         "WITH completed_orders AS (\n  SELECT DISTINCT customer_id FROM orders WHERE status = 'completed'\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM completed_orders);"),
        
        ("Find all orders containing electronics products",
         "WITH electronics AS (\n  SELECT product_id FROM products WHERE category = 'Electronics'\n)\nSELECT DISTINCT o.* FROM orders o JOIN order_items oi ON o.order_id = oi.order_id WHERE oi.product_id IN (SELECT product_id FROM electronics);"),
        
        ("Show customers who created accounts in 2023",
         "WITH new_customers AS (\n  SELECT customer_id, name, email FROM customers WHERE created_at LIKE '2023%'\n)\nSELECT * FROM new_customers;"),
        
        ("Find products that are out of stock",
         "WITH out_of_stock AS (\n  SELECT product_id, name, category FROM products WHERE stock = 0\n)\nSELECT * FROM out_of_stock;"),
        
        ("List orders with total amount greater than 500",
         "WITH high_value_orders AS (\n  SELECT order_id, customer_id, total_amount FROM orders WHERE total_amount > 500\n)\nSELECT * FROM high_value_orders;"),
        
        ("Show all pending orders",
         "WITH pending AS (\n  SELECT * FROM orders WHERE status = 'pending'\n)\nSELECT * FROM pending;"),
        
        ("Find customers from Canada or UK",
         "WITH target_countries AS (\n  SELECT * FROM customers WHERE country IN ('Canada', 'UK')\n)\nSELECT * FROM target_countries;"),
        
        ("List products in the Books category",
         "WITH books AS (\n  SELECT * FROM products WHERE category = 'Books'\n)\nSELECT * FROM books;"),
        
        ("Show orders placed in January 2024",
         "WITH jan_orders AS (\n  SELECT * FROM orders WHERE order_date LIKE '2024-01%'\n)\nSELECT * FROM jan_orders;"),
        
        ("Find products priced under 50",
         "WITH affordable AS (\n  SELECT * FROM products WHERE price < 50\n)\nSELECT * FROM affordable;"),
        
        ("List customers with gmail addresses",
         "WITH gmail_users AS (\n  SELECT * FROM customers WHERE email LIKE '%@gmail.com'\n)\nSELECT * FROM gmail_users;"),
        
        ("Show order items with quantity greater than 5",
         "WITH bulk_items AS (\n  SELECT * FROM order_items WHERE quantity > 5\n)\nSELECT * FROM bulk_items;"),
        
        ("Find cancelled orders",
         "WITH cancelled AS (\n  SELECT * FROM orders WHERE status = 'cancelled'\n)\nSELECT * FROM cancelled;"),
        
        ("List products with stock above 100",
         "WITH well_stocked AS (\n  SELECT * FROM products WHERE stock > 100\n)\nSELECT * FROM well_stocked;"),
        
        ("Show customers from European countries (UK, Germany, France)",
         "WITH european_customers AS (\n  SELECT * FROM customers WHERE country IN ('UK', 'Germany', 'France')\n)\nSELECT * FROM european_customers;"),
        
        ("Find orders from 2023",
         "WITH orders_2023 AS (\n  SELECT * FROM orders WHERE order_date LIKE '2023%'\n)\nSELECT * FROM orders_2023;"),
        
        ("List products in Clothing category",
         "WITH clothing AS (\n  SELECT * FROM products WHERE category = 'Clothing'\n)\nSELECT * FROM clothing;"),
        
        ("Show order items priced over 100",
         "WITH expensive_items AS (\n  SELECT * FROM order_items WHERE price > 100\n)\nSELECT * FROM expensive_items;"),
        
        ("Find shipped orders",
         "WITH shipped AS (\n  SELECT * FROM orders WHERE status = 'shipped'\n)\nSELECT * FROM shipped;"),
        
        ("List customers created before 2022",
         "WITH old_customers AS (\n  SELECT * FROM customers WHERE created_at < '2022-01-01'\n)\nSELECT * FROM old_customers;"),
        
        ("Show products in Sports category",
         "WITH sports_products AS (\n  SELECT * FROM products WHERE category = 'Sports'\n)\nSELECT * FROM sports_products;"),
        
        ("Find orders with total amount under 100",
         "WITH small_orders AS (\n  SELECT * FROM orders WHERE total_amount < 100\n)\nSELECT * FROM small_orders;"),
        
        ("List products priced between 50 and 200",
         "WITH mid_range AS (\n  SELECT * FROM products WHERE price BETWEEN 50 AND 200\n)\nSELECT * FROM mid_range;"),
        
        ("Show customers from USA or Canada",
         "WITH north_america AS (\n  SELECT * FROM customers WHERE country IN ('USA', 'Canada')\n)\nSELECT * FROM north_america;"),
        
        ("Find orders placed in Q1 2024",
         "WITH q1_orders AS (\n  SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31'\n)\nSELECT * FROM q1_orders;"),
        
        ("List products with low stock (less than 10)",
         "WITH low_stock AS (\n  SELECT * FROM products WHERE stock < 10\n)\nSELECT * FROM low_stock;"),
        
        ("Show order items for order 100",
         "WITH items_100 AS (\n  SELECT * FROM order_items WHERE order_id = 100\n)\nSELECT * FROM items_100;"),
        
        ("Find customers who joined in the last year",
         "WITH recent_customers AS (\n  SELECT * FROM customers WHERE created_at > '2023-01-01'\n)\nSELECT * FROM recent_customers;"),
    ],
    "university": [
        ("Find all students majoring in Computer Science",
         "WITH cs_students AS (\n  SELECT * FROM students WHERE major = 'Computer Science'\n)\nSELECT * FROM cs_students;"),
        
        ("Show courses offered by the Mathematics department",
         "WITH math_courses AS (\n  SELECT * FROM courses WHERE department = 'Mathematics'\n)\nSELECT * FROM math_courses;"),
        
        ("List students with GPA above 3.5",
         "WITH high_achievers AS (\n  SELECT * FROM students WHERE gpa > 3.5\n)\nSELECT * FROM high_achievers;"),
        
        ("Find all senior students (year 4)",
         "WITH seniors AS (\n  SELECT * FROM students WHERE year = 4\n)\nSELECT * FROM seniors;"),
        
        ("Show enrollments from Fall 2023 semester",
         "WITH fall_2023 AS (\n  SELECT * FROM enrollments WHERE semester = 'Fall 2023'\n)\nSELECT * FROM fall_2023;"),
        
        ("List professors in Computer Science department",
         "WITH cs_profs AS (\n  SELECT * FROM professors WHERE department = 'Computer Science'\n)\nSELECT * FROM cs_profs;"),
        
        ("Find 3-credit courses",
         "WITH three_credit AS (\n  SELECT * FROM courses WHERE credits = 3\n)\nSELECT * FROM three_credit;"),
        
        ("Show students who got an A grade",
         "WITH a_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'A'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM a_students);"),
        
        ("List freshman students (year 1)",
         "WITH freshmen AS (\n  SELECT * FROM students WHERE year = 1\n)\nSELECT * FROM freshmen;"),
        
        ("Find courses in the Physics department",
         "WITH physics AS (\n  SELECT * FROM courses WHERE department = 'Physics'\n)\nSELECT * FROM physics;"),
        
        ("Show students with GPA below 2.0",
         "WITH struggling AS (\n  SELECT * FROM students WHERE gpa < 2.0\n)\nSELECT * FROM struggling;"),
        
        ("List enrollments from Spring 2024",
         "WITH spring_2024 AS (\n  SELECT * FROM enrollments WHERE semester = 'Spring 2024'\n)\nSELECT * FROM spring_2024;"),
        
        ("Find Biology majors",
         "WITH bio_students AS (\n  SELECT * FROM students WHERE major = 'Biology'\n)\nSELECT * FROM bio_students;"),
        
        ("Show 4-credit courses",
         "WITH four_credit AS (\n  SELECT * FROM courses WHERE credits = 4\n)\nSELECT * FROM four_credit;"),
        
        ("List professors in Engineering department",
         "WITH eng_profs AS (\n  SELECT * FROM professors WHERE department = 'Engineering'\n)\nSELECT * FROM eng_profs;"),
        
        ("Find sophomore students (year 2)",
         "WITH sophomores AS (\n  SELECT * FROM students WHERE year = 2\n)\nSELECT * FROM sophomores;"),
        
        ("Show students with perfect GPA (4.0)",
         "WITH perfect_gpa AS (\n  SELECT * FROM students WHERE gpa = 4.0\n)\nSELECT * FROM perfect_gpa;"),
        
        ("List Chemistry department courses",
         "WITH chem_courses AS (\n  SELECT * FROM courses WHERE department = 'Chemistry'\n)\nSELECT * FROM chem_courses;"),
        
        ("Find students who failed a course (grade F)",
         "WITH failed AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'F'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM failed);"),
        
        ("Show junior students (year 3)",
         "WITH juniors AS (\n  SELECT * FROM students WHERE year = 3\n)\nSELECT * FROM juniors;"),
        
        ("List History majors",
         "WITH history_students AS (\n  SELECT * FROM students WHERE major = 'History'\n)\nSELECT * FROM history_students;"),
        
        ("Find professors in Business department",
         "WITH biz_profs AS (\n  SELECT * FROM professors WHERE department = 'Business'\n)\nSELECT * FROM biz_profs;"),
        
        ("Show students with GPA between 3.0 and 3.5",
         "WITH good_students AS (\n  SELECT * FROM students WHERE gpa BETWEEN 3.0 AND 3.5\n)\nSELECT * FROM good_students;"),
        
        ("List enrollments with B grade",
         "WITH b_grades AS (\n  SELECT * FROM enrollments WHERE grade = 'B'\n)\nSELECT * FROM b_grades;"),
        
        ("Find English department courses",
         "WITH english AS (\n  SELECT * FROM courses WHERE department = 'English'\n)\nSELECT * FROM english;"),
        
        ("Show Economics majors",
         "WITH econ_students AS (\n  SELECT * FROM students WHERE major = 'Economics'\n)\nSELECT * FROM econ_students;"),
        
        ("List 2-credit courses",
         "WITH two_credit AS (\n  SELECT * FROM courses WHERE credits = 2\n)\nSELECT * FROM two_credit;"),
        
        ("Find students with GPA above 3.8",
         "WITH top_students AS (\n  SELECT * FROM students WHERE gpa > 3.8\n)\nSELECT * FROM top_students;"),
        
        ("Show enrollments from Summer 2023",
         "WITH summer_2023 AS (\n  SELECT * FROM enrollments WHERE semester = 'Summer 2023'\n)\nSELECT * FROM summer_2023;"),
        
        ("List Art majors",
         "WITH art_students AS (\n  SELECT * FROM students WHERE major = 'Art'\n)\nSELECT * FROM art_students;"),
    ],
    "hospital": [
        ("Find all male patients",
         "WITH male_patients AS (\n  SELECT * FROM patients WHERE gender = 'M'\n)\nSELECT * FROM male_patients;"),
        
        ("Show doctors specializing in Cardiology",
         "WITH cardiologists AS (\n  SELECT * FROM doctors WHERE specialty = 'Cardiology'\n)\nSELECT * FROM cardiologists;"),
        
        ("List appointments scheduled for today",
         "WITH today_appts AS (\n  SELECT * FROM appointments WHERE date = DATE('now')\n)\nSELECT * FROM today_appts;"),
        
        ("Find patients born before 1970",
         "WITH older_patients AS (\n  SELECT * FROM patients WHERE dob < '1970-01-01'\n)\nSELECT * FROM older_patients;"),
        
        ("Show doctors in Emergency department",
         "WITH er_doctors AS (\n  SELECT * FROM doctors WHERE department = 'Emergency'\n)\nSELECT * FROM er_doctors;"),
        
        ("List prescriptions for Aspirin",
         "WITH aspirin_rx AS (\n  SELECT * FROM prescriptions WHERE medication = 'Aspirin'\n)\nSELECT * FROM aspirin_rx;"),
        
        ("Find female patients",
         "WITH female_patients AS (\n  SELECT * FROM patients WHERE gender = 'F'\n)\nSELECT * FROM female_patients;"),
        
        ("Show appointments with diagnosis of Flu",
         "WITH flu_cases AS (\n  SELECT * FROM appointments WHERE diagnosis = 'Flu'\n)\nSELECT * FROM flu_cases;"),
        
        ("List doctors specializing in Pediatrics",
         "WITH pediatricians AS (\n  SELECT * FROM doctors WHERE specialty = 'Pediatrics'\n)\nSELECT * FROM pediatricians;"),
        
        ("Find patients born after 2000",
         "WITH young_patients AS (\n  SELECT * FROM patients WHERE dob > '2000-01-01'\n)\nSELECT * FROM young_patients;"),
        
        ("Show doctors in Surgery department",
         "WITH surgeons AS (\n  SELECT * FROM doctors WHERE department = 'Surgery'\n)\nSELECT * FROM surgeons;"),
        
        ("List appointments in January 2024",
         "WITH jan_appts AS (\n  SELECT * FROM appointments WHERE date LIKE '2024-01%'\n)\nSELECT * FROM jan_appts;"),
        
        ("Find prescriptions with high dosage (500mg)",
         "WITH high_dose AS (\n  SELECT * FROM prescriptions WHERE dosage = '500mg'\n)\nSELECT * FROM high_dose;"),
        
        ("Show Orthopedics specialists",
         "WITH ortho_docs AS (\n  SELECT * FROM doctors WHERE specialty = 'Orthopedics'\n)\nSELECT * FROM ortho_docs;"),
        
        ("List patients born in the 1980s",
         "WITH eighties_babies AS (\n  SELECT * FROM patients WHERE dob BETWEEN '1980-01-01' AND '1989-12-31'\n)\nSELECT * FROM eighties_babies;"),
        
        ("Find appointments with Diabetes diagnosis",
         "WITH diabetes_cases AS (\n  SELECT * FROM appointments WHERE diagnosis = 'Diabetes'\n)\nSELECT * FROM diabetes_cases;"),
        
        ("Show doctors in Radiology department",
         "WITH radiologists AS (\n  SELECT * FROM doctors WHERE department = 'Radiology'\n)\nSELECT * FROM radiologists;"),
        
        ("List prescriptions for Ibuprofen",
         "WITH ibuprofen_rx AS (\n  SELECT * FROM prescriptions WHERE medication = 'Ibuprofen'\n)\nSELECT * FROM ibuprofen_rx;"),
        
        ("Find Neurology specialists",
         "WITH neurologists AS (\n  SELECT * FROM doctors WHERE specialty = 'Neurology'\n)\nSELECT * FROM neurologists;"),
        
        ("Show patients born between 1990 and 2000",
         "WITH nineties_patients AS (\n  SELECT * FROM patients WHERE dob BETWEEN '1990-01-01' AND '2000-12-31'\n)\nSELECT * FROM nineties_patients;"),
        
        ("List appointments in February 2024",
         "WITH feb_appts AS (\n  SELECT * FROM appointments WHERE date LIKE '2024-02%'\n)\nSELECT * FROM feb_appts;"),
        
        ("Find doctors in Oncology department",
         "WITH oncologists AS (\n  SELECT * FROM doctors WHERE department = 'Oncology'\n)\nSELECT * FROM oncologists;"),
        
        ("Show prescriptions with 250mg dosage",
         "WITH med_dose AS (\n  SELECT * FROM prescriptions WHERE dosage = '250mg'\n)\nSELECT * FROM med_dose;"),
        
        ("List Dermatology specialists",
         "WITH dermatologists AS (\n  SELECT * FROM doctors WHERE specialty = 'Dermatology'\n)\nSELECT * FROM dermatologists;"),
        
        ("Find appointments with Hypertension diagnosis",
         "WITH hypertension AS (\n  SELECT * FROM appointments WHERE diagnosis = 'Hypertension'\n)\nSELECT * FROM hypertension;"),
        
        ("Show patients born before 1960",
         "WITH senior_patients AS (\n  SELECT * FROM patients WHERE dob < '1960-01-01'\n)\nSELECT * FROM senior_patients;"),
        
        ("List doctors in Psychiatry department",
         "WITH psychiatrists AS (\n  SELECT * FROM doctors WHERE department = 'Psychiatry'\n)\nSELECT * FROM psychiatrists;"),
        
        ("Find prescriptions for Metformin",
         "WITH metformin_rx AS (\n  SELECT * FROM prescriptions WHERE medication = 'Metformin'\n)\nSELECT * FROM metformin_rx;"),
        
        ("Show appointments from March 2024",
         "WITH mar_appts AS (\n  SELECT * FROM appointments WHERE date LIKE '2024-03%'\n)\nSELECT * FROM mar_appts;"),
        
        ("List Ophthalmology specialists",
         "WITH eye_doctors AS (\n  SELECT * FROM doctors WHERE specialty = 'Ophthalmology'\n)\nSELECT * FROM eye_doctors;"),
    ],
    "hr": [
        ("Find employees in Engineering department",
         "WITH eng_emps AS (\n  SELECT * FROM employees WHERE department = 'Engineering'\n)\nSELECT * FROM eng_emps;"),
        
        ("Show employees with salary above 80000",
         "WITH high_earners AS (\n  SELECT * FROM employees WHERE salary > 80000\n)\nSELECT * FROM high_earners;"),
        
        ("List departments with budget over 1 million",
         "WITH large_depts AS (\n  SELECT * FROM departments WHERE budget > 1000000\n)\nSELECT * FROM large_depts;"),
        
        ("Find employees hired in 2023",
         "WITH new_hires AS (\n  SELECT * FROM employees WHERE hire_date LIKE '2023%'\n)\nSELECT * FROM new_hires;"),
        
        ("Show projects that started in 2024",
         "WITH current_projects AS (\n  SELECT * FROM projects WHERE start_date LIKE '2024%'\n)\nSELECT * FROM current_projects;"),
        
        ("List employees in Sales department",
         "WITH sales_team AS (\n  SELECT * FROM employees WHERE department = 'Sales'\n)\nSELECT * FROM sales_team;"),
        
        ("Find departments located in New York",
         "WITH ny_depts AS (\n  SELECT * FROM departments WHERE location = 'New York'\n)\nSELECT * FROM ny_depts;"),
        
        ("Show employees with salary below 50000",
         "WITH lower_paid AS (\n  SELECT * FROM employees WHERE salary < 50000\n)\nSELECT * FROM lower_paid;"),
        
        ("List assignments with role 'Lead'",
         "WITH leads AS (\n  SELECT * FROM assignments WHERE role = 'Lead'\n)\nSELECT * FROM leads;"),
        
        ("Find projects ending in 2024",
         "WITH ending_projects AS (\n  SELECT * FROM projects WHERE end_date LIKE '2024%'\n)\nSELECT * FROM ending_projects;"),
        
        ("Show employees in Marketing department",
         "WITH marketing_team AS (\n  SELECT * FROM employees WHERE department = 'Marketing'\n)\nSELECT * FROM marketing_team;"),
        
        ("List employees hired before 2020",
         "WITH veteran_emps AS (\n  SELECT * FROM employees WHERE hire_date < '2020-01-01'\n)\nSELECT * FROM veteran_emps;"),
        
        ("Find departments with budget under 500000",
         "WITH small_depts AS (\n  SELECT * FROM departments WHERE budget < 500000\n)\nSELECT * FROM small_depts;"),
        
        ("Show assignments with over 100 hours",
         "WITH intensive_work AS (\n  SELECT * FROM assignments WHERE hours > 100\n)\nSELECT * FROM intensive_work;"),
        
        ("List employees in HR department",
         "WITH hr_team AS (\n  SELECT * FROM employees WHERE department = 'HR'\n)\nSELECT * FROM hr_team;"),
        
        ("Find projects that started in 2023",
         "WITH projects_2023 AS (\n  SELECT * FROM projects WHERE start_date LIKE '2023%'\n)\nSELECT * FROM projects_2023;"),
        
        ("Show employees with salary between 60000 and 80000",
         "WITH mid_earners AS (\n  SELECT * FROM employees WHERE salary BETWEEN 60000 AND 80000\n)\nSELECT * FROM mid_earners;"),
        
        ("List departments in San Francisco",
         "WITH sf_depts AS (\n  SELECT * FROM departments WHERE location = 'San Francisco'\n)\nSELECT * FROM sf_depts;"),
        
        ("Find employees in Finance department",
         "WITH finance_team AS (\n  SELECT * FROM employees WHERE department = 'Finance'\n)\nSELECT * FROM finance_team;"),
        
        ("Show assignments with role 'Developer'",
         "WITH developers AS (\n  SELECT * FROM assignments WHERE role = 'Developer'\n)\nSELECT * FROM developers;"),
        
        ("List projects ending in 2025",
         "WITH future_ends AS (\n  SELECT * FROM projects WHERE end_date LIKE '2025%'\n)\nSELECT * FROM future_ends;"),
        
        ("Find employees hired in 2022",
         "WITH hires_2022 AS (\n  SELECT * FROM employees WHERE hire_date LIKE '2022%'\n)\nSELECT * FROM hires_2022;"),
        
        ("Show departments with budget over 2 million",
         "WITH major_depts AS (\n  SELECT * FROM departments WHERE budget > 2000000\n)\nSELECT * FROM major_depts;"),
        
        ("List employees in Operations department",
         "WITH ops_team AS (\n  SELECT * FROM employees WHERE department = 'Operations'\n)\nSELECT * FROM ops_team;"),
        
        ("Find assignments with less than 50 hours",
         "WITH light_work AS (\n  SELECT * FROM assignments WHERE hours < 50\n)\nSELECT * FROM light_work;"),
        
        ("Show employees with salary over 100000",
         "WITH top_earners AS (\n  SELECT * FROM employees WHERE salary > 100000\n)\nSELECT * FROM top_earners;"),
        
        ("List departments in Chicago",
         "WITH chicago_depts AS (\n  SELECT * FROM departments WHERE location = 'Chicago'\n)\nSELECT * FROM chicago_depts;"),
        
        ("Find projects started before 2023",
         "WITH old_projects AS (\n  SELECT * FROM projects WHERE start_date < '2023-01-01'\n)\nSELECT * FROM old_projects;"),
        
        ("Show employees hired in 2021",
         "WITH hires_2021 AS (\n  SELECT * FROM employees WHERE hire_date LIKE '2021%'\n)\nSELECT * FROM hires_2021;"),
        
        ("List assignments with role 'Manager'",
         "WITH managers AS (\n  SELECT * FROM assignments WHERE role = 'Manager'\n)\nSELECT * FROM managers;"),
    ]
}

# Multiple CTEs examples (30% = 90 examples)
MULTIPLE_CTE_TEMPLATES = {
    "ecommerce": [
        ("Find customers who have both high total spending (over 1000) and recent orders (after 2023-06-01)",
         "WITH high_spenders AS (\n  SELECT customer_id FROM orders GROUP BY customer_id HAVING SUM(total_amount) > 1000\n),\nrecent_buyers AS (\n  SELECT DISTINCT customer_id FROM orders WHERE order_date > '2023-06-01'\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM high_spenders) AND c.customer_id IN (SELECT customer_id FROM recent_buyers);"),
        
        ("Show products that are both low in stock (under 20) and frequently ordered (in more than 10 orders)",
         "WITH low_stock AS (\n  SELECT product_id FROM products WHERE stock < 20\n),\npopular_products AS (\n  SELECT product_id FROM order_items GROUP BY product_id HAVING COUNT(DISTINCT order_id) > 10\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM low_stock) AND p.product_id IN (SELECT product_id FROM popular_products);"),
        
        ("List customers from USA who have completed orders",
         "WITH usa_customers AS (\n  SELECT customer_id FROM customers WHERE country = 'USA'\n),\ncompleted_orders AS (\n  SELECT DISTINCT customer_id FROM orders WHERE status = 'completed'\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM usa_customers) AND c.customer_id IN (SELECT customer_id FROM completed_orders);"),
        
        ("Find products in Electronics category that have been ordered more than 5 times",
         "WITH electronics AS (\n  SELECT product_id FROM products WHERE category = 'Electronics'\n),\nfrequent_items AS (\n  SELECT product_id FROM order_items GROUP BY product_id HAVING COUNT(*) > 5\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM electronics) AND p.product_id IN (SELECT product_id FROM frequent_items);"),
        
        ("Show customers who joined in 2023 and have placed orders worth over 500",
         "WITH new_customers AS (\n  SELECT customer_id FROM customers WHERE created_at LIKE '2023%'\n),\nbig_orders AS (\n  SELECT DISTINCT customer_id FROM orders WHERE total_amount > 500\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM new_customers) AND c.customer_id IN (SELECT customer_id FROM big_orders);"),
        
        ("List products in Books category that are in stock and priced under 30",
         "WITH books AS (\n  SELECT product_id FROM products WHERE category = 'Books'\n),\navailable_affordable AS (\n  SELECT product_id FROM products WHERE stock > 0 AND price < 30\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM books) AND p.product_id IN (SELECT product_id FROM available_affordable);"),
        
        ("Find orders from UK customers with pending status",
         "WITH uk_customers AS (\n  SELECT customer_id FROM customers WHERE country = 'UK'\n),\npending_orders AS (\n  SELECT order_id, customer_id FROM orders WHERE status = 'pending'\n)\nSELECT o.* FROM orders o WHERE o.customer_id IN (SELECT customer_id FROM uk_customers) AND o.order_id IN (SELECT order_id FROM pending_orders);"),
        
        ("Show products that are expensive (over 100) and have low stock (under 10)",
         "WITH expensive AS (\n  SELECT product_id FROM products WHERE price > 100\n),\nlow_stock AS (\n  SELECT product_id FROM products WHERE stock < 10\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM expensive) AND p.product_id IN (SELECT product_id FROM low_stock);"),
        
        ("List customers who have both cancelled and completed orders",
         "WITH cancelled_customers AS (\n  SELECT DISTINCT customer_id FROM orders WHERE status = 'cancelled'\n),\ncompleted_customers AS (\n  SELECT DISTINCT customer_id FROM orders WHERE status = 'completed'\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM cancelled_customers) AND c.customer_id IN (SELECT customer_id FROM completed_customers);"),
        
        ("Find orders placed in 2024 by customers from Canada",
         "WITH orders_2024 AS (\n  SELECT order_id, customer_id FROM orders WHERE order_date LIKE '2024%'\n),\ncanadian_customers AS (\n  SELECT customer_id FROM customers WHERE country = 'Canada'\n)\nSELECT o.* FROM orders o WHERE o.order_id IN (SELECT order_id FROM orders_2024) AND o.customer_id IN (SELECT customer_id FROM canadian_customers);"),
        
        ("Show products in Clothing category ordered more than once",
         "WITH clothing AS (\n  SELECT product_id FROM products WHERE category = 'Clothing'\n),\nrepeat_orders AS (\n  SELECT product_id FROM order_items GROUP BY product_id HAVING COUNT(*) > 1\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM clothing) AND p.product_id IN (SELECT product_id FROM repeat_orders);"),
        
        ("List customers with gmail addresses who have placed high-value orders (over 300)",
         "WITH gmail_users AS (\n  SELECT customer_id FROM customers WHERE email LIKE '%@gmail.com'\n),\nhigh_value AS (\n  SELECT DISTINCT customer_id FROM orders WHERE total_amount > 300\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM gmail_users) AND c.customer_id IN (SELECT customer_id FROM high_value);"),
        
        ("Find products that are both in Electronics and have stock over 50",
         "WITH electronics AS (\n  SELECT product_id FROM products WHERE category = 'Electronics'\n),\nwell_stocked AS (\n  SELECT product_id FROM products WHERE stock > 50\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM electronics) AND p.product_id IN (SELECT product_id FROM well_stocked);"),
        
        ("Show orders from European customers (UK, Germany, France) that are shipped",
         "WITH european_customers AS (\n  SELECT customer_id FROM customers WHERE country IN ('UK', 'Germany', 'France')\n),\nshipped_orders AS (\n  SELECT order_id, customer_id FROM orders WHERE status = 'shipped'\n)\nSELECT o.* FROM orders o WHERE o.customer_id IN (SELECT customer_id FROM european_customers) AND o.order_id IN (SELECT order_id FROM shipped_orders);"),
        
        ("List products priced between 50 and 150 that have been ordered",
         "WITH mid_priced AS (\n  SELECT product_id FROM products WHERE price BETWEEN 50 AND 150\n),\nordered_products AS (\n  SELECT DISTINCT product_id FROM order_items\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM mid_priced) AND p.product_id IN (SELECT product_id FROM ordered_products);"),
        
        ("Find customers from USA who have recent orders (after 2024-01-01) and high spending",
         "WITH usa_customers AS (\n  SELECT customer_id FROM customers WHERE country = 'USA'\n),\nrecent_high_spenders AS (\n  SELECT customer_id FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id HAVING SUM(total_amount) > 500\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM usa_customers) AND c.customer_id IN (SELECT customer_id FROM recent_high_spenders);"),
        
        ("Show products in Sports category with stock below 30",
         "WITH sports AS (\n  SELECT product_id FROM products WHERE category = 'Sports'\n),\nlow_stock AS (\n  SELECT product_id FROM products WHERE stock < 30\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM sports) AND p.product_id IN (SELECT product_id FROM low_stock);"),
        
        ("List old customers (before 2022) who have placed orders in 2024",
         "WITH old_customers AS (\n  SELECT customer_id FROM customers WHERE created_at < '2022-01-01'\n),\nrecent_orders AS (\n  SELECT DISTINCT customer_id FROM orders WHERE order_date LIKE '2024%'\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM old_customers) AND c.customer_id IN (SELECT customer_id FROM recent_orders);"),
        
        ("Find orders with high quantity items (over 10) and total amount over 200",
         "WITH high_qty_orders AS (\n  SELECT DISTINCT order_id FROM order_items WHERE quantity > 10\n),\nhigh_value_orders AS (\n  SELECT order_id FROM orders WHERE total_amount > 200\n)\nSELECT o.* FROM orders o WHERE o.order_id IN (SELECT order_id FROM high_qty_orders) AND o.order_id IN (SELECT order_id FROM high_value_orders);"),
        
        ("Show customers from Canada or UK who have completed more than 3 orders",
         "WITH target_countries AS (\n  SELECT customer_id FROM customers WHERE country IN ('Canada', 'UK')\n),\nfrequent_buyers AS (\n  SELECT customer_id FROM orders WHERE status = 'completed' GROUP BY customer_id HAVING COUNT(*) > 3\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM target_countries) AND c.customer_id IN (SELECT customer_id FROM frequent_buyers);"),
        
        ("List affordable products (under 40) in Books category that are in stock",
         "WITH affordable AS (\n  SELECT product_id FROM products WHERE price < 40\n),\nbooks_in_stock AS (\n  SELECT product_id FROM products WHERE category = 'Books' AND stock > 0\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM affordable) AND p.product_id IN (SELECT product_id FROM books_in_stock);"),
        
        ("Find customers who joined in 2023 and have not cancelled any orders",
         "WITH new_customers AS (\n  SELECT customer_id FROM customers WHERE created_at LIKE '2023%'\n),\nno_cancellations AS (\n  SELECT DISTINCT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders WHERE status = 'cancelled')\n)\nSELECT c.* FROM customers c WHERE c.customer_id IN (SELECT customer_id FROM new_customers) AND c.customer_id IN (SELECT customer_id FROM no_cancellations);"),
        
        ("Show products that are popular (ordered 20+ times) and well-stocked (over 100 units)",
         "WITH popular AS (\n  SELECT product_id FROM order_items GROUP BY product_id HAVING COUNT(*) > 20\n),\nwell_stocked AS (\n  SELECT product_id FROM products WHERE stock > 100\n)\nSELECT p.* FROM products p WHERE p.product_id IN (SELECT product_id FROM popular) AND p.product_id IN (SELECT product_id FROM well_stocked);"),
    ],
    "university": [
        ("Find Computer Science majors with GPA above 3.5",
         "WITH cs_students AS (\n  SELECT student_id FROM students WHERE major = 'Computer Science'\n),\nhigh_gpa AS (\n  SELECT student_id FROM students WHERE gpa > 3.5\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM cs_students) AND s.student_id IN (SELECT student_id FROM high_gpa);"),
        
        ("Show senior students (year 4) who got an A in at least one course",
         "WITH seniors AS (\n  SELECT student_id FROM students WHERE year = 4\n),\na_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'A'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM seniors) AND s.student_id IN (SELECT student_id FROM a_students);"),
        
        ("List 3-credit Computer Science courses",
         "WITH three_credit AS (\n  SELECT course_id FROM courses WHERE credits = 3\n),\ncs_courses AS (\n  SELECT course_id FROM courses WHERE department = 'Computer Science'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM three_credit) AND c.course_id IN (SELECT course_id FROM cs_courses);"),
        
        ("Find students enrolled in Fall 2023 who have GPA over 3.0",
         "WITH fall_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE semester = 'Fall 2023'\n),\ngood_gpa AS (\n  SELECT student_id FROM students WHERE gpa > 3.0\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM fall_students) AND s.student_id IN (SELECT student_id FROM good_gpa);"),
        
        ("Show Biology majors who are juniors (year 3)",
         "WITH bio_majors AS (\n  SELECT student_id FROM students WHERE major = 'Biology'\n),\njuniors AS (\n  SELECT student_id FROM students WHERE year = 3\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM bio_majors) AND s.student_id IN (SELECT student_id FROM juniors);"),
        
        ("List 4-credit Mathematics courses",
         "WITH four_credit AS (\n  SELECT course_id FROM courses WHERE credits = 4\n),\nmath_courses AS (\n  SELECT course_id FROM courses WHERE department = 'Mathematics'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM four_credit) AND c.course_id IN (SELECT course_id FROM math_courses);"),
        
        ("Find freshmen (year 1) with GPA above 3.8",
         "WITH freshmen AS (\n  SELECT student_id FROM students WHERE year = 1\n),\ntop_gpa AS (\n  SELECT student_id FROM students WHERE gpa > 3.8\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM freshmen) AND s.student_id IN (SELECT student_id FROM top_gpa);"),
        
        ("Show students who took courses in Spring 2024 and got at least one B",
         "WITH spring_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE semester = 'Spring 2024'\n),\nb_grades AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'B'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM spring_students) AND s.student_id IN (SELECT student_id FROM b_grades);"),
        
        ("List Engineering majors who are sophomores (year 2)",
         "WITH eng_majors AS (\n  SELECT student_id FROM students WHERE major = 'Engineering'\n),\nsophomores AS (\n  SELECT student_id FROM students WHERE year = 2\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM eng_majors) AND s.student_id IN (SELECT student_id FROM sophomores);"),
        
        ("Find 3-credit Physics courses",
         "WITH three_credit AS (\n  SELECT course_id FROM courses WHERE credits = 3\n),\nphysics_courses AS (\n  SELECT course_id FROM courses WHERE department = 'Physics'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM three_credit) AND c.course_id IN (SELECT course_id FROM physics_courses);"),
        
        ("Show students with perfect GPA (4.0) who are seniors",
         "WITH perfect_gpa AS (\n  SELECT student_id FROM students WHERE gpa = 4.0\n),\nseniors AS (\n  SELECT student_id FROM students WHERE year = 4\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM perfect_gpa) AND s.student_id IN (SELECT student_id FROM seniors);"),
        
        ("List Chemistry majors with GPA between 3.0 and 3.5",
         "WITH chem_majors AS (\n  SELECT student_id FROM students WHERE major = 'Chemistry'\n),\nmid_gpa AS (\n  SELECT student_id FROM students WHERE gpa BETWEEN 3.0 AND 3.5\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM chem_majors) AND s.student_id IN (SELECT student_id FROM mid_gpa);"),
        
        ("Find students enrolled in Fall 2023 who received an A grade",
         "WITH fall_enrolled AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE semester = 'Fall 2023'\n),\na_grades AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'A'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM fall_enrolled) AND s.student_id IN (SELECT student_id FROM a_grades);"),
        
        ("Show 4-credit English courses",
         "WITH four_credit AS (\n  SELECT course_id FROM courses WHERE credits = 4\n),\nenglish_courses AS (\n  SELECT course_id FROM courses WHERE department = 'English'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM four_credit) AND c.course_id IN (SELECT course_id FROM english_courses);"),
        
        ("List History majors who are juniors (year 3)",
         "WITH history_majors AS (\n  SELECT student_id FROM students WHERE major = 'History'\n),\njuniors AS (\n  SELECT student_id FROM students WHERE year = 3\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM history_majors) AND s.student_id IN (SELECT student_id FROM juniors);"),
        
        ("Find students with GPA over 3.7 enrolled in Spring 2024",
         "WITH high_gpa AS (\n  SELECT student_id FROM students WHERE gpa > 3.7\n),\nspring_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE semester = 'Spring 2024'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM high_gpa) AND s.student_id IN (SELECT student_id FROM spring_students);"),
        
        ("Show Economics majors who are freshmen",
         "WITH econ_majors AS (\n  SELECT student_id FROM students WHERE major = 'Economics'\n),\nfreshmen AS (\n  SELECT student_id FROM students WHERE year = 1\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM econ_majors) AND s.student_id IN (SELECT student_id FROM freshmen);"),
        
        ("List 2-credit Business courses",
         "WITH two_credit AS (\n  SELECT course_id FROM courses WHERE credits = 2\n),\nbusiness_courses AS (\n  SELECT course_id FROM courses WHERE department = 'Business'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM two_credit) AND c.course_id IN (SELECT course_id FROM business_courses);"),
        
        ("Find students who got B grades in Fall 2023",
         "WITH fall_enrolled AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE semester = 'Fall 2023'\n),\nb_students AS (\n  SELECT DISTINCT student_id FROM enrollments WHERE grade = 'B'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM fall_enrolled) AND s.student_id IN (SELECT student_id FROM b_students);"),
        
        ("Show Art majors with GPA above 3.2",
         "WITH art_majors AS (\n  SELECT student_id FROM students WHERE major = 'Art'\n),\ngood_gpa AS (\n  SELECT student_id FROM students WHERE gpa > 3.2\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM art_majors) AND s.student_id IN (SELECT student_id FROM good_gpa);"),
        
        ("List seniors (year 4) majoring in Physics",
         "WITH seniors AS (\n  SELECT student_id FROM students WHERE year = 4\n),\nphysics_majors AS (\n  SELECT student_id FROM students WHERE major = 'Physics'\n)\nSELECT s.* FROM students s WHERE s.student_id IN (SELECT student_id FROM seniors) AND s.student_id IN (SELECT student_id FROM physics_majors);"),
        
        ("Find 3-credit Chemistry courses",
         "WITH three_credit AS (\n  SELECT course_id FROM courses WHERE credits = 3\n),\nchem_courses AS (\n  SELECT course_id FROM courses WHERE department = 'Chemistry'\n)\nSELECT c.* FROM courses c WHERE c.course_id IN (SELECT course_id FROM three_credit) AND c.course_id IN (SELECT course_id FROM chem_courses);"),
    ],
    "hospital": [
        ("Find male patients who have appointments with Cardiology specialists",
         "WITH male_patients AS (\n  SELECT patient_id FROM patients WHERE gender = 'M'\n),\ncardio_appts AS (\n  SELECT DISTINCT a.patient_id FROM appointments a JOIN doctors d ON a.doctor_id = d.doctor_id WHERE d.specialty = 'Cardiology'\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM male_patients) AND p.patient_id IN (SELECT patient_id FROM cardio_appts);"),
        
        ("Show patients born before 1980 who received Aspirin prescriptions",
         "WITH older_patients AS (\n  SELECT patient_id FROM patients WHERE dob < '1980-01-01'\n),\naspirin_patients AS (\n  SELECT DISTINCT a.patient_id FROM appointments a JOIN prescriptions pr ON a.appt_id = pr.appt_id WHERE pr.medication = 'Aspirin'\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM older_patients) AND p.patient_id IN (SELECT patient_id FROM aspirin_patients);"),
        
        ("List doctors in Emergency department who specialize in Surgery",
         "WITH er_docs AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Emergency'\n),\nsurgeons AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Surgery'\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM er_docs) AND d.doctor_id IN (SELECT doctor_id FROM surgeons);"),
        
        ("Find female patients with appointments in January 2024",
         "WITH female_patients AS (\n  SELECT patient_id FROM patients WHERE gender = 'F'\n),\njan_appts AS (\n  SELECT DISTINCT patient_id FROM appointments WHERE date LIKE '2024-01%'\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM female_patients) AND p.patient_id IN (SELECT patient_id FROM jan_appts);"),
        
        ("Show appointments with Flu diagnosis by Pediatrics specialists",
         "WITH flu_appts AS (\n  SELECT appt_id, doctor_id FROM appointments WHERE diagnosis = 'Flu'\n),\npediatricians AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Pediatrics'\n)\nSELECT a.* FROM appointments a WHERE a.appt_id IN (SELECT appt_id FROM flu_appts) AND a.doctor_id IN (SELECT doctor_id FROM pediatricians);"),
        
        ("List patients born after 1990 who have appointments",
         "WITH young_patients AS (\n  SELECT patient_id FROM patients WHERE dob > '1990-01-01'\n),\nhas_appts AS (\n  SELECT DISTINCT patient_id FROM appointments\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM young_patients) AND p.patient_id IN (SELECT patient_id FROM has_appts);"),
        
        ("Find doctors in Cardiology department who have seen patients",
         "WITH cardio_docs AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Cardiology'\n),\nactive_docs AS (\n  SELECT DISTINCT doctor_id FROM appointments\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM cardio_docs) AND d.doctor_id IN (SELECT doctor_id FROM active_docs);"),
        
        ("Show prescriptions for Ibuprofen given in appointments from February 2024",
         "WITH ibuprofen AS (\n  SELECT rx_id, appt_id FROM prescriptions WHERE medication = 'Ibuprofen'\n),\nfeb_appts AS (\n  SELECT appt_id FROM appointments WHERE date LIKE '2024-02%'\n)\nSELECT pr.* FROM prescriptions pr WHERE pr.rx_id IN (SELECT rx_id FROM ibuprofen) AND pr.appt_id IN (SELECT appt_id FROM feb_appts);"),
        
        ("List male patients born in the 1970s",
         "WITH male_patients AS (\n  SELECT patient_id FROM patients WHERE gender = 'M'\n),\nseventies AS (\n  SELECT patient_id FROM patients WHERE dob BETWEEN '1970-01-01' AND '1979-12-31'\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM male_patients) AND p.patient_id IN (SELECT patient_id FROM seventies);"),
        
        ("Find appointments with Diabetes diagnosis in March 2024",
         "WITH diabetes AS (\n  SELECT appt_id FROM appointments WHERE diagnosis = 'Diabetes'\n),\nmar_appts AS (\n  SELECT appt_id FROM appointments WHERE date LIKE '2024-03%'\n)\nSELECT a.* FROM appointments a WHERE a.appt_id IN (SELECT appt_id FROM diabetes) AND a.appt_id IN (SELECT appt_id FROM mar_appts);"),
        
        ("Show Neurology specialists in the Neurology department",
         "WITH neuro_specialty AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Neurology'\n),\nneuro_dept AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Neurology'\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM neuro_specialty) AND d.doctor_id IN (SELECT doctor_id FROM neuro_dept);"),
        
        ("List female patients with prescriptions",
         "WITH female_patients AS (\n  SELECT patient_id FROM patients WHERE gender = 'F'\n),\nhas_prescriptions AS (\n  SELECT DISTINCT a.patient_id FROM appointments a JOIN prescriptions pr ON a.appt_id = pr.appt_id\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM female_patients) AND p.patient_id IN (SELECT patient_id FROM has_prescriptions);"),
        
        ("Find doctors in Surgery who have performed appointments",
         "WITH surgery_docs AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Surgery'\n),\nactive_docs AS (\n  SELECT DISTINCT doctor_id FROM appointments\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM surgery_docs) AND d.doctor_id IN (SELECT doctor_id FROM active_docs);"),
        
        ("Show patients born before 1970 with appointments in 2024",
         "WITH older_patients AS (\n  SELECT patient_id FROM patients WHERE dob < '1970-01-01'\n),\nrecent_appts AS (\n  SELECT DISTINCT patient_id FROM appointments WHERE date LIKE '2024%'\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM older_patients) AND p.patient_id IN (SELECT patient_id FROM recent_appts);"),
        
        ("List Orthopedics doctors in Surgery department",
         "WITH ortho AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Orthopedics'\n),\nsurgery_dept AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Surgery'\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM ortho) AND d.doctor_id IN (SELECT doctor_id FROM surgery_dept);"),
        
        ("Find appointments with Hypertension by Cardiology specialists",
         "WITH hyper_appts AS (\n  SELECT appt_id, doctor_id FROM appointments WHERE diagnosis = 'Hypertension'\n),\ncardio_docs AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Cardiology'\n)\nSELECT a.* FROM appointments a WHERE a.appt_id IN (SELECT appt_id FROM hyper_appts) AND a.doctor_id IN (SELECT doctor_id FROM cardio_docs);"),
        
        ("Show prescriptions with 500mg dosage from January 2024 appointments",
         "WITH high_dose AS (\n  SELECT rx_id, appt_id FROM prescriptions WHERE dosage = '500mg'\n),\njan_appts AS (\n  SELECT appt_id FROM appointments WHERE date LIKE '2024-01%'\n)\nSELECT pr.* FROM prescriptions pr WHERE pr.rx_id IN (SELECT rx_id FROM high_dose) AND pr.appt_id IN (SELECT appt_id FROM jan_appts);"),
        
        ("List young patients (born after 2000) with appointments",
         "WITH young AS (\n  SELECT patient_id FROM patients WHERE dob > '2000-01-01'\n),\nhas_appts AS (\n  SELECT DISTINCT patient_id FROM appointments\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM young) AND p.patient_id IN (SELECT patient_id FROM has_appts);"),
        
        ("Find Dermatology specialists in Dermatology department",
         "WITH derm_specialty AS (\n  SELECT doctor_id FROM doctors WHERE specialty = 'Dermatology'\n),\nderm_dept AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Dermatology'\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM derm_specialty) AND d.doctor_id IN (SELECT doctor_id FROM derm_dept);"),
        
        ("Show male patients born in the 1980s with appointments",
         "WITH male AS (\n  SELECT patient_id FROM patients WHERE gender = 'M'\n),\neighties_with_appts AS (\n  SELECT patient_id FROM patients WHERE dob BETWEEN '1980-01-01' AND '1989-12-31' AND patient_id IN (SELECT patient_id FROM appointments)\n)\nSELECT p.* FROM patients p WHERE p.patient_id IN (SELECT patient_id FROM male) AND p.patient_id IN (SELECT patient_id FROM eighties_with_appts);"),
        
        ("List appointments in February 2024 with prescriptions",
         "WITH feb_appts AS (\n  SELECT appt_id FROM appointments WHERE date LIKE '2024-02%'\n),\nwith_rx AS (\n  SELECT DISTINCT appt_id FROM prescriptions\n)\nSELECT a.* FROM appointments a WHERE a.appt_id IN (SELECT appt_id FROM feb_appts) AND a.appt_id IN (SELECT appt_id FROM with_rx);"),
        
        ("Find Oncology doctors who have seen patients",
         "WITH onco_docs AS (\n  SELECT doctor_id FROM doctors WHERE department = 'Oncology'\n),\nactive AS (\n  SELECT DISTINCT doctor_id FROM appointments\n)\nSELECT d.* FROM doctors d WHERE d.doctor_id IN (SELECT doctor_id FROM onco_docs) AND d.doctor_id IN (SELECT doctor_id FROM active);"),
    ],
    "hr": [
        ("Find Engineering employees with salary above 90000",
         "WITH eng_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Engineering'\n),\nhigh_salary AS (\n  SELECT emp_id FROM employees WHERE salary > 90000\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM eng_emps) AND e.emp_id IN (SELECT emp_id FROM high_salary);"),
        
        ("Show employees hired in 2023 who are assigned to projects",
         "WITH new_hires AS (\n  SELECT emp_id FROM employees WHERE hire_date LIKE '2023%'\n),\nassigned_emps AS (\n  SELECT DISTINCT emp_id FROM assignments\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM new_hires) AND e.emp_id IN (SELECT emp_id FROM assigned_emps);"),
        
        ("List departments in New York with budget over 1 million",
         "WITH ny_depts AS (\n  SELECT dept_id FROM departments WHERE location = 'New York'\n),\nlarge_budget AS (\n  SELECT dept_id FROM departments WHERE budget > 1000000\n)\nSELECT d.* FROM departments d WHERE d.dept_id IN (SELECT dept_id FROM ny_depts) AND d.dept_id IN (SELECT dept_id FROM large_budget);"),
        
        ("Find employees in Sales with assignments over 100 hours",
         "WITH sales_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Sales'\n),\nintensive_work AS (\n  SELECT DISTINCT emp_id FROM assignments WHERE hours > 100\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM sales_emps) AND e.emp_id IN (SELECT emp_id FROM intensive_work);"),
        
        ("Show projects starting in 2024 in Engineering department",
         "WITH projects_2024 AS (\n  SELECT proj_id, dept_id FROM projects WHERE start_date LIKE '2024%'\n),\neng_dept AS (\n  SELECT dept_id FROM departments WHERE name = 'Engineering'\n)\nSELECT p.* FROM projects p WHERE p.proj_id IN (SELECT proj_id FROM projects_2024) AND p.dept_id IN (SELECT dept_id FROM eng_dept);"),
        
        ("List employees hired before 2020 with salary over 100000",
         "WITH veteran AS (\n  SELECT emp_id FROM employees WHERE hire_date < '2020-01-01'\n),\nhigh_earners AS (\n  SELECT emp_id FROM employees WHERE salary > 100000\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM veteran) AND e.emp_id IN (SELECT emp_id FROM high_earners);"),
        
        ("Find Marketing employees assigned as Lead role",
         "WITH marketing AS (\n  SELECT emp_id FROM employees WHERE department = 'Marketing'\n),\nleads AS (\n  SELECT DISTINCT emp_id FROM assignments WHERE role = 'Lead'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM marketing) AND e.emp_id IN (SELECT emp_id FROM leads);"),
        
        ("Show departments in San Francisco with budget under 2 million",
         "WITH sf_depts AS (\n  SELECT dept_id FROM departments WHERE location = 'San Francisco'\n),\nmoderate_budget AS (\n  SELECT dept_id FROM departments WHERE budget < 2000000\n)\nSELECT d.* FROM departments d WHERE d.dept_id IN (SELECT dept_id FROM sf_depts) AND d.dept_id IN (SELECT dept_id FROM moderate_budget);"),
        
        ("List employees in Finance hired in 2022",
         "WITH finance_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Finance'\n),\nhires_2022 AS (\n  SELECT emp_id FROM employees WHERE hire_date LIKE '2022%'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM finance_emps) AND e.emp_id IN (SELECT emp_id FROM hires_2022);"),
        
        ("Find projects ending in 2024 with assignments over 50 hours",
         "WITH ending_2024 AS (\n  SELECT proj_id FROM projects WHERE end_date LIKE '2024%'\n),\nsubstantial_work AS (\n  SELECT DISTINCT proj_id FROM assignments WHERE hours > 50\n)\nSELECT p.* FROM projects p WHERE p.proj_id IN (SELECT proj_id FROM ending_2024) AND p.proj_id IN (SELECT proj_id FROM substantial_work);"),
        
        ("Show employees with salary between 70000 and 90000 in Engineering",
         "WITH mid_salary AS (\n  SELECT emp_id FROM employees WHERE salary BETWEEN 70000 AND 90000\n),\neng_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Engineering'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM mid_salary) AND e.emp_id IN (SELECT emp_id FROM eng_emps);"),
        
        ("List HR employees assigned to projects",
         "WITH hr_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'HR'\n),\nassigned AS (\n  SELECT DISTINCT emp_id FROM assignments\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM hr_emps) AND e.emp_id IN (SELECT emp_id FROM assigned);"),
        
        ("Find departments in Chicago with budget over 500000",
         "WITH chicago AS (\n  SELECT dept_id FROM departments WHERE location = 'Chicago'\n),\ngood_budget AS (\n  SELECT dept_id FROM departments WHERE budget > 500000\n)\nSELECT d.* FROM departments d WHERE d.dept_id IN (SELECT dept_id FROM chicago) AND d.dept_id IN (SELECT dept_id FROM good_budget);"),
        
        ("Show employees in Operations hired in 2021",
         "WITH ops_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Operations'\n),\nhires_2021 AS (\n  SELECT emp_id FROM employees WHERE hire_date LIKE '2021%'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM ops_emps) AND e.emp_id IN (SELECT emp_id FROM hires_2021);"),
        
        ("List employees with salary over 80000 assigned as Developer",
         "WITH high_salary AS (\n  SELECT emp_id FROM employees WHERE salary > 80000\n),\ndevelopers AS (\n  SELECT DISTINCT emp_id FROM assignments WHERE role = 'Developer'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM high_salary) AND e.emp_id IN (SELECT emp_id FROM developers);"),
        
        ("Find projects started in 2023 that end in 2025",
         "WITH start_2023 AS (\n  SELECT proj_id FROM projects WHERE start_date LIKE '2023%'\n),\nend_2025 AS (\n  SELECT proj_id FROM projects WHERE end_date LIKE '2025%'\n)\nSELECT p.* FROM projects p WHERE p.proj_id IN (SELECT proj_id FROM start_2023) AND p.proj_id IN (SELECT proj_id FROM end_2025);"),
        
        ("Show Sales employees with salary under 60000",
         "WITH sales AS (\n  SELECT emp_id FROM employees WHERE department = 'Sales'\n),\nlower_salary AS (\n  SELECT emp_id FROM employees WHERE salary < 60000\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM sales) AND e.emp_id IN (SELECT emp_id FROM lower_salary);"),
        
        ("List employees hired in 2023 assigned as Manager",
         "WITH new_hires AS (\n  SELECT emp_id FROM employees WHERE hire_date LIKE '2023%'\n),\nmanagers AS (\n  SELECT DISTINCT emp_id FROM assignments WHERE role = 'Manager'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM new_hires) AND e.emp_id IN (SELECT emp_id FROM managers);"),
        
        ("Find departments in New York or San Francisco with budget over 1.5 million",
         "WITH target_cities AS (\n  SELECT dept_id FROM departments WHERE location IN ('New York', 'San Francisco')\n),\nlarge_budget AS (\n  SELECT dept_id FROM departments WHERE budget > 1500000\n)\nSELECT d.* FROM departments d WHERE d.dept_id IN (SELECT dept_id FROM target_cities) AND d.dept_id IN (SELECT dept_id FROM large_budget);"),
        
        ("Show Engineering employees with assignments under 80 hours",
         "WITH eng_emps AS (\n  SELECT emp_id FROM employees WHERE department = 'Engineering'\n),\nlight_work AS (\n  SELECT DISTINCT emp_id FROM assignments WHERE hours < 80\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM eng_emps) AND e.emp_id IN (SELECT emp_id FROM light_work);"),
        
        ("List projects starting in 2024 in Sales department",
         "WITH start_2024 AS (\n  SELECT proj_id, dept_id FROM projects WHERE start_date LIKE '2024%'\n),\nsales_dept AS (\n  SELECT dept_id FROM departments WHERE name = 'Sales'\n)\nSELECT p.* FROM projects p WHERE p.proj_id IN (SELECT proj_id FROM start_2024) AND p.dept_id IN (SELECT dept_id FROM sales_dept);"),
        
        ("Find employees hired before 2021 in Marketing",
         "WITH old_hires AS (\n  SELECT emp_id FROM employees WHERE hire_date < '2021-01-01'\n),\nmarketing AS (\n  SELECT emp_id FROM employees WHERE department = 'Marketing'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM old_hires) AND e.emp_id IN (SELECT emp_id FROM marketing);"),
        
        ("Show employees with salary over 95000 hired in 2022 or 2023",
         "WITH high_salary AS (\n  SELECT emp_id FROM employees WHERE salary > 95000\n),\nrecent_hires AS (\n  SELECT emp_id FROM employees WHERE hire_date BETWEEN '2022-01-01' AND '2023-12-31'\n)\nSELECT e.* FROM employees e WHERE e.emp_id IN (SELECT emp_id FROM high_salary) AND e.emp_id IN (SELECT emp_id FROM recent_hires);"),
    ]
}

# CTE with aggregation examples (20% = 60 examples)
AGG_CTE_TEMPLATES = {
    "ecommerce": [
        ("What is the average order value by country?",
         "WITH order_totals AS (\n  SELECT c.country, o.total_amount\n  FROM orders o\n  JOIN customers c ON o.customer_id = c.customer_id\n)\nSELECT country, AVG(total_amount) as avg_order_value\nFROM order_totals\nGROUP BY country;"),
        
        ("Show the total revenue per product category",
         "WITH category_sales AS (\n  SELECT p.category, oi.price * oi.quantity as revenue\n  FROM order_items oi\n  JOIN products p ON oi.product_id = p.product_id\n)\nSELECT category, SUM(revenue) as total_revenue\nFROM category_sales\nGROUP BY category;"),
        
        ("Count how many customers each country has",
         "WITH country_counts AS (\n  SELECT country, COUNT(*) as customer_count\n  FROM customers\n  GROUP BY country\n)\nSELECT * FROM country_counts ORDER BY customer_count DESC;"),
        
        ("What is the average stock level by product category?",
         "WITH category_stock AS (\n  SELECT category, stock\n  FROM products\n)\nSELECT category, AVG(stock) as avg_stock\nFROM category_stock\nGROUP BY category;"),
        
        ("Show total number of orders by status",
         "WITH order_status_counts AS (\n  SELECT status, COUNT(*) as order_count\n  FROM orders\n  GROUP BY status\n)\nSELECT * FROM order_status_counts;"),
        
        ("Find the total quantity sold per product",
         "WITH product_quantities AS (\n  SELECT product_id, SUM(quantity) as total_sold\n  FROM order_items\n  GROUP BY product_id\n)\nSELECT p.name, pq.total_sold\nFROM product_quantities pq\nJOIN products p ON pq.product_id = p.product_id\nORDER BY pq.total_sold DESC;"),
        
        ("Calculate average price by product category",
         "WITH category_prices AS (\n  SELECT category, price\n  FROM products\n)\nSELECT category, AVG(price) as avg_price\nFROM category_prices\nGROUP BY category;"),
        
        ("Show total spending per customer",
         "WITH customer_spending AS (\n  SELECT customer_id, SUM(total_amount) as total_spent\n  FROM orders\n  GROUP BY customer_id\n)\nSELECT c.name, cs.total_spent\nFROM customer_spending cs\nJOIN customers c ON cs.customer_id = c.customer_id\nORDER BY cs.total_spent DESC;"),
        
        ("Count orders per customer",
         "WITH order_counts AS (\n  SELECT customer_id, COUNT(*) as num_orders\n  FROM orders\n  GROUP BY customer_id\n)\nSELECT c.name, oc.num_orders\nFROM order_counts oc\nJOIN customers c ON oc.customer_id = c.customer_id\nORDER BY oc.num_orders DESC;"),
        
        ("What is the maximum order value per country?",
         "WITH country_orders AS (\n  SELECT c.country, o.total_amount\n  FROM orders o\n  JOIN customers c ON o.customer_id = c.customer_id\n)\nSELECT country, MAX(total_amount) as max_order\nFROM country_orders\nGROUP BY country;"),
        
        ("Show the total number of products in each category",
         "WITH category_products AS (\n  SELECT category, COUNT(*) as product_count\n  FROM products\n  GROUP BY category\n)\nSELECT * FROM category_products ORDER BY product_count DESC;"),
        
        ("Calculate the average quantity per order",
         "WITH order_quantities AS (\n  SELECT order_id, SUM(quantity) as total_qty\n  FROM order_items\n  GROUP BY order_id\n)\nSELECT AVG(total_qty) as avg_quantity_per_order FROM order_quantities;"),
        
        ("Show total revenue by order status",
         "WITH status_revenue AS (\n  SELECT status, SUM(total_amount) as revenue\n  FROM orders\n  GROUP BY status\n)\nSELECT * FROM status_revenue ORDER BY revenue DESC;"),
        
        ("Find customers who have spent more than the average",
         "WITH customer_totals AS (\n  SELECT customer_id, SUM(total_amount) as total_spent\n  FROM orders\n  GROUP BY customer_id\n)\nSELECT c.*, ct.total_spent\nFROM customer_totals ct\nJOIN customers c ON ct.customer_id = c.customer_id\nWHERE ct.total_spent > (SELECT AVG(total_spent) FROM customer_totals);"),
        
        ("Count how many times each product has been ordered",
         "WITH product_order_counts AS (\n  SELECT product_id, COUNT(*) as times_ordered\n  FROM order_items\n  GROUP BY product_id\n)\nSELECT p.name, poc.times_ordered\nFROM product_order_counts poc\nJOIN products p ON poc.product_id = p.product_id\nORDER BY poc.times_ordered DESC;"),
    ],
    "university": [
        ("What is the average GPA by major?",
         "WITH major_gpas AS (\n  SELECT major, gpa\n  FROM students\n)\nSELECT major, AVG(gpa) as avg_gpa\nFROM major_gpas\nGROUP BY major\nORDER BY avg_gpa DESC;"),
        
        ("Count students in each year level",
         "WITH year_counts AS (\n  SELECT year, COUNT(*) as student_count\n  FROM students\n  GROUP BY year\n)\nSELECT * FROM year_counts ORDER BY year;"),
        
        ("Show the number of courses per department",
         "WITH dept_courses AS (\n  SELECT department, COUNT(*) as course_count\n  FROM courses\n  GROUP BY department\n)\nSELECT * FROM dept_courses ORDER BY course_count DESC;"),
        
        ("Find the average credits per department",
         "WITH dept_credits AS (\n  SELECT department, credits\n  FROM courses\n)\nSELECT department, AVG(credits) as avg_credits\nFROM dept_credits\nGROUP BY department;"),
        
        ("Count enrollments per semester",
         "WITH semester_counts AS (\n  SELECT semester, COUNT(*) as enrollment_count\n  FROM enrollments\n  GROUP BY semester\n)\nSELECT * FROM semester_counts ORDER BY semester;"),
        
        ("Show the number of students per major",
         "WITH major_counts AS (\n  SELECT major, COUNT(*) as student_count\n  FROM students\n  GROUP BY major\n)\nSELECT * FROM major_counts ORDER BY student_count DESC;"),
        
        ("Calculate average GPA by year level",
         "WITH year_gpas AS (\n  SELECT year, gpa\n  FROM students\n)\nSELECT year, AVG(gpa) as avg_gpa\nFROM year_gpas\nGROUP BY year\nORDER BY year;"),
        
        ("Count professors in each department",
         "WITH dept_profs AS (\n  SELECT department, COUNT(*) as prof_count\n  FROM professors\n  GROUP BY department\n)\nSELECT * FROM dept_profs ORDER BY prof_count DESC;"),
        
        ("Find students with above-average GPA",
         "WITH gpa_stats AS (\n  SELECT student_id, gpa\n  FROM students\n)\nSELECT s.*\nFROM students s\nWHERE s.gpa > (SELECT AVG(gpa) FROM gpa_stats);"),
        
        ("Show grade distribution (count of each grade)",
         "WITH grade_counts AS (\n  SELECT grade, COUNT(*) as count\n  FROM enrollments\n  GROUP BY grade\n)\nSELECT * FROM grade_counts ORDER BY grade;"),
        
        ("Calculate total credits offered per department",
         "WITH dept_total_credits AS (\n  SELECT department, SUM(credits) as total_credits\n  FROM courses\n  GROUP BY department\n)\nSELECT * FROM dept_total_credits ORDER BY total_credits DESC;"),
        
        ("Count enrollments per student",
         "WITH student_enrollments AS (\n  SELECT student_id, COUNT(*) as enrollment_count\n  FROM enrollments\n  GROUP BY student_id\n)\nSELECT s.name, se.enrollment_count\nFROM student_enrollments se\nJOIN students s ON se.student_id = s.student_id\nORDER BY se.enrollment_count DESC;"),
        
        ("Find the maximum GPA per major",
         "WITH major_max_gpa AS (\n  SELECT major, MAX(gpa) as max_gpa\n  FROM students\n  GROUP BY major\n)\nSELECT * FROM major_max_gpa ORDER BY max_gpa DESC;"),
        
        ("Show average number of enrollments per student",
         "WITH student_enrollment_counts AS (\n  SELECT student_id, COUNT(*) as enrollment_count\n  FROM enrollments\n  GROUP BY student_id\n)\nSELECT AVG(enrollment_count) as avg_enrollments FROM student_enrollment_counts;"),
        
        ("Count courses by credit value",
         "WITH credit_counts AS (\n  SELECT credits, COUNT(*) as course_count\n  FROM courses\n  GROUP BY credits\n)\nSELECT * FROM credit_counts ORDER BY credits;"),
    ],
    "hospital": [
        ("Count appointments per doctor",
         "WITH doctor_appts AS (\n  SELECT doctor_id, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY doctor_id\n)\nSELECT d.name, da.appt_count\nFROM doctor_appts da\nJOIN doctors d ON da.doctor_id = d.doctor_id\nORDER BY da.appt_count DESC;"),
        
        ("Show number of doctors per specialty",
         "WITH specialty_counts AS (\n  SELECT specialty, COUNT(*) as doctor_count\n  FROM doctors\n  GROUP BY specialty\n)\nSELECT * FROM specialty_counts ORDER BY doctor_count DESC;"),
        
        ("Count appointments per patient",
         "WITH patient_appts AS (\n  SELECT patient_id, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY patient_id\n)\nSELECT p.name, pa.appt_count\nFROM patient_appts pa\nJOIN patients p ON pa.patient_id = p.patient_id\nORDER BY pa.appt_count DESC;"),
        
        ("Find the number of doctors in each department",
         "WITH dept_doctors AS (\n  SELECT department, COUNT(*) as doctor_count\n  FROM doctors\n  GROUP BY department\n)\nSELECT * FROM dept_doctors ORDER BY doctor_count DESC;"),
        
        ("Count prescriptions per medication",
         "WITH med_counts AS (\n  SELECT medication, COUNT(*) as prescription_count\n  FROM prescriptions\n  GROUP BY medication\n)\nSELECT * FROM med_counts ORDER BY prescription_count DESC;"),
        
        ("Show number of patients by gender",
         "WITH gender_counts AS (\n  SELECT gender, COUNT(*) as patient_count\n  FROM patients\n  GROUP BY gender\n)\nSELECT * FROM gender_counts;"),
        
        ("Count appointments by diagnosis",
         "WITH diagnosis_counts AS (\n  SELECT diagnosis, COUNT(*) as count\n  FROM appointments\n  GROUP BY diagnosis\n)\nSELECT * FROM diagnosis_counts ORDER BY count DESC;"),
        
        ("Find average number of appointments per patient",
         "WITH patient_appt_counts AS (\n  SELECT patient_id, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY patient_id\n)\nSELECT AVG(appt_count) as avg_appts_per_patient FROM patient_appt_counts;"),
        
        ("Show the number of prescriptions per appointment",
         "WITH appt_rx_counts AS (\n  SELECT appt_id, COUNT(*) as rx_count\n  FROM prescriptions\n  GROUP BY appt_id\n)\nSELECT AVG(rx_count) as avg_prescriptions_per_appt FROM appt_rx_counts;"),
        
        ("Count appointments per month in 2024",
         "WITH monthly_appts AS (\n  SELECT SUBSTR(date, 1, 7) as month, COUNT(*) as appt_count\n  FROM appointments\n  WHERE date LIKE '2024%'\n  GROUP BY month\n)\nSELECT * FROM monthly_appts ORDER BY month;"),
        
        ("Find patients with more appointments than average",
         "WITH patient_appt_counts AS (\n  SELECT patient_id, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY patient_id\n)\nSELECT p.*, pac.appt_count\nFROM patient_appt_counts pac\nJOIN patients p ON pac.patient_id = p.patient_id\nWHERE pac.appt_count > (SELECT AVG(appt_count) FROM patient_appt_counts);"),
        
        ("Show doctors with more than 10 appointments",
         "WITH doctor_workload AS (\n  SELECT doctor_id, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY doctor_id\n  HAVING COUNT(*) > 10\n)\nSELECT d.*, dw.appt_count\nFROM doctor_workload dw\nJOIN doctors d ON dw.doctor_id = d.doctor_id;"),
        
        ("Count patients by age group",
         "WITH age_groups AS (\n  SELECT \n    CASE \n      WHEN dob > '2000-01-01' THEN 'Young'\n      WHEN dob > '1970-01-01' THEN 'Middle'\n      ELSE 'Senior'\n    END as age_group\n  FROM patients\n)\nSELECT age_group, COUNT(*) as patient_count\nFROM age_groups\nGROUP BY age_group;"),
        
        ("Find the most common dosage",
         "WITH dosage_counts AS (\n  SELECT dosage, COUNT(*) as count\n  FROM prescriptions\n  GROUP BY dosage\n)\nSELECT * FROM dosage_counts ORDER BY count DESC LIMIT 1;"),
        
        ("Show number of appointments per day of the week",
         "WITH day_appts AS (\n  SELECT STRFTIME('%w', date) as day_of_week, COUNT(*) as appt_count\n  FROM appointments\n  GROUP BY day_of_week\n)\nSELECT * FROM day_appts ORDER BY day_of_week;"),
    ],
    "hr": [
        ("What is the average salary by department?",
         "WITH dept_salaries AS (\n  SELECT department, salary\n  FROM employees\n)\nSELECT department, AVG(salary) as avg_salary\nFROM dept_salaries\nGROUP BY department\nORDER BY avg_salary DESC;"),
        
        ("Count employees in each department",
         "WITH dept_counts AS (\n  SELECT department, COUNT(*) as employee_count\n  FROM employees\n  GROUP BY department\n)\nSELECT * FROM dept_counts ORDER BY employee_count DESC;"),
        
        ("Show total budget by department location",
         "WITH location_budgets AS (\n  SELECT location, SUM(budget) as total_budget\n  FROM departments\n  GROUP BY location\n)\nSELECT * FROM location_budgets ORDER BY total_budget DESC;"),
        
        ("Calculate average project hours per employee",
         "WITH emp_hours AS (\n  SELECT emp_id, SUM(hours) as total_hours\n  FROM assignments\n  GROUP BY emp_id\n)\nSELECT AVG(total_hours) as avg_hours FROM emp_hours;"),
        
        ("Count projects per department",
         "WITH dept_projects AS (\n  SELECT d.name, COUNT(p.proj_id) as project_count\n  FROM departments d\n  LEFT JOIN projects p ON d.dept_id = p.dept_id\n  GROUP BY d.dept_id, d.name\n)\nSELECT * FROM dept_projects ORDER BY project_count DESC;"),
        
        ("Show total hours worked per project",
         "WITH project_hours AS (\n  SELECT proj_id, SUM(hours) as total_hours\n  FROM assignments\n  GROUP BY proj_id\n)\nSELECT p.name, ph.total_hours\nFROM project_hours ph\nJOIN projects p ON ph.proj_id = p.proj_id\nORDER BY ph.total_hours DESC;"),
        
        ("Find employees with above-average salary",
         "WITH salary_stats AS (\n  SELECT emp_id, salary\n  FROM employees\n)\nSELECT e.*\nFROM employees e\nWHERE e.salary > (SELECT AVG(salary) FROM salary_stats);"),
        
        ("Count assignments per role",
         "WITH role_counts AS (\n  SELECT role, COUNT(*) as assignment_count\n  FROM assignments\n  GROUP BY role\n)\nSELECT * FROM role_counts ORDER BY assignment_count DESC;"),
        
        ("Show maximum salary per department",
         "WITH dept_max_salary AS (\n  SELECT department, MAX(salary) as max_salary\n  FROM employees\n  GROUP BY department\n)\nSELECT * FROM dept_max_salary ORDER BY max_salary DESC;"),
        
        ("Count employees hired per year",
         "WITH hire_years AS (\n  SELECT SUBSTR(hire_date, 1, 4) as hire_year, COUNT(*) as employee_count\n  FROM employees\n  GROUP BY hire_year\n)\nSELECT * FROM hire_years ORDER BY hire_year;"),
        
        ("Find departments with budget over average",
         "WITH budget_stats AS (\n  SELECT dept_id, budget\n  FROM departments\n)\nSELECT d.*\nFROM departments d\nWHERE d.budget > (SELECT AVG(budget) FROM budget_stats);"),
        
        ("Show total salary expenditure per department",
         "WITH dept_salary_totals AS (\n  SELECT department, SUM(salary) as total_salary\n  FROM employees\n  GROUP BY department\n)\nSELECT * FROM dept_salary_totals ORDER BY total_salary DESC;"),
        
        ("Count employees per manager",
         "WITH manager_counts AS (\n  SELECT manager_id, COUNT(*) as subordinate_count\n  FROM employees\n  WHERE manager_id IS NOT NULL\n  GROUP BY manager_id\n)\nSELECT e.name, mc.subordinate_count\nFROM manager_counts mc\nJOIN employees e ON mc.manager_id = e.emp_id\nORDER BY mc.subordinate_count DESC;"),
        
        ("Find average hours per assignment role",
         "WITH role_hours AS (\n  SELECT role, AVG(hours) as avg_hours\n  FROM assignments\n  GROUP BY role\n)\nSELECT * FROM role_hours ORDER BY avg_hours DESC;"),
        
        ("Show number of departments per location",
         "WITH location_counts AS (\n  SELECT location, COUNT(*) as dept_count\n  FROM departments\n  GROUP BY location\n)\nSELECT * FROM location_counts ORDER BY dept_count DESC;"),
    ]
}

# Recursive CTE examples (10% = 30 examples)
RECURSIVE_CTE_TEMPLATES = {
    "ecommerce": [
        ("Generate a series of customer IDs from 1 to 10",
         "WITH RECURSIVE customer_series AS (\n  SELECT 1 as id\n  UNION ALL\n  SELECT id + 1 FROM customer_series WHERE id < 10\n)\nSELECT * FROM customer_series;"),
        
        ("Create a sequence of order amounts incrementing by 100 up to 1000",
         "WITH RECURSIVE amount_series AS (\n  SELECT 100 as amount\n  UNION ALL\n  SELECT amount + 100 FROM amount_series WHERE amount < 1000\n)\nSELECT * FROM amount_series;"),
        
        ("Generate dates for the first 7 days of 2024",
         "WITH RECURSIVE date_series AS (\n  SELECT DATE('2024-01-01') as date\n  UNION ALL\n  SELECT DATE(date, '+1 day') FROM date_series WHERE date < '2024-01-07'\n)\nSELECT * FROM date_series;"),
        
        ("Create a series of product IDs from 100 to 110",
         "WITH RECURSIVE product_series AS (\n  SELECT 100 as product_id\n  UNION ALL\n  SELECT product_id + 1 FROM product_series WHERE product_id < 110\n)\nSELECT * FROM product_series;"),
        
        ("Generate order IDs in increments of 5 from 1000 to 1050",
         "WITH RECURSIVE order_series AS (\n  SELECT 1000 as order_id\n  UNION ALL\n  SELECT order_id + 5 FROM order_series WHERE order_id < 1050\n)\nSELECT * FROM order_series;"),
        
        ("Create a price ladder from 10 to 100 in steps of 10",
         "WITH RECURSIVE price_ladder AS (\n  SELECT 10 as price\n  UNION ALL\n  SELECT price + 10 FROM price_ladder WHERE price < 100\n)\nSELECT * FROM price_ladder;"),
        
        ("Generate quantity levels from 1 to 20",
         "WITH RECURSIVE qty_series AS (\n  SELECT 1 as quantity\n  UNION ALL\n  SELECT quantity + 1 FROM qty_series WHERE quantity < 20\n)\nSELECT * FROM qty_series;"),
        
        ("Create months from January to December 2024",
         "WITH RECURSIVE month_series AS (\n  SELECT 1 as month\n  UNION ALL\n  SELECT month + 1 FROM month_series WHERE month < 12\n)\nSELECT month, '2024-' || PRINTF('%02d', month) || '-01' as date FROM month_series;"),
    ],
    "university": [
        ("Generate a series of student IDs from 1001 to 1010",
         "WITH RECURSIVE student_series AS (\n  SELECT 1001 as student_id\n  UNION ALL\n  SELECT student_id + 1 FROM student_series WHERE student_id < 1010\n)\nSELECT * FROM student_series;"),
        
        ("Create a GPA scale from 0.0 to 4.0 in 0.5 increments",
         "WITH RECURSIVE gpa_scale AS (\n  SELECT 0.0 as gpa\n  UNION ALL\n  SELECT gpa + 0.5 FROM gpa_scale WHERE gpa < 4.0\n)\nSELECT * FROM gpa_scale;"),
        
        ("Generate credit values from 1 to 6",
         "WITH RECURSIVE credit_series AS (\n  SELECT 1 as credits\n  UNION ALL\n  SELECT credits + 1 FROM credit_series WHERE credits < 6\n)\nSELECT * FROM credit_series;"),
        
        ("Create a series of years from 1 to 4",
         "WITH RECURSIVE year_series AS (\n  SELECT 1 as year\n  UNION ALL\n  SELECT year + 1 FROM year_series WHERE year < 4\n)\nSELECT * FROM year_series;"),
        
        ("Generate semesters from Fall 2020 to Spring 2024",
         "WITH RECURSIVE semester_series AS (\n  SELECT 2020 as year, 'Fall' as term\n  UNION ALL\n  SELECT \n    CASE WHEN term = 'Fall' THEN year ELSE year + 1 END,\n    CASE WHEN term = 'Fall' THEN 'Spring' ELSE 'Fall' END\n  FROM semester_series\n  WHERE year < 2024 OR (year = 2024 AND term = 'Fall')\n)\nSELECT term || ' ' || year as semester FROM semester_series;"),
        
        ("Create course numbers from 101 to 110",
         "WITH RECURSIVE course_nums AS (\n  SELECT 101 as course_num\n  UNION ALL\n  SELECT course_num + 1 FROM course_nums WHERE course_num < 110\n)\nSELECT * FROM course_nums;"),
        
        ("Generate professor IDs from 500 to 510",
         "WITH RECURSIVE prof_series AS (\n  SELECT 500 as prof_id\n  UNION ALL\n  SELECT prof_id + 1 FROM prof_series WHERE prof_id < 510\n)\nSELECT * FROM prof_series;"),
    ],
    "hospital": [
        ("Generate patient IDs from 2001 to 2010",
         "WITH RECURSIVE patient_series AS (\n  SELECT 2001 as patient_id\n  UNION ALL\n  SELECT patient_id + 1 FROM patient_series WHERE patient_id < 2010\n)\nSELECT * FROM patient_series;"),
        
        ("Create appointment dates for January 2024",
         "WITH RECURSIVE appt_dates AS (\n  SELECT DATE('2024-01-01') as date\n  UNION ALL\n  SELECT DATE(date, '+1 day') FROM appt_dates WHERE date < '2024-01-31'\n)\nSELECT * FROM appt_dates;"),
        
        ("Generate doctor IDs from 100 to 120",
         "WITH RECURSIVE doctor_series AS (\n  SELECT 100 as doctor_id\n  UNION ALL\n  SELECT doctor_id + 1 FROM doctor_series WHERE doctor_id < 120\n)\nSELECT * FROM doctor_series;"),
        
        ("Create dosage levels from 50mg to 500mg in 50mg increments",
         "WITH RECURSIVE dosage_series AS (\n  SELECT 50 as dosage_mg\n  UNION ALL\n  SELECT dosage_mg + 50 FROM dosage_series WHERE dosage_mg < 500\n)\nSELECT dosage_mg || 'mg' as dosage FROM dosage_series;"),
        
        ("Generate appointment IDs from 5000 to 5020",
         "WITH RECURSIVE appt_series AS (\n  SELECT 5000 as appt_id\n  UNION ALL\n  SELECT appt_id + 1 FROM appt_series WHERE appt_id < 5020\n)\nSELECT * FROM appt_series;"),
        
        ("Create birth years from 1950 to 2000 in 5-year intervals",
         "WITH RECURSIVE year_series AS (\n  SELECT 1950 as birth_year\n  UNION ALL\n  SELECT birth_year + 5 FROM year_series WHERE birth_year < 2000\n)\nSELECT * FROM year_series;"),
        
        ("Generate prescription IDs from 8000 to 8015",
         "WITH RECURSIVE rx_series AS (\n  SELECT 8000 as rx_id\n  UNION ALL\n  SELECT rx_id + 1 FROM rx_series WHERE rx_id < 8015\n)\nSELECT * FROM rx_series;"),
        
        ("Create a weekly schedule for March 2024",
         "WITH RECURSIVE week_series AS (\n  SELECT DATE('2024-03-01') as week_start\n  UNION ALL\n  SELECT DATE(week_start, '+7 days') FROM week_series WHERE week_start < '2024-03-25'\n)\nSELECT * FROM week_series;"),
    ],
    "hr": [
        ("Generate employee IDs from 1000 to 1020",
         "WITH RECURSIVE emp_series AS (\n  SELECT 1000 as emp_id\n  UNION ALL\n  SELECT emp_id + 1 FROM emp_series WHERE emp_id < 1020\n)\nSELECT * FROM emp_series;"),
        
        ("Find all employees and their subordinates starting from a manager",
         "WITH RECURSIVE org_tree AS (\n  SELECT emp_id, name, manager_id, 0 as level\n  FROM employees\n  WHERE manager_id IS NULL\n  UNION ALL\n  SELECT e.emp_id, e.name, e.manager_id, ot.level + 1\n  FROM employees e\n  JOIN org_tree ot ON e.manager_id = ot.emp_id\n)\nSELECT * FROM org_tree;"),
        
        ("Create salary bands from 40000 to 120000 in 10000 increments",
         "WITH RECURSIVE salary_bands AS (\n  SELECT 40000 as salary_min\n  UNION ALL\n  SELECT salary_min + 10000 FROM salary_bands WHERE salary_min < 120000\n)\nSELECT salary_min, salary_min + 10000 as salary_max FROM salary_bands;"),
        
        ("Generate department IDs from 1 to 15",
         "WITH RECURSIVE dept_series AS (\n  SELECT 1 as dept_id\n  UNION ALL\n  SELECT dept_id + 1 FROM dept_series WHERE dept_id < 15\n)\nSELECT * FROM dept_series;"),
        
        ("Create project IDs from 100 to 120",
         "WITH RECURSIVE proj_series AS (\n  SELECT 100 as proj_id\n  UNION ALL\n  SELECT proj_id + 1 FROM proj_series WHERE proj_id < 120\n)\nSELECT * FROM proj_series;"),
        
        ("Generate hire dates for each month of 2023",
         "WITH RECURSIVE month_series AS (\n  SELECT DATE('2023-01-01') as hire_month\n  UNION ALL\n  SELECT DATE(hire_month, '+1 month') FROM month_series WHERE hire_month < '2023-12-01'\n)\nSELECT * FROM month_series;"),
        
        ("Find reporting chain for a specific employee",
         "WITH RECURSIVE reporting_chain AS (\n  SELECT emp_id, name, manager_id, 0 as level\n  FROM employees\n  WHERE emp_id = 1005\n  UNION ALL\n  SELECT e.emp_id, e.name, e.manager_id, rc.level + 1\n  FROM employees e\n  JOIN reporting_chain rc ON e.emp_id = rc.manager_id\n)\nSELECT * FROM reporting_chain;"),
    ]
}

def create_example(schema_name, schema_ddl, question, sql):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{question}\n\nSchema:\n{schema_ddl}"
            },
            {
                "role": "assistant",
                "content": sql
            }
        ]
    }

def get_schema(schema_name):
    schemas = {
        "ecommerce": ECOMMERCE_SCHEMA,
        "university": UNIVERSITY_SCHEMA,
        "hospital": HOSPITAL_SCHEMA,
        "hr": HR_SCHEMA
    }
    return schemas[schema_name]

def main():
    examples = []
    
    # Generate simple CTE examples (120 total)
    for schema_name, templates in SIMPLE_CTE_TEMPLATES.items():
        schema_ddl = get_schema(schema_name)
        for question, sql in templates:
            examples.append(create_example(schema_name, schema_ddl, question, sql))
    
    # Generate multiple CTE examples (90 total)
    for schema_name, templates in MULTIPLE_CTE_TEMPLATES.items():
        schema_ddl = get_schema(schema_name)
        for question, sql in templates:
            examples.append(create_example(schema_name, schema_ddl, question, sql))
    
    # Generate aggregation CTE examples (60 total)
    for schema_name, templates in AGG_CTE_TEMPLATES.items():
        schema_ddl = get_schema(schema_name)
        for question, sql in templates:
            examples.append(create_example(schema_name, schema_ddl, question, sql))
    
    # Generate recursive CTE examples (30 total)
    for schema_name, templates in RECURSIVE_CTE_TEMPLATES.items():
        schema_ddl = get_schema(schema_name)
        for question, sql in templates:
            examples.append(create_example(schema_name, schema_ddl, question, sql))
    
    # Shuffle to mix patterns
    random.shuffle(examples)
    
    # Write to JSONL file
    output_path = "/Users/arnav/programming/lm/data/training/t9/augment_cte.jsonl"
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(examples)} examples")
    print(f"Output: {output_path}")
    
    # Print statistics
    print(f"\nBreakdown:")
    print(f"- Simple CTE: {sum(1 for k in SIMPLE_CTE_TEMPLATES for _ in SIMPLE_CTE_TEMPLATES[k])}")
    print(f"- Multiple CTEs: {sum(1 for k in MULTIPLE_CTE_TEMPLATES for _ in MULTIPLE_CTE_TEMPLATES[k])}")
    print(f"- CTE with aggregation: {sum(1 for k in AGG_CTE_TEMPLATES for _ in AGG_CTE_TEMPLATES[k])}")
    print(f"- Recursive CTE: {sum(1 for k in RECURSIVE_CTE_TEMPLATES for _ in RECURSIVE_CTE_TEMPLATES[k])}")

if __name__ == "__main__":
    main()
