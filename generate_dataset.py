"""
Dataset Generator for Student Performance Prediction Project.
Uses Python's standard library to generate realistic academic data.
"""

import csv
import os
import random

def generate_student_dataset(num_records: int = 1000, output_path: str = "data/student_data.csv", seed: int = 42):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = [
        "student_id",
        "study_hours_per_day",
        "attendance_percentage",
        "previous_marks",
        "sleep_hours",
        "practice_tests_taken",
        "final_marks",
        "pass_fail"
    ]
    
    rows = []
    for i in range(1, num_records + 1):
        student_id = f"STU{i:04d}"
        
        # Realistic features
        study_hours = round(random.uniform(1.0, 10.0), 2)
        attendance = round(random.uniform(45.0, 100.0), 1)
        prev_marks = round(random.gauss(65.0, 15.0), 1)
        prev_marks = max(20.0, min(100.0, prev_marks))
        
        sleep_hours = round(random.gauss(7.0, 1.2), 1)
        sleep_hours = max(4.0, min(10.0, sleep_hours))
        
        practice_tests = random.randint(0, 12)
        
        # Realistic weighted score model with domain-specific correlation
        # Base mark influenced by past marks (35%), attendance (25%), study hours (25%), practice tests (10%), sleep (5%)
        # Plus realistic random variance (noise)
        noise = random.gauss(0, 4.0)
        
        computed_score = (
            0.35 * prev_marks +
            0.25 * attendance +
            2.80 * study_hours +
            1.20 * practice_tests +
            0.80 * (sleep_hours - 5.0) +
            noise
        )
        
        # Normalize and clip to 0 - 100 range
        final_marks = round(max(0.0, min(100.0, computed_score)), 1)
        
        # Pass criterion: 40% or higher
        pass_fail = 1 if final_marks >= 40.0 else 0
        
        rows.append([
            student_id,
            study_hours,
            attendance,
            prev_marks,
            sleep_hours,
            practice_tests,
            final_marks,
            pass_fail
        ])
        
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"Successfully generated {num_records} student records at: {output_path}")

if __name__ == "__main__":
    generate_student_dataset()

