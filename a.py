import csv

# Course reviews data with expanded columns
course_data = [
    {
        'course_name': 'Introduction to Psychology',
        'professor': 'Dr. Sarah Johnson',
        'semester': 'Fall 2024',
        'rating': 5,
        'difficulty': 3,
        'workload_hours': 8,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'A',
        'review': 'Excellent course! Professor made complex concepts easy to understand. Loved the real-world applications.'
    },
    {
        'course_name': 'Calculus I',
        'professor': 'Prof. Michael Chen',
        'semester': 'Spring 2024',
        'rating': 3,
        'difficulty': 5,
        'workload_hours': 12,
        'would_take_again': 'Maybe',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'B',
        'review': 'Challenging material but the professor moved too fast. Office hours were helpful though.'
    },
    {
        'course_name': 'World History',
        'professor': 'Dr. Amanda Rodriguez',
        'semester': 'Fall 2024',
        'rating': 4,
        'difficulty': 3,
        'workload_hours': 10,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'A-',
        'review': 'Very interesting lectures with engaging discussions. Lots of reading but worthwhile.'
    },
    {
        'course_name': 'Computer Science 101',
        'professor': 'Prof. James Lee',
        'semester': 'Fall 2024',
        'rating': 5,
        'difficulty': 4,
        'workload_hours': 15,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'No',
        'grade_received': 'A',
        'review': 'Perfect introduction to programming. Clear explanations and hands-on projects made learning fun.'
    },
    {
        'course_name': 'Organic Chemistry',
        'professor': 'Dr. Robert Williams',
        'semester': 'Spring 2024',
        'rating': 2,
        'difficulty': 5,
        'workload_hours': 18,
        'would_take_again': 'No',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'C+',
        'review': 'Extremely difficult. Labs were disorganized and lectures were hard to follow. Need better support.'
    },
    {
        'course_name': 'Creative Writing',
        'professor': 'Prof. Emily Thompson',
        'semester': 'Fall 2024',
        'rating': 5,
        'difficulty': 2,
        'workload_hours': 6,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'No',
        'grade_received': 'A',
        'review': 'Amazing experience! Professor provided great feedback and the workshop format was very collaborative.'
    },
    {
        'course_name': 'Microeconomics',
        'professor': 'Dr. David Martinez',
        'semester': 'Spring 2024',
        'rating': 4,
        'difficulty': 4,
        'workload_hours': 9,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'B+',
        'review': 'Solid course with practical examples. Tests were fair but homework was time-consuming.'
    },
    {
        'course_name': 'Public Speaking',
        'professor': 'Prof. Lisa Anderson',
        'semester': 'Fall 2024',
        'rating': 4,
        'difficulty': 2,
        'workload_hours': 5,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'No',
        'grade_received': 'A-',
        'review': 'Really helped build confidence. Professor was encouraging and provided constructive criticism.'
    },
    {
        'course_name': 'Biology Lab',
        'professor': 'Dr. Kevin Park',
        'semester': 'Spring 2024',
        'rating': 3,
        'difficulty': 3,
        'workload_hours': 7,
        'would_take_again': 'Maybe',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'B',
        'review': 'Lab experiments were interesting but the manual was confusing. TAs were helpful when available.'
    },
    {
        'course_name': 'Philosophy 101',
        'professor': 'Prof. Thomas Wright',
        'semester': 'Fall 2024',
        'rating': 5,
        'difficulty': 3,
        'workload_hours': 8,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'A',
        'review': 'Mind-blowing content! Made me think differently about life. Discussions were always thought-provoking.'
    },
    {
        'course_name': 'Statistics',
        'professor': 'Dr. Rachel Kim',
        'semester': 'Spring 2024',
        'rating': 3,
        'difficulty': 4,
        'workload_hours': 11,
        'would_take_again': 'Maybe',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'B-',
        'review': 'Useful material but dry presentation. Would benefit from more real-world examples.'
    },
    {
        'course_name': 'English Literature',
        'professor': 'Prof. Margaret Davis',
        'semester': 'Fall 2024',
        'rating': 4,
        'difficulty': 3,
        'workload_hours': 12,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'A-',
        'review': 'Great selection of books and deep analysis. Essay requirements were intensive but improved my writing.'
    },
    {
        'course_name': 'Physics II',
        'professor': 'Dr. John Miller',
        'semester': 'Spring 2024',
        'rating': 2,
        'difficulty': 5,
        'workload_hours': 16,
        'would_take_again': 'No',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'C',
        'review': 'Too theoretical without enough practical application. Professor seemed unapproachable during office hours.'
    },
    {
        'course_name': 'Digital Marketing',
        'professor': 'Prof. Jennifer Brown',
        'semester': 'Fall 2024',
        'rating': 5,
        'difficulty': 2,
        'workload_hours': 6,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'No',
        'grade_received': 'A',
        'review': 'Very relevant to today\'s job market. Professor has industry experience which added great value.'
    },
    {
        'course_name': 'Art History',
        'professor': 'Dr. Catherine Moore',
        'semester': 'Spring 2024',
        'rating': 4,
        'difficulty': 2,
        'workload_hours': 7,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'A',
        'review': 'Beautiful imagery and fascinating stories behind famous works. Field trip to the museum was a highlight.'
    },
    {
        'course_name': 'Data Structures',
        'professor': 'Prof. Andrew Zhang',
        'semester': 'Fall 2024',
        'rating': 4,
        'difficulty': 5,
        'workload_hours': 14,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'No',
        'grade_received': 'B+',
        'review': 'Challenging but rewarding. Programming assignments were tough but taught me a lot about problem-solving.'
    },
    {
        'course_name': 'Spanish 201',
        'professor': 'Prof. Maria Garcia',
        'semester': 'Spring 2024',
        'rating': 5,
        'difficulty': 3,
        'workload_hours': 8,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'A',
        'review': 'Immersive and fun! Professor only spoke Spanish in class which really helped. Cultural activities were great.'
    },
    {
        'course_name': 'Business Ethics',
        'professor': 'Dr. Steven Taylor',
        'semester': 'Fall 2024',
        'rating': 3,
        'difficulty': 2,
        'workload_hours': 5,
        'would_take_again': 'Maybe',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'B+',
        'review': 'Interesting case studies but discussions sometimes felt repetitive. Good for fulfilling requirements.'
    },
    {
        'course_name': 'Environmental Science',
        'professor': 'Dr. Patricia Green',
        'semester': 'Spring 2024',
        'rating': 5,
        'difficulty': 3,
        'workload_hours': 9,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'No',
        'textbook_required': 'Yes',
        'grade_received': 'A-',
        'review': 'Eye-opening course about climate change and sustainability. Field work was hands-on and meaningful.'
    },
    {
        'course_name': 'Music Theory',
        'professor': 'Prof. Daniel White',
        'semester': 'Fall 2024',
        'rating': 4,
        'difficulty': 4,
        'workload_hours': 10,
        'would_take_again': 'Yes',
        'attendance_mandatory': 'Yes',
        'textbook_required': 'Yes',
        'grade_received': 'B',
        'review': 'Complex material but professor broke it down well. Ear training exercises were particularly helpful.'
    }
]

# Write to CSV file
output_filename = 'college_course_reviews.csv'

with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    # Define the field names (column headers)
    fieldnames = ['course_name', 'professor', 'semester', 'rating', 'difficulty', 
                  'workload_hours', 'would_take_again', 'attendance_mandatory', 
                  'textbook_required', 'grade_received', 'review']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write all the course data
    writer.writerows(course_data)

print(f"CSV file '{output_filename}' has been created successfully!")
print(f"Total courses: {len(course_data)}")
print(f"\nColumns included:")
for i, field in enumerate(fieldnames, 1):
    print(f"  {i}. {field}")