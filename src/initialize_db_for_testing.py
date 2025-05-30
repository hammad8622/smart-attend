from pymongo import MongoClient


DB_name = 'smartAttend'
client = MongoClient('localhost', 27017)
client.drop_database(DB_name)
DB = client[DB_name]


teacher_ID_1, teacher_ID_2 = 'T432', 'T748'
class_ID_1, class_ID_2, class_ID_3 = 'C321', 'C740', 'C904'
student_ID_1, student_ID_2, student_ID_3, student_ID_4 = 'S432', 'S231', 'S650', 'S172'

used_IDs_collection = DB['usedIDs']

used_IDs_entry = {
    'teacherIDs': ['T432', 'T748'],
    'studentIDs': ['S432', 'S231', 'S650', 'S172'],
    'classIDs': ['C321', 'C740', 'C904']
}

used_IDs_collection.insert_one(used_IDs_entry)

teacher_entry_1 = {
    'teacherID': teacher_ID_1,
    'userName': 'teacher123',
    'passWord': 'password123'
}

teacher_entry_2 = {
    'teacherID': teacher_ID_2,
    'userName': 'teacher321',
    'passWord': 'password321'
}

teacher_collection = DB['teacher']
teacher_collection.insert_one(teacher_entry_1) 
teacher_collection.insert_one(teacher_entry_2)

class_entry_1 = {
    'classID': class_ID_1,
    'className': 'RLN53',
    'teacherID': teacher_ID_1,
    'students': []
}

class_entry_2 = {
    'classID': class_ID_2,
    'className': 'JFK1019',
    'teacherID': teacher_ID_1,
    'students': []
}

class_entry_3 = {
    'classID': class_ID_3,
    'className': 'DSA969',
    'teacherID': teacher_ID_2,
    'students': []
}

class_collection = DB['class']
class_collection.insert_one(class_entry_1) 
class_collection.insert_one(class_entry_2) 
class_collection.insert_one(class_entry_3)

student_entry_1 = {
    'studentID': student_ID_1,
    'name': 'Hania',
    'age': 21,
    'gender': 'F',
    'batch': '2022',
    'major': 'Accounting'
}

student_entry_2 = {
    'studentID': student_ID_2,
    'name': 'Ali',
    'age': 20,
    'gender': 'M',
    'batch': '2021',
    'major': 'Accounting'
}

student_entry_3 = {
    'studentID': student_ID_3,
    'name': 'Ahad',
    'age': 21,
    'gender': 'M',
    'batch': '2022',
    'major': 'Accounting'
}

student_entry_4 = {
    'studentID': student_ID_4,
    'name': 'Amir',
    'age': 19,
    'gender': 'M',
    'batch': '2023',
    'major': 'A.I'
}

class_collection.update_one({'classID': class_ID_1}, {'$push': {'students': student_entry_1}})
class_collection.update_one({'classID': class_ID_1}, {'$push': {'students': student_entry_2}})
class_collection.update_one({'classID': class_ID_1}, {'$push': {'students': student_entry_3}})
class_collection.update_one({'classID': class_ID_2}, {'$push': {'students': student_entry_4}})
