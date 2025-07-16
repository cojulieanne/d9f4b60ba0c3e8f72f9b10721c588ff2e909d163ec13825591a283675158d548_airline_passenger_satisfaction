import hashlib
student_number = "2024-10104"
salt = "MLOPS2025B"
hashed = hashlib.sha256((student_number + salt).encode()).hexdigest()
print(hashed)