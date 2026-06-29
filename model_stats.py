from models import TeacherCGRU, EdgeEEGNet

def model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    return params, size

teacher = TeacherCGRU()
student = EdgeEEGNet()

teacher_params, teacher_size = model_stats(teacher)
student_params, student_size = model_stats(student)

print(f"Teacher : {teacher_params:,} params | {teacher_size:.4f} MB")
print(f"Student : {student_params:,} params | {student_size:.4f} MB")
print(f"Compression Ratio : {teacher_params/student_params:.2f}x")
