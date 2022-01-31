import argparse
from ast import While

def parse():
    parser = argparse.ArgumentParser(description='Examination script')

    parser.add_argument("--student", type=str, help='Path to student.txt')
    parser.add_argument("--teacher", type=str, help='Path to teacher.txt')
    parser.add_argument("--start_time", type=int, default=0, help='Minimum evaluation time')
    parser.add_argument("--end_time", type=int, default=-1, help='Maximum evaluation time')

    return parser.parse_args()

def parse_file(path_file):
    result = []

    with open(path_file, 'r') as f:
        for line in f.readlines():
            data = line.split('\n')[0].split(' ')
            result.append([int(data[0]), int(data[1]), int(data[2])])
            assert result[-1][0] <= result[-1][1], result[-1]
            if len(result) > 1:
                assert result[-1][0] == result[-2][1], (result[-1], result[-2])
    return result

def cut_interval(interval, start_time, end_time):

    interval = [i for i in interval if i[1] > start_time]
    interval = [i for i in interval if (i[0] < end_time or end_time < 0)]

    if interval[0][0] <= start_time:
        interval[0][0] = start_time

    if interval[-1][1] >= end_time and end_time > 0:
        interval[-1][1] = end_time

    return interval

def get_interval(student, teacher, start_time, end_time):

    student = cut_interval(student, start_time, end_time)
    teacher = cut_interval(teacher, start_time, end_time)

    interval = []

    i = 0
    j = 0
    while(True):
        if (student[i][1] < teacher[j][1]):
            interval.append({'start': interval[-1]['end'] if len(interval) else start_time, 'end': student[i][1], 'student': student[i][2], 'teacher': teacher[j][2]})
            i += 1
        else:
            interval.append({'start': interval[-1]['end'] if len(interval) else start_time, 'end': teacher[j][1], 'student': student[i][2], 'teacher': teacher[j][2]})
            j += 1

        if i == len(student) or j == len(teacher):
            break

    return interval

def iou(student, teacher, start_time, end_time):
    interval = get_interval(student, teacher, start_time, end_time)

    result = 0.
    for el in interval:
        result += el['end'] - el['start'] if el['teacher'] == el['student'] else 0

    return result / (interval[-1]['end'] - interval[0]['start'])

if __name__ == '__main__':
    args = parse()

    student_data = parse_file(args.student)
    teacher_data = parse_file(args.teacher)
    start_time, end_time = args.start_time, args.end_time

    assert start_time >= 0, start_time
    assert start_time < end_time or end_time < 0, (start_time, end_time)
    assert student_data[-1][1] == teacher_data[-1][1], (student_data[-1][1], teacher_data[-1][1])

    result = iou(student_data, teacher_data, start_time, end_time)

    print(result)
