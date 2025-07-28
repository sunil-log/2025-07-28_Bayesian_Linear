# src/analysis/data_loader.py
import numpy as np

def get_raw_data() -> dict:
	"""분석에 사용할 원본 데이터를 반환한다."""
	return {
		'student_id': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
		'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
		'score': [75, 78, 83, 80, 84, 85, 70, 72, 73]
	}

def preprocess_data(data_dict: dict) -> dict:
	"""
	원본 데이터를 LMM 분석에 필요한 형태로 전처리한다.
	"""
	y = np.array(data_dict['score'], dtype=float)
	time = np.array(data_dict['time'], dtype=float)

	student_names = sorted(list(set(data_dict['student_id'])))
	student_map = {name: i for i, name in enumerate(student_names)}
	student_indices = np.array([student_map[sid] for sid in data_dict['student_id']])

	return {
		'y': y,
		'time': time,
		'student_indices': student_indices,
		'n_students': len(student_names),
		'n_obs': len(y),
		'student_names': student_names
	}