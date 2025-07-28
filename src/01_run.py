import numpy as np
from scipy.optimize import minimize
from scipy.stats import t


def preprocess_data(data_dict: dict) -> dict:
	"""
	원본 데이터를 LMM 분석에 필요한 형태로 전처리한다.

	Args:
		data_dict (dict): 'student_id', 'time', 'score' key를 포함한 데이터 딕셔너리.

	Returns:
		dict: y, time, student_indices, n_students, n_obs 등 전처리된 데이터.
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


def build_design_matrices(time: np.ndarray, student_indices: np.ndarray, n_obs: int, n_students: int) -> tuple[
	np.ndarray, np.ndarray]:
	"""
	고정 효과(X) 및 임의 효과(Z) 디자인 행렬을 생성한다.
	Model: score ~ time + (time | student_id)

	Args:
		time (np.ndarray): 시간 변수 배열.
		student_indices (np.ndarray): 학생 ID 인덱스 배열.
		n_obs (int): 전체 관측치 수.
		n_students (int): 전체 학생 수.

	Returns:
		tuple[np.ndarray, np.ndarray]: 고정 효과 행렬 X와 임의 효과 행렬 Z.
	"""
	# 고정 효과(Fixed Effect) 디자인 행렬 X: Intercept, time
	X = np.vstack([np.ones(n_obs), time]).T

	# 임의 효과(Random Effect) 디자인 행렬 Z: Random Intercept, Random Slope
	Z = np.zeros((n_obs, n_students * 2))
	for i in range(n_obs):
		student_idx = student_indices[i]
		Z[i, student_idx * 2] = 1  # Random Intercept
		Z[i, student_idx * 2 + 1] = time[i]  # Random Slope

	return X, Z


def neg_reml_log_likelihood(params: list, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
	"""
	음의 REML(Restricted Maximum Likelihood) 로그 가능도를 계산한다.
	scipy.optimize.minimize의 목적 함수(objective function)로 사용된다.

	Args:
		params (list): 최적화할 파라미터 [log(sigma_e), log(sigma_u0), log(sigma_u1)].
		y (np.ndarray): 종속 변수 벡터.
		X (np.ndarray): 고정 효과 디자인 행렬.
		Z (np.ndarray): 임의 효과 디자인 행렬.

	Returns:
		float: 계산된 음의 REML 로그 가능도 값.
	"""
	n_obs = len(y)
	n_students = Z.shape[1] // 2

	# 파라미터로부터 분산 성분(variance components) 복원
	sigma2_e = np.exp(params[0]) ** 2
	sigma2_u0 = np.exp(params[1]) ** 2
	sigma2_u1 = np.exp(params[2]) ** 2

	# 분산-공분산 행렬 구성
	R = np.eye(n_obs) * sigma2_e
	G0 = np.diag([sigma2_u0, sigma2_u1])
	G = np.kron(np.eye(n_students, dtype=float), G0)
	V = Z @ G @ Z.T + R

	try:
		V_inv = np.linalg.inv(V)
		Xt_Vinv_X = X.T @ V_inv @ X
		Xt_Vinv_X_inv = np.linalg.inv(Xt_Vinv_X)
	except np.linalg.LinAlgError:
		return np.inf

	# REML 로그 가능도 계산
	beta = Xt_Vinv_X_inv @ (X.T @ V_inv @ y)
	residuals = y - X @ beta

	log_det_V = np.linalg.slogdet(V)[1]
	log_det_Xt_Vinv_X = np.linalg.slogdet(Xt_Vinv_X)[1]

	reml_ll = -0.5 * (log_det_V + log_det_Xt_Vinv_X + residuals.T @ V_inv @ residuals)

	return -reml_ll


def fit_lmm_reml(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> 'scipy.optimize.OptimizeResult':
	"""
	최적화 알고리즘을 실행하여 REML 기반 LMM 모델을 피팅한다.

	Args:
		y (np.ndarray): 종속 변수 벡터.
		X (np.ndarray): 고정 효과 디자인 행렬.
		Z (np.ndarray): 임의 효과 디자인 행렬.

	Returns:
		scipy.optimize.OptimizeResult: 최적화 결과 객체.
	"""
	initial_params = np.log([np.std(y) * 0.5, np.std(y), 1.0])

	optimization_result = minimize(
		fun=neg_reml_log_likelihood,
		x0=initial_params,
		args=(y, X, Z),
		method='L-BFGS-B',
	)
	return optimization_result


def calculate_model_statistics(opt_result: 'scipy.optimize.OptimizeResult', y: np.ndarray, X: np.ndarray,
                               Z: np.ndarray) -> dict:
	"""
	최적화된 결과로부터 모델의 주요 통계량(고정효과, 분산 등)을 계산한다.

	Args:
		opt_result (scipy.optimize.OptimizeResult): minimize 함수의 결과.
		y (np.ndarray): 종속 변수 벡터.
		X (np.ndarray): 고정 효과 디자인 행렬.
		Z (np.ndarray): 임의 효과 디자인 행렬.

	Returns:
		dict: 계산된 모든 통계량을 포함하는 딕셔너리.
	"""
	n_obs = len(y)
	n_students = Z.shape[1] // 2

	# 최적화된 분산 성분 추출
	log_stdevs = opt_result.x
	var_e = np.exp(log_stdevs[0]) ** 2
	var_u0 = np.exp(log_stdevs[1]) ** 2
	var_u1 = np.exp(log_stdevs[2]) ** 2

	# 최종 분산-공분산 행렬 추정
	R_est = np.eye(n_obs) * var_e
	G0_est = np.diag([var_u0, var_u1])
	G_est = np.kron(np.eye(n_students, dtype=float), G0_est)
	V_est = Z @ G_est @ Z.T + R_est
	V_est_inv = np.linalg.inv(V_est)

	# 최종 고정 효과 및 관련 통계량 계산
	Xt_Vinv_X_est = X.T @ V_est_inv @ X
	Xt_Vinv_X_est_inv = np.linalg.inv(Xt_Vinv_X_est)
	beta_est = Xt_Vinv_X_est_inv @ X.T @ V_est_inv @ y
	std_errors = np.sqrt(np.diag(Xt_Vinv_X_est_inv))

	# p-value 계산 (근사치)
	df = n_obs - X.shape[1]
	t_values = beta_est / std_errors
	p_values = [2 * (1 - t.cdf(np.abs(t_val), df=df)) for t_val in t_values]

	return {
		'beta_est': beta_est,
		'std_errors': std_errors,
		't_values': t_values,
		'p_values': p_values,
		'variances': {
			'residual': var_e,
			'intercept': var_u0,
			'time': var_u1
		}
	}


def print_summary(results: dict):
	"""
	계산된 모델 통계량을 지정된 형식으로 출력한다.

	Args:
		results (dict): calculate_model_statistics에서 반환된 결과 딕셔너리.
	"""
	print("### Pure Python을 이용한 Frequentist LMM 결과 ###\n")
	print("-" * 60)
	print("결과물:")

	print("\n  - Fixed Effects (고정 효과):")
	print(f"    - Intercept (시간=0일 때 평균 점수): {results['beta_est'][0]:.2f} 점")
	print(f"    - time (시간이 1개월 지날 때마다 평균 점수 변화량): {results['beta_est'][1]:.2f} 점")

	print("\n  - Random Effects (임의 효과의 분산):")
	print(f"    - student_id (Intercept)의 분산: {results['variances']['intercept']:.2f} (학생들의 시작 점수 분산)")
	print(f"    - student_id (time)의 분산: {results['variances']['time']:.2f} (시간 효과의 학생 간 분산)")
	print(f"    - Residual의 분산: {results['variances']['residual']:.2f} (모델로 설명되지 않는 오차의 분산)")

	print("\n  - 'time' 효과에 대한 통계적 유의성:")
	print(f"    - 표준 오차 (Std. Error): {results['std_errors'][1]:.2f}")
	print(f"    - t-value: {results['t_values'][1]:.2f}")
	print(f"    - p-value (근사치): {results['p_values'][1]:.4f}")
	print("-" * 60)

	p_value_time = results['p_values'][1]
	if p_value_time < 0.05:
		print(f"\n[결론] p-value가 0.05보다 작으므로(p={p_value_time:.4f}), '시간에 따른 점수 변화는 통계적으로 유의미하다'고 해석할 수 있습니다.")
	else:
		print(f"\n[결론] p-value가 0.05보다 크므로(p={p_value_time:.4f}), '시간에 따른 점수 변화가 통계적으로 유의미하다는 강력한 증거를 찾지 못했습니다.")


if __name__ == '__main__':
	# --------------------------------------------------------------------------
	# 0. 가상 데이터 (Pseudo-Data)
	# --------------------------------------------------------------------------
	data = {
		'student_id': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
		'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
		'score': [75, 78, 83, 80, 84, 85, 70, 72, 73]
	}

	# --------------------------------------------------------------------------
	# 실행 파이프라인 (Execution Pipeline)
	# --------------------------------------------------------------------------
	# 1. 데이터 전처리
	proc_data = preprocess_data(data)
	y = proc_data['y']
	n_obs = proc_data['n_obs']
	n_students = proc_data['n_students']

	# 2. 디자인 행렬 생성
	X, Z = build_design_matrices(proc_data['time'], proc_data['student_indices'], n_obs, n_students)

	# 3. 모델 피팅 (REML 최적화)
	optimization_result = fit_lmm_reml(y, X, Z)

	# 4. 최종 통계량 계산
	if optimization_result.success:
		final_results = calculate_model_statistics(optimization_result, y, X, Z)

		# 5. 결과 출력
		print_summary(final_results)
	else:
		print("모델 최적화에 실패했습니다.")
		print(optimization_result.message)