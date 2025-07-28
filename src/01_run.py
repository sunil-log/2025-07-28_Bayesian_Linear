import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import t


def visualize_raw_data(proc_data: dict):
	"""
	1. 데이터 탐색: 학생별 점수 변화 추이 시각화
	"""
	# FIX: Convert student_names list to numpy array for advanced indexing
	student_names_array = np.array(proc_data['student_names'])
	df = pd.DataFrame({
		'student_id': student_names_array[proc_data['student_indices']],
		'time': proc_data['time'],
		'score': proc_data['y']
	})

	plt.figure(figsize=(10, 6))
	sns.lineplot(data=df, x='time', y='score', hue='student_id', marker='o')
	plt.title('1. Raw Data: Student Score Trajectories')
	plt.xlabel('Time (months)')
	plt.ylabel('Score')
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend(title='Student ID')
	plt.savefig('1_raw_data_trajectories.png')
	plt.close()
	print("[INFO] Saved '1_raw_data_trajectories.png'")


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


def visualize_design_matrices(X: np.ndarray, Z: np.ndarray):
	"""
	2. 디자인 행렬 구조 시각화
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	sns.heatmap(X, ax=ax1, cmap='viridis', annot=True, fmt='.0f', cbar=False)
	ax1.set_title('2a. Fixed Effects Design Matrix (X)')
	ax1.set_xlabel('Fixed Effects (Intercept, Time)')
	ax1.set_ylabel('Observations')

	sns.heatmap(Z, ax=ax2, cmap='YlGnBu', cbar=True)
	ax2.set_title('2b. Random Effects Design Matrix (Z)')
	ax2.set_xlabel('Random Effects (Intercepts & Slopes per Student)')
	ax2.set_ylabel('Observations')

	plt.tight_layout()
	plt.savefig('2_design_matrices_heatmap.png')
	plt.close()
	print("[INFO] Saved '2_design_matrices_heatmap.png'")


def build_design_matrices(time: np.ndarray, student_indices: np.ndarray, n_obs: int, n_students: int) -> tuple[
	np.ndarray, np.ndarray]:
	"""
	고정 효과(X) 및 임의 효과(Z) 디자인 행렬을 생성한다.
	"""
	X = np.vstack([np.ones(n_obs), time]).T
	Z = np.zeros((n_obs, n_students * 2))
	for i in range(n_obs):
		student_idx = student_indices[i]
		Z[i, student_idx * 2] = 1
		Z[i, student_idx * 2 + 1] = time[i]
	return X, Z


def neg_reml_log_likelihood(params: list, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
	"""
	음의 REML(Restricted Maximum Likelihood) 로그 가능도를 계산한다.
	"""
	n_obs = len(y)
	n_students = Z.shape[1] // 2
	sigma2_e = np.exp(params[0]) ** 2
	sigma2_u0 = np.exp(params[1]) ** 2
	sigma2_u1 = np.exp(params[2]) ** 2

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

	beta = Xt_Vinv_X_inv @ (X.T @ V_inv @ y)
	residuals = y - X @ beta
	log_det_V = np.linalg.slogdet(V)[1]
	log_det_Xt_Vinv_X = np.linalg.slogdet(Xt_Vinv_X)[1]
	reml_ll = -0.5 * (log_det_V + log_det_Xt_Vinv_X + residuals.T @ V_inv @ residuals)
	return -reml_ll


def visualize_convergence(convergence_history: list):
	"""
	3. 최적화 과정 시각화
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(convergence_history, marker='.', linestyle='-')
	plt.title('3. Optimization Convergence Plot')
	plt.xlabel('Iteration')
	plt.ylabel('Negative REML Log-Likelihood')
	plt.grid(True)
	plt.savefig('3_reml_convergence.png')
	plt.close()
	print("[INFO] Saved '3_reml_convergence.png'")


def fit_lmm_reml(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> tuple['scipy.optimize.OptimizeResult', list]:
	"""
	최적화 알고리즘을 실행하여 REML 기반 LMM 모델을 피팅한다.
	"""
	convergence_history = []

	def callback_fn(params):
		ll = neg_reml_log_likelihood(params, y, X, Z)
		convergence_history.append(ll)

	initial_params = np.log([np.std(y) * 0.5, np.std(y), 1.0])
	optimization_result = minimize(
		fun=neg_reml_log_likelihood,
		x0=initial_params,
		args=(y, X, Z),
		method='L-BFGS-B',
		callback=callback_fn
	)
	return optimization_result, convergence_history


def calculate_model_statistics(opt_result: 'scipy.optimize.OptimizeResult', y: np.ndarray, X: np.ndarray,
                               Z: np.ndarray) -> dict:
	"""
	최적화된 결과로부터 모델의 주요 통계량을 계산한다.
	"""
	n_obs = len(y)
	n_students = Z.shape[1] // 2
	log_stdevs = opt_result.x
	var_e = np.exp(log_stdevs[0]) ** 2
	var_u0 = np.exp(log_stdevs[1]) ** 2
	var_u1 = np.exp(log_stdevs[2]) ** 2

	R_est = np.eye(n_obs) * var_e
	G0_est = np.diag([var_u0, var_u1])
	G_est = np.kron(np.eye(n_students, dtype=float), G0_est)
	V_est = Z @ G_est @ Z.T + R_est
	V_est_inv = np.linalg.inv(V_est)

	Xt_Vinv_X_est = X.T @ V_est_inv @ X
	Xt_Vinv_X_est_inv = np.linalg.inv(Xt_Vinv_X_est)
	beta_est = Xt_Vinv_X_est_inv @ X.T @ V_est_inv @ y
	std_errors = np.sqrt(np.diag(Xt_Vinv_X_est_inv))

	# BLUPs (Best Linear Unbiased Predictors) for random effects
	residuals = y - X @ beta_est
	blups_vec = G_est @ Z.T @ V_est_inv @ residuals
	blups = blups_vec.reshape((n_students, 2))  # Reshape to (n_students, 2) for intercept and slope

	df = n_obs - X.shape[1]
	t_values = beta_est / std_errors
	p_values = [2 * (1 - t.cdf(np.abs(t_val), df=df)) for t_val in t_values]

	return {
		'beta_est': beta_est, 'std_errors': std_errors, 't_values': t_values, 'p_values': p_values,
		'variances': {'residual': var_e, 'intercept': var_u0, 'time': var_u1},
		'blups': blups
	}


def visualize_results(final_results: dict, proc_data: dict):
	"""
	4. 최종 결과 시각화: 예측선 및 분산 성분
	"""
	# 4.1 Fitted Lines Plot
	# FIX: Convert student_names list to numpy array for advanced indexing
	student_names_array = np.array(proc_data['student_names'])
	df = pd.DataFrame({
		'student_id': student_names_array[proc_data['student_indices']],
		'time': proc_data['time'],
		'score': proc_data['y']
	})

	beta_intercept, beta_time = final_results['beta_est']
	blups = final_results['blups']

	plt.figure(figsize=(12, 8))

	# Raw data points
	sns.scatterplot(data=df, x='time', y='score', hue='student_id', s=100, alpha=0.7)

	# Population-level fit (Fixed effects only)
	time_range = np.linspace(df['time'].min(), df['time'].max(), 100)
	plt.plot(time_range, beta_intercept + beta_time * time_range, 'k--', linewidth=3,
	         label='Population Fit (Fixed Effects)')

	# Individual-level fits (Fixed + Random effects)
	for i, student_name in enumerate(proc_data['student_names']):
		student_blup_intercept, student_blup_slope = blups[i]
		individual_intercept = beta_intercept + student_blup_intercept
		individual_slope = beta_time + student_blup_slope
		color = sns.color_palette()[i % len(sns.color_palette())]
		plt.plot(time_range, individual_intercept + individual_slope * time_range, color=color, linewidth=1.5)

	plt.title('4a. Fitted Lines Plot: Population and Individual Predictions')
	plt.xlabel('Time (months)')
	plt.ylabel('Score')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.savefig('4a_fitted_lines_plot.png')
	plt.close()
	print("[INFO] Saved '4a_fitted_lines_plot.png'")

	# 4.2 Bar Chart for Variance Components
	variances = final_results['variances']
	var_df = pd.DataFrame.from_dict(variances, orient='index', columns=['Variance']).reset_index()
	var_df = var_df.rename(columns={'index': 'Component'})

	plt.figure(figsize=(8, 6))
	sns.barplot(data=var_df, x='Component', y='Variance')
	plt.title('4b. Estimated Variance Components')
	plt.ylabel('Variance')
	plt.xlabel('Random Effect Component')
	plt.tight_layout()
	plt.savefig('4b_variance_components.png')
	plt.close()
	print("[INFO] Saved '4b_variance_components.png'")


def print_summary(results: dict):
	"""
	계산된 모델 통계량을 지정된 형식으로 출력한다.
	"""
	print("\n### Pure Python을 이용한 Frequentist LMM 결과 ###\n")
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
	data = {
		'student_id': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
		'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
		'score': [75, 78, 83, 80, 84, 85, 70, 72, 73]
	}

	# 1. 데이터 전처리 및 시각화
	proc_data = preprocess_data(data)
	visualize_raw_data(proc_data)
	y, n_obs, n_students = proc_data['y'], proc_data['n_obs'], proc_data['n_students']

	# 2. 디자인 행렬 생성 및 시각화
	X, Z = build_design_matrices(proc_data['time'], proc_data['student_indices'], n_obs, n_students)
	visualize_design_matrices(X, Z)

	# 3. 모델 피팅 및 최적화 과정 시각화
	optimization_result, history = fit_lmm_reml(y, X, Z)
	visualize_convergence(history)

	# 4. 최종 통계량 계산 및 결과 출력/시각화
	if optimization_result.success:
		final_results = calculate_model_statistics(optimization_result, y, X, Z)
		print_summary(final_results)
		visualize_results(final_results, proc_data)
	else:
		print("모델 최적화에 실패했습니다.")
		print(optimization_result.message)