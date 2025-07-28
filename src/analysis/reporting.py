# src/analysis/reporting.py
import numpy as np
from scipy.stats import t

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

	residuals = y - X @ beta_est
	blups_vec = G_est @ Z.T @ V_est_inv @ residuals
	blups = blups_vec.reshape((n_students, 2))

	df = n_obs - X.shape[1]
	t_values = beta_est / std_errors
	p_values = [2 * (1 - t.cdf(np.abs(t_val), df=df)) for t_val in t_values]

	return {
		'beta_est': beta_est, 'std_errors': std_errors, 't_values': t_values, 'p_values': p_values,
		'variances': {'residual': var_e, 'intercept': var_u0, 'time': var_u1},
		'blups': blups
	}

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