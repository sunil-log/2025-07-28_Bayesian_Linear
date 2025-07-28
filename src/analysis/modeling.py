# src/analysis/modeling.py
import numpy as np
from scipy.optimize import minimize

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